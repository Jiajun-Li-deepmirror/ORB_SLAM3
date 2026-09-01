import cv2
import gtsam
import numpy as np

from splg_slam.data.euroc import ImuCalibration
from splg_slam.features.splg import SPLG
from splg_slam.geometry.pnp import solve_pnp_ransac
from splg_slam.geometry.pose_utils import pose_delta
from splg_slam.geometry.stereo import StereoDepthEstimator, StereoRectifier
from splg_slam.localization.retrieval import GlobalDescriptorIndex
from splg_slam.map.frame import Frame
from splg_slam.map.keyframe import KeyFrame
from splg_slam.map.world_map import WorldMap
from splg_slam.mapping.gtsam_utils import pose_cw_to_body_gtsam
from splg_slam.mapping.imu_init import choose_imu_init_mode
from splg_slam.mapping.imu_preintegration import (
    bias_from_vector,
    gravity_alignment_rotation,
    make_preintegration_params,
    preintegrate,
)
from splg_slam.mapping.local_map import gather_local_map_point_ids, search_local_map


class OfflineMapper:
    """Sequential stereo mapping: track each new frame against the last keyframe via
    LightGlue + PnP to get an initial pose, then expand correspondences by projecting
    the local map (map points seen by a window of recent keyframes) into the frame and
    matching by descriptor nearest-neighbor within a pixel radius, then re-solves PnP on
    the combined set. Inserts a new keyframe once the motion or tracked-point ratio
    crosses a threshold. If regular tracking fails outright, falls back to relocalizing
    against the whole map (global-descriptor retrieval + LightGlue + PnP) before giving
    up on the frame - recovers from a real tracking loss where the reference keyframe's
    viewpoint has drifted too far to ever re-match directly, instead of just hoping the
    next frame happens to land somewhere matchable."""

    def __init__(
        self, cfg, rectifier: StereoRectifier, global_extractor=None,
        imu_measurements: np.ndarray | None = None, imu_calib: ImuCalibration | None = None,
    ):
        self.cfg = cfg
        self.rectifier = rectifier
        self.depth_est = StereoDepthEstimator(
            rectifier,
            min_disp=cfg.stereo.min_disp,
            num_disp=cfg.stereo.num_disp,
            block_size=cfg.stereo.block_size,
        )
        self.splg = SPLG(max_keypoints=cfg.features.max_keypoints)
        self.global_extractor = global_extractor
        self.world_map = WorldMap()

        self.ref_keyframe: KeyFrame | None = None
        self.ref_feats: dict | None = None
        self.last_frame: Frame | None = None
        self._next_frame_id = 0
        self.n_keyframes_inserted = 0
        self.n_relocalizations = 0
        self.track_stats: list[dict] = []

        self.reloc_index = GlobalDescriptorIndex() if global_extractor is not None else None
        self._reloc_feats_cache: dict[int, dict] = {}

        self.imu_enabled = bool(
            getattr(cfg, "imu", None) and cfg.imu.enabled and imu_measurements is not None and imu_calib is not None
        )
        self.imu_measurements = imu_measurements
        self.imu_calib = imu_calib
        self._imu_params = (
            make_preintegration_params(imu_calib, cfg.imu.gravity_norm, cfg.imu.integration_sigma)
            if self.imu_enabled else None
        )
        self._imu_last_ts_ns: int | None = None
        self._pending_imu_samples: list[np.ndarray] = []
        self.imu_init_mode: str | None = None  # "static" | "dynamic", decided at bootstrap
        self.imu_init_pending = False  # True while dynamic init still needs to run (see build_map.py)

    def _pull_imu_samples(self, timestamp_ns: int) -> None:
        if self._imu_last_ts_ns is None:
            self._imu_last_ts_ns = timestamp_ns
            return
        ts_col = self.imu_measurements[:, 0]
        start = np.searchsorted(ts_col, self._imu_last_ts_ns, side="right")
        end = np.searchsorted(ts_col, timestamp_ns, side="right")
        if end > start:
            self._pending_imu_samples.append(self.imu_measurements[start:end])
        self._imu_last_ts_ns = timestamp_ns

    def process_stereo_pair(self, img_left: np.ndarray, img_right: np.ndarray,
                             timestamp_ns: int, image_path: str | None = None):
        """Returns (frame_or_keyframe, is_keyframe). frame_or_keyframe is None if tracking failed."""
        rect_l, rect_r = self.rectifier.rectify(img_left, img_right)

        feats = self.splg.extract(rect_l)
        kpts, desc = SPLG.to_frame_arrays(feats)

        disp = self.depth_est.compute_disparity(rect_l, rect_r)
        depths = self.depth_est.depths_at_points(disp, kpts)
        depths[depths > self.cfg.stereo.max_depth_m] = np.nan

        frame_id = self._next_frame_id
        self._next_frame_id += 1

        if self.imu_enabled:
            self._pull_imu_samples(timestamp_ns)

        if self.ref_keyframe is None:
            pose_cw0 = np.eye(4)
            if self.imu_enabled:
                # World frame is defined by this bootstrap keyframe, so its orientation
                # fixes gravity's direction in world frame for every later IMU factor.
                # EuRoC sequences are typically handled/moved before takeoff (not static),
                # so check the actual leading IMU data instead of assuming either case.
                self.imu_init_mode, accel_samples = choose_imu_init_mode(
                    self.imu_measurements, self.cfg.imu.init_static_samples,
                    search_samples=10 * self.cfg.imu.init_static_samples,
                    gyro_static_threshold=self.cfg.imu.init_gyro_static_threshold,
                )
                if self.imu_init_mode == "static":
                    r_world_body0 = gravity_alignment_rotation(accel_samples)
                    r_cam0_body = self.imu_calib.T_cam0_body[:3, :3]
                    pose_cw0[:3, :3] = r_cam0_body @ r_world_body0.T
                else:
                    # Motion-based (ORB-SLAM-style) init: bootstrap at an arbitrary,
                    # gravity-unaware orientation and let run_dynamic_imu_init (called from
                    # build_map.py once enough keyframes/IMU data have accumulated) solve
                    # gyro bias + gravity + velocities and retroactively re-align everything.
                    self.imu_init_pending = True
            kf = KeyFrame(
                frame_id=frame_id, timestamp_ns=timestamp_ns,
                keypoints=kpts, descriptors=desc, depths=depths,
                pose_cw=pose_cw0, image_path=str(image_path) if image_path else None,
                velocity=np.zeros(3) if self.imu_enabled else None,
                imu_bias=np.zeros(6) if self.imu_enabled else None,
            )
            self._insert_keyframe(kf, feats, rect_l)
            self.last_frame = kf
            return kf, True

        match = self.splg.match(self.ref_feats, feats)
        matches = match["matches"]

        obj_pts, img_pts, point_ids, kp_idx_list = [], [], [], []
        for ref_i, cur_i in matches:
            mp_id = self.ref_keyframe.map_point_ids[ref_i]
            if mp_id >= 0:
                obj_pts.append(self.world_map.map_points[mp_id].position)
                img_pts.append(kpts[cur_i])
                point_ids.append(int(mp_id))
                kp_idx_list.append(int(cur_i))

        if len(obj_pts) < self.cfg.tracking.min_inlier_matches:
            return self._on_tracking_failure(rect_l, feats, kpts, desc, depths, frame_id, timestamp_ns)

        ok, pose_init, _ = solve_pnp_ransac(
            np.asarray(obj_pts), np.asarray(img_pts), self.rectifier.K_rect,
            reproj_threshold_px=self.cfg.tracking.pnp_reproj_threshold_px,
        )
        if not ok:
            return self._on_tracking_failure(rect_l, feats, kpts, desc, depths, frame_id, timestamp_ns)

        used_kp_mask = np.zeros(len(kpts), dtype=bool)
        used_kp_mask[kp_idx_list] = True

        window = self.world_map.covisible_window(
            self.ref_keyframe.frame_id, window_size=self.cfg.tracking.local_map_window_size,
            min_shared=self.cfg.tracking.local_map_min_shared,
        )
        local_point_ids = gather_local_map_point_ids(self.world_map, window, exclude=set(point_ids))
        lm_obj, lm_img, lm_pids, lm_kpidx = search_local_map(
            self.world_map, local_point_ids, pose_init, kpts, desc, self.rectifier.K_rect,
            image_size=(rect_l.shape[1], rect_l.shape[0]), used_kp_mask=used_kp_mask,
            radius_px=self.cfg.tracking.local_map_radius_px,
            desc_threshold=self.cfg.tracking.local_map_desc_threshold,
        )

        obj_pts = np.asarray(obj_pts + lm_obj)
        img_pts = np.asarray(img_pts + lm_img)
        point_ids = point_ids + lm_pids
        kp_idx_list = kp_idx_list + lm_kpidx

        ok, pose_cw, inlier_mask = solve_pnp_ransac(
            obj_pts, img_pts, self.rectifier.K_rect,
            reproj_threshold_px=self.cfg.tracking.pnp_reproj_threshold_px,
        )
        if not ok or inlier_mask.sum() < self.cfg.tracking.min_inlier_matches:
            return self._on_tracking_failure(rect_l, feats, kpts, desc, depths, frame_id, timestamp_ns)

        translation, rotation_deg = pose_delta(self.ref_keyframe.pose_cw, pose_cw)
        if (
            translation > self.cfg.tracking.max_pose_jump_m
            or rotation_deg > self.cfg.tracking.max_pose_jump_deg
        ):
            # A "valid" (enough inliers, low reprojection error) but physically
            # implausible PnP solution - e.g. a near-degenerate point configuration in a
            # repetitive scene. Silently accepting this creates a keyframe at a nonsense
            # position that every later frame then tracks forward from. Treat it as a
            # tracking failure instead of a keyframe.
            return self._on_tracking_failure(rect_l, feats, kpts, desc, depths, frame_id, timestamp_ns)

        frame = Frame(
            frame_id=frame_id, timestamp_ns=timestamp_ns,
            keypoints=kpts, descriptors=desc, depths=depths, pose_cw=pose_cw,
        )
        self.last_frame = frame

        tracked_ratio = int(inlier_mask.sum()) / max(len(kpts), 1)

        need_new_kf = (
            translation > self.cfg.tracking.keyframe_translation_m
            or rotation_deg > self.cfg.tracking.keyframe_rotation_deg
            or tracked_ratio < self.cfg.tracking.keyframe_min_tracked_ratio
        )
        self.track_stats.append({
            "frame_id": frame_id, "translation": translation, "rotation_deg": rotation_deg,
            "tracked_ratio": tracked_ratio, "num_matched_with_mp": len(obj_pts),
            "num_local_map_matches": len(lm_obj),
            "num_inliers": int(inlier_mask.sum()), "total_kpts": len(kpts),
        })
        if not need_new_kf:
            return frame, False

        kf = KeyFrame(
            frame_id=frame_id, timestamp_ns=timestamp_ns,
            keypoints=kpts, descriptors=desc, depths=depths, pose_cw=pose_cw,
            image_path=str(image_path) if image_path else None,
        )
        for pid, kp_i, is_inlier in zip(point_ids, kp_idx_list, inlier_mask):
            if is_inlier:
                kf.map_point_ids[kp_i] = pid
        self._insert_keyframe(kf, feats, rect_l)
        return kf, True

    def _insert_keyframe(self, kf: KeyFrame, feats: dict, rect_left_img: np.ndarray) -> None:
        if self.imu_enabled and self.ref_keyframe is not None and self._pending_imu_samples:
            samples = np.concatenate(self._pending_imu_samples, axis=0)
            prev_bias_vec = self.ref_keyframe.imu_bias
            prev_bias = bias_from_vector(prev_bias_vec)
            preint = preintegrate(samples, prev_bias, self._imu_params)
            prev_body_pose = pose_cw_to_body_gtsam(self.ref_keyframe.pose_cw, self.imu_calib.T_cam0_body)
            prev_velocity = self.ref_keyframe.velocity if self.ref_keyframe.velocity is not None else np.zeros(3)
            prev_state = gtsam.NavState(prev_body_pose, prev_velocity)
            predicted = preint.predict(prev_state, prev_bias)
            kf.velocity = predicted.velocity()
            kf.imu_bias = prev_bias_vec if prev_bias_vec is not None else np.zeros(6)
            self.world_map.add_imu_factor(self.ref_keyframe.frame_id, kf.frame_id, samples)
        self._pending_imu_samples = []

        if self.global_extractor is not None:
            kf.global_descriptor = self.global_extractor.extract(rect_left_img)

        self.world_map.add_keyframe(kf)
        self.n_keyframes_inserted += 1

        for kp_idx, mp_id in enumerate(kf.map_point_ids):
            if mp_id >= 0:
                self.world_map.add_observation(int(mp_id), kf.frame_id, kp_idx)

        valid = kf.valid_depth_mask() & (kf.map_point_ids < 0)
        valid_idx = np.nonzero(valid)[0]
        if len(valid_idx) > 0:
            pts_cam = self.rectifier.backproject(kf.keypoints[valid_idx], kf.depths[valid_idx])
            pose_wc = kf.pose_wc()
            pts_world = (pose_wc[:3, :3] @ pts_cam.T).T + pose_wc[:3, 3]
            for local_i, kp_idx in enumerate(valid_idx):
                point_id = self.world_map.new_map_point(
                    pts_world[local_i], kf.descriptors[kp_idx], created_at_kf_count=self.n_keyframes_inserted,
                )
                self.world_map.add_observation(point_id, kf.frame_id, kp_idx)

        if self.cfg.mapping.point_probation_kf > 0:
            self.world_map.cull_probationary_points(
                self.n_keyframes_inserted,
                probation_kf=self.cfg.mapping.point_probation_kf,
                min_observations=self.cfg.mapping.point_min_observations,
            )

        if self.reloc_index is not None:
            self.reloc_index.build(self.world_map)

        self.ref_keyframe = kf
        self.ref_feats = feats

    def _cached_keyframe_feats(self, kf_id: int) -> dict:
        if kf_id not in self._reloc_feats_cache:
            kf = self.world_map.keyframes[kf_id]
            img = cv2.imread(kf.image_path, cv2.IMREAD_GRAYSCALE)
            rect_l, _ = self.rectifier.rectify(img, img)
            self._reloc_feats_cache[kf_id] = self.splg.extract(rect_l)
        return self._reloc_feats_cache[kf_id]

    def _try_relocalize(self, rect_l: np.ndarray, feats: dict, kpts: np.ndarray):
        """Global-descriptor retrieval + LightGlue + PnP against the whole map, same
        approach as the standalone Relocalizer used for query-time localization. Returns
        (candidate_keyframe, candidate_feats, num_inliers, pose_cw) or None."""
        if self.reloc_index is None or not self.world_map.keyframes:
            return None

        query_global = self.global_extractor.extract(rect_l)
        candidates = self.reloc_index.query(query_global, top_k=self.cfg.tracking.relocalization_top_k)

        best = None
        for kf_id, _sim in candidates:
            cand_kf = self.world_map.keyframes[kf_id]
            cand_feats = self._cached_keyframe_feats(kf_id)
            matches = self.splg.match(cand_feats, feats)["matches"]

            obj_pts, img_pts = [], []
            for cand_i, cur_i in matches:
                mp_id = cand_kf.map_point_ids[cand_i]
                if mp_id >= 0:
                    obj_pts.append(self.world_map.map_points[mp_id].position)
                    img_pts.append(kpts[cur_i])

            if len(obj_pts) < self.cfg.tracking.min_inlier_matches:
                continue
            ok, pose_cw, inlier_mask = solve_pnp_ransac(
                np.asarray(obj_pts), np.asarray(img_pts), self.rectifier.K_rect,
                reproj_threshold_px=self.cfg.tracking.pnp_reproj_threshold_px,
            )
            if not ok:
                continue

            n_inliers = int(inlier_mask.sum())
            if best is None or n_inliers > best[2]:
                best = (cand_kf, cand_feats, n_inliers, pose_cw)

        if best is None or best[2] < self.cfg.tracking.min_inlier_matches:
            return None
        return best

    def _on_tracking_failure(self, rect_l, feats, kpts, desc, depths, frame_id, timestamp_ns):
        reloc = self._try_relocalize(rect_l, feats, kpts)
        if reloc is None:
            return None, False

        cand_kf, cand_feats, _n_inliers, pose_cw = reloc
        self.n_relocalizations += 1
        if self.imu_enabled:
            # cand_kf is not temporally adjacent to the old ref_keyframe (reloc jumps
            # non-locally in the map) - the interval we were accumulating no longer
            # corresponds to a real between-keyframe motion, so it can't become a factor.
            self._pending_imu_samples = []
            if cand_kf.imu_bias is None:
                cand_kf.imu_bias = (
                    self.ref_keyframe.imu_bias
                    if self.ref_keyframe is not None and self.ref_keyframe.imu_bias is not None
                    else np.zeros(6)
                )
        frame = Frame(
            frame_id=frame_id, timestamp_ns=timestamp_ns,
            keypoints=kpts, descriptors=desc, depths=depths, pose_cw=pose_cw,
        )
        self.last_frame = frame
        self.ref_keyframe = cand_kf
        self.ref_feats = cand_feats
        return frame, False
