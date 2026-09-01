import cv2
import numpy as np

from splg_slam.features.splg import SPLG
from splg_slam.geometry.pnp import solve_pnp_ransac
from splg_slam.geometry.pose_utils import pose_delta
from splg_slam.geometry.stereo import StereoDepthEstimator, StereoRectifier
from splg_slam.localization.retrieval import GlobalDescriptorIndex
from splg_slam.map.frame import Frame
from splg_slam.map.keyframe import KeyFrame
from splg_slam.map.world_map import WorldMap
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

    def __init__(self, cfg, rectifier: StereoRectifier, global_extractor=None):
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

        if self.ref_keyframe is None:
            kf = KeyFrame(
                frame_id=frame_id, timestamp_ns=timestamp_ns,
                keypoints=kpts, descriptors=desc, depths=depths,
                pose_cw=np.eye(4), image_path=str(image_path) if image_path else None,
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
        frame = Frame(
            frame_id=frame_id, timestamp_ns=timestamp_ns,
            keypoints=kpts, descriptors=desc, depths=depths, pose_cw=pose_cw,
        )
        self.last_frame = frame
        self.ref_keyframe = cand_kf
        self.ref_feats = cand_feats
        return frame, False
