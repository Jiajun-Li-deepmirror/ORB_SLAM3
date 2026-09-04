import numpy as np

from splg_slam.features.splg import SPLG
from splg_slam.geometry.pnp import solve_pnp_ransac
from splg_slam.geometry.pose_utils import camera_center
from splg_slam.geometry.stereo import StereoRectifier
from splg_slam.localization.relocalizer import Relocalizer
from splg_slam.map.world_map import WorldMap


def estimate_max_track_jump_m(
    world_map: WorldMap, query_dt_s: float, percentile: float = 99.9, safety_factor: float = 6.0,
) -> float:
    """Derives a plausible per-query motion bound directly from the map's own keyframe
    trajectory, instead of a manually-tuned constant: computes the empirical speed between
    consecutive keyframes (robust to a few genuinely fast keyframe-to-keyframe motions via
    a high percentile rather than the max), then scales by the ACTUAL time between two
    consecutively QUERIED frames (`query_dt_s` - depends on the dataset's frame rate and
    --stride, not just the dataset). This is what lets the same check self-calibrate for a
    slow drone vs a fast car instead of needing a separate hand-picked value per dataset -
    on KITTI (a car), reusing an EuRoC-tuned constant here rejected 27% of genuine motion as
    implausible.

    `safety_factor` is well above 1x because keyframe-to-keyframe speed is an AVERAGE over
    the interval between them - keyframes are only inserted once accumulated motion crosses
    a threshold, so this systematically understates true instantaneous per-frame speed
    (verified on MH01: at 3x it undershot the real 99.9th-percentile per-query motion by
    about 2x). Tuned against both ends observed in practice: too small (3x) produced false-
    positive rejections of genuine motion on MH01; too large (10x) on KITTI once let a real
    ~12m single-frame teleport through uncaught (the derived bound landed just above it).
    6x cleanly separates both: comfortably above normal per-query motion on either dataset,
    comfortably below the smallest confirmed-bad jump seen on either."""
    kf_ids = world_map.keyframe_ids_sorted()
    kfs = [world_map.keyframes[i] for i in kf_ids]
    centers = np.array([camera_center(kf.pose_cw) for kf in kfs])
    ts_s = np.array([kf.timestamp_ns for kf in kfs]) * 1e-9
    dt = np.diff(ts_s)
    dist = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    valid = dt > 1e-6
    speeds = dist[valid] / dt[valid]
    speed_bound = float(np.percentile(speeds, percentile))
    return speed_bound * query_dt_s * safety_factor


class ContinuousLocalizer:
    """Localization-only analog of mapping/tracker.OfflineMapper's tracking loop: cheap
    frame-to-reference-keyframe tracking (LightGlue match against one cached keyframe's
    features + PnP) on every frame, falling back to Relocalizer's expensive full-map
    retrieval query only when that tracking is lost. Unlike OfflineMapper, this never
    inserts a keyframe or map point - the map (`world_map`) is read-only throughout.

    This mirrors how real AMR/robot localization stacks separate a cheap, high-frequency
    tracking step (odometry-like: relate this frame to the last known one) from a rare,
    expensive global relocalization step used only for bootstrap or recovery after
    tracking loss - instead of paying for a full global-descriptor retrieval + multi-
    candidate LightGlue sweep on every single frame, as a from-scratch Relocalizer call
    per frame does.
    """

    def __init__(
        self, world_map: WorldMap, rectifier: StereoRectifier, cfg,
        max_track_jump_m: float | None = None, force_relocalize_every: int | None = None,
    ):
        """`max_track_jump_m`: rejects a track-mode PnP solution whose camera center lands
        further than this from the immediately preceding successful localization (regardless
        of mode), treating it as tracking failure (falls through to relocalization) instead
        of a valid frame. Frame-to-reference-keyframe PnP with no such check occasionally
        accepts a "valid" (enough inliers, passes RANSAC) but physically implausible solve -
        a degenerate point configuration can do this even with good matches - which then
        looks like a single-frame teleport that snaps back the very next query. Set relative
        to how much a real target could plausibly move between two CONSECUTIVE queried
        frames at your actual query stride - the default here (None) leaves that check off,
        since there's no dataset-independent value safe to assume by default.

        `force_relocalize_every`: after this many CONSECUTIVE successful track-mode calls
        anchored to the same reference keyframe, force a full relocalization instead of
        continuing to track, even though tracking hasn't failed. Frame-to-single-keyframe
        PnP has no cross-view consistency check, so a run of visually ambiguous frames
        (repetitive structure the matcher confuses) can produce a "valid" (enough inliers,
        passes RANSAC) but systematically WRONG pose that stays self-consistent - smoothly
        varying frame-to-frame, so max_track_jump_m's single-step discontinuity check never
        fires - until the reference keyframe happens to change. Forcing a fresh, independent
        global-retrieval re-check periodically bounds how long any one bad anchor can persist
        undetected, at the cost of paying for an extra expensive relocalization periodically
        even when tracking looks fine. None (default) disables this."""
        self.world_map = world_map
        self.rectifier = rectifier
        self.cfg = cfg
        self.max_track_jump_m = max_track_jump_m
        self.force_relocalize_every = force_relocalize_every
        self.relocalizer = Relocalizer(world_map, rectifier, cfg)
        self.splg = self.relocalizer.splg  # share one SuperPoint+LightGlue instance
        self.ref_kf_id: int | None = None
        self.last_pos: np.ndarray | None = None
        self.track_streak = 0  # consecutive successful track-mode calls since the last relocalize
        self.n_track = 0
        self.n_relocalize = 0
        self.n_track_lost = 0
        self.n_track_jump_rejected = 0
        self.n_forced_relocalize = 0

    def _track_against_ref(self, query_feats: dict, kpts_q: np.ndarray):
        """Tries to extend tracking from self.ref_kf_id using already-extracted query
        features. Returns (pose_cw, num_inliers) or None."""
        ref_feats = self.relocalizer._feats_for_keyframe(self.ref_kf_id)
        matches = self.splg.match(ref_feats, query_feats)["matches"]
        ref_kf = self.world_map.keyframes[self.ref_kf_id]

        obj_pts, img_pts = [], []
        for ref_i, q_i in matches:
            mp_id = ref_kf.map_point_ids[ref_i]
            if mp_id >= 0:
                obj_pts.append(self.world_map.map_points[mp_id].position)
                img_pts.append(kpts_q[q_i])

        if len(obj_pts) < self.cfg.tracking.min_inlier_matches:
            return None
        ok, pose_cw, inlier_mask = solve_pnp_ransac(
            np.asarray(obj_pts), np.asarray(img_pts), self.rectifier.K_rect,
            reproj_threshold_px=self.cfg.tracking.pnp_reproj_threshold_px,
        )
        if not ok or inlier_mask.sum() < self.cfg.tracking.min_inlier_matches:
            return None
        return pose_cw, int(inlier_mask.sum())

    def localize(self, rect_img_left: np.ndarray) -> tuple[bool, np.ndarray | None, dict]:
        query_feats = self.splg.extract(rect_img_left)
        kpts_q, _ = SPLG.to_frame_arrays(query_feats)

        force_refresh = (
            self.force_relocalize_every is not None and self.track_streak >= self.force_relocalize_every
        )
        if force_refresh:
            self.n_forced_relocalize += 1
            ok, pose_cw, info = self.relocalizer.localize(rect_img_left, query_feats=query_feats)
            if ok:
                self.ref_kf_id = info["candidate_kf"]
                self.last_pos = camera_center(pose_cw)
                self.track_streak = 0
                self.n_relocalize += 1
                info = dict(info)
                info["mode"] = "relocalize"
                return True, pose_cw, info
            # forced refresh found nothing better - fall through and keep trusting the
            # existing anchor rather than discarding a track state that wasn't actually lost

        if self.ref_kf_id is not None:
            result = self._track_against_ref(query_feats, kpts_q)
            if result is not None:
                pose_cw, num_inliers = result
                jump_m = None
                if self.max_track_jump_m is not None and self.last_pos is not None:
                    jump_m = float(np.linalg.norm(camera_center(pose_cw) - self.last_pos))
                if jump_m is None or jump_m <= self.max_track_jump_m:
                    self.n_track += 1
                    self.track_streak += 1
                    self.last_pos = camera_center(pose_cw)
                    return True, pose_cw, {"mode": "track", "ref_kf": self.ref_kf_id, "num_inliers": num_inliers}
                self.n_track_jump_rejected += 1
            self.n_track_lost += 1

        ok, pose_cw, info = self.relocalizer.localize(rect_img_left, query_feats=query_feats)
        if ok:
            self.ref_kf_id = info["candidate_kf"]
            self.last_pos = camera_center(pose_cw)
            self.track_streak = 0
            self.n_relocalize += 1
            info = dict(info)
            info["mode"] = "relocalize"
            return True, pose_cw, info

        self.ref_kf_id = None
        self.track_streak = 0
        return False, None, info
