import gtsam
import numpy as np
from gtsam import (
    BetweenFactorPose3,
    Cal3_S2,
    CombinedImuFactor,
    GenericProjectionFactorCal3_S2,
    Point2,
    Point3,
    PriorFactorConstantBias,
    PriorFactorPoint3,
    PriorFactorPose3,
    PriorFactorVector,
)
from gtsam import Symbol as _Symbol
from gtsam.symbol_shorthand import B, L, V, X

from splg_slam.features.splg import SPLG
from splg_slam.geometry.pnp import solve_pnp_ransac
from splg_slam.geometry.pose_utils import camera_center
from splg_slam.geometry.stereo import StereoRectifier
from splg_slam.localization.relocalizer import Relocalizer
from splg_slam.map.world_map import WorldMap
from splg_slam.mapping.gtsam_utils import (
    gtsam_pose_to_cw,
    matrix_to_gtsam_pose3,
    pose_cw_to_body_gtsam,
    pose_cw_to_gtsam,
)
from splg_slam.mapping.imu_preintegration import bias_from_vector, preintegrate


# Not a gtsam builtin shorthand - mirrors mapping/local_ba.py's own Y(): the IMU body-pose
# symbol, kept distinct from X() (camera pose, used by the reprojection factors below)
# since GenericProjectionFactorCal3_S2 factors here are built against camera poses exactly
# like local_ba.py's, for the same reason (see local_ba.py's own comment on this).
def Y(j: int) -> int:
    return _Symbol("y", j).key()


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
        imu_calib=None, imu_params=None, imu_measurements: np.ndarray | None = None,
    ):
        """`imu_calib`/`imu_params` (both None by default = IMU disabled, original
        behavior): if given, `localize()` gains a THIRD fallback tier below track-mode and
        relocalize-mode - when both fail, instead of reporting "lost" (ok=False, no pose
        at all), it dead-reckons forward from the last real (track/relocalize) anchor's own
        pose/velocity/bias via real IMU preintegration (same math as
        tracker.py's _imu_bridge_pose_cw), returning mode="imu_only". This never resets
        _consecutive-loss state the way relocalize would - it just keeps predicting one
        frame at a time, chaining forward, until a real anchor is found again - so a query
        session that starts with a real localization can never afterward report "lost"
        outright, matching a hard requirement for continuous positioning (a robot control
        loop always needs *some* pose, even an uncertain one, never nothing).

        Every call to localize() (regardless of mode) is appended to self.session_log, and
        every real (track/relocalize) frame's own 2D-3D correspondences are recorded too -
        together these let smooth_session_trajectory() run ONE fixed-map factor-graph
        optimization at the end of a run, retroactively correcting every imu_only frame's
        pose to be physically consistent with the real, map-anchored frames on both sides
        of it (the live/streaming pose during an imu_only stretch is allowed to be
        approximate; only the logged, corrected trajectory needs to be accurate).

        `max_track_jump_m`: rejects a track-mode PnP solution whose camera center lands
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
        self.n_imu_only = 0

        self.imu_calib = imu_calib
        self.imu_params = imu_params
        self.imu_measurements = imu_measurements
        self.imu_enabled = imu_calib is not None and imu_params is not None and imu_measurements is not None
        # (pose_cw, velocity, bias_vec, timestamp_ns) of the most recent REAL (track or
        # relocalize) frame - the anchor an imu_only chain dead-reckons forward from, and
        # the state carried forward frame-to-frame while imu_only (so a chain of several
        # consecutive imu_only frames keeps integrating from where the last one left off,
        # not from the original anchor's samples re-integrated from scratch each time).
        self.last_real_state: tuple | None = None
        self.last_state: tuple | None = None  # same shape, updated every frame incl. imu_only
        # (timestamp_ns, position) for the last few REAL (track/relocalize) anchors only -
        # see _estimate_velocity's docstring for why imu_only positions are excluded here.
        self.recent_real_positions: list[tuple[int, np.ndarray]] = []
        self._recent_real_positions_max = 10
        self._velocity_fit_max_age_s = 2.0
        # Full per-frame record for the end-of-session fixed-map smoothing BA (see
        # smooth_session_trajectory below): one entry per localize() call, in order.
        self.session_log: list[dict] = []
        self.n_forced_relocalize = 0

    def _track_against_ref(self, query_feats: dict, kpts_q: np.ndarray):
        """Tries to extend tracking from self.ref_kf_id using already-extracted query
        features. Returns (pose_cw, num_inliers, correspondences) or None, where
        `correspondences` is a list of (map_point_id, image_point) pairs for every PnP
        inlier - consumed by session_log/smooth_session_trajectory (see localize() and the
        module-level smooth_session_trajectory below)."""
        ref_feats = self.relocalizer._feats_for_keyframe(self.ref_kf_id)
        matches = self.splg.match(ref_feats, query_feats)["matches"]
        ref_kf = self.world_map.keyframes[self.ref_kf_id]

        obj_pts, img_pts, mp_ids = [], [], []
        for ref_i, q_i in matches:
            mp_id = ref_kf.map_point_ids[ref_i]
            if mp_id >= 0:
                obj_pts.append(self.world_map.map_points[mp_id].position)
                img_pts.append(kpts_q[q_i])
                mp_ids.append(int(mp_id))

        if len(obj_pts) < self.cfg.tracking.min_inlier_matches:
            return None
        ok, pose_cw, inlier_mask = solve_pnp_ransac(
            np.asarray(obj_pts), np.asarray(img_pts), self.rectifier.K_rect,
            reproj_threshold_px=self.cfg.tracking.pnp_reproj_threshold_px,
        )
        if not ok or inlier_mask.sum() < self.cfg.tracking.min_inlier_matches:
            return None
        correspondences = [(mp_ids[j], img_pts[j]) for j in range(len(mp_ids)) if inlier_mask[j]]
        return pose_cw, int(inlier_mask.sum()), correspondences

    def _pull_imu_samples(self, start_ts_ns: int, end_ts_ns: int) -> np.ndarray | None:
        """Slices self.imu_measurements for the raw rows strictly between two frame
        timestamps - mirrors tracker.py's `_pull_imu_samples`/`_imu_bridge_pose_cw` pattern
        exactly (same slicing, same preintegrate() call), just inlined here instead of
        accumulating a pending-samples list across calls, since continuous_localizer only
        ever needs "the samples since the reference frame", never a longer backlog."""
        if not self.imu_enabled:
            return None
        ts_col = self.imu_measurements[:, 0]
        start = np.searchsorted(ts_col, start_ts_ns, side="right")
        end = np.searchsorted(ts_col, end_ts_ns, side="right")
        if end <= start + 1:
            return None
        return self.imu_measurements[start:end]

    def _dead_reckon_imu(self, timestamp_ns: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Third fallback tier: predicts this frame's pose by integrating real IMU samples
        forward from self.last_state (the last frame's pose/velocity/bias, whichever mode
        produced it - real or itself imu_only, so a run of imu_only frames chains forward
        correctly instead of re-integrating from the original anchor every time). Returns
        (pose_cw, velocity) or None if IMU is disabled or there's nothing to integrate from
        yet (e.g. the very first frame of a session can never be imu_only)."""
        if not self.imu_enabled or self.last_state is None:
            return None
        prev_pose_cw, prev_velocity, prev_bias_vec, prev_ts_ns = self.last_state
        samples = self._pull_imu_samples(prev_ts_ns, timestamp_ns) if prev_ts_ns is not None else None
        if samples is None:
            return None
        bias = bias_from_vector(prev_bias_vec)
        preint = preintegrate(samples, bias, self.imu_params)
        body_pose = pose_cw_to_body_gtsam(prev_pose_cw, self.imu_calib.T_cam0_body)
        predicted = preint.predict(gtsam.NavState(body_pose, prev_velocity), bias)
        t_cam_body = matrix_to_gtsam_pose3(self.imu_calib.T_cam0_body)
        cam_wc = predicted.pose().compose(t_cam_body.inverse())
        return gtsam_pose_to_cw(cam_wc), np.array(predicted.velocity())

    def _kf_bias(self, kf_id: int) -> np.ndarray | None:
        """The given keyframe's own mapping-time-solved IMU bias (if the map was built with
        tight_fusion), or None if unavailable - seeds a real anchor's bias_vec with a real
        estimate instead of the zero-bias assumption _log_and_advance otherwise falls back
        to. Matters a lot in practice: measured on MH01, mapping's own solved gyro bias sat
        at a near-constant ~0.08 rad/s (~4.7deg/s) throughout the whole sequence - dead-
        reckoning through even a few seconds with that left uncompensated lets attitude
        drift leak gravity into a large spurious horizontal acceleration (g*sin(drift)),
        which is what actually dominated the multi-meter error measured on a 5s all-imu_only
        gap, not the smaller, directly-integrated accel-bias term."""
        kf = self.world_map.keyframes.get(kf_id)
        return kf.imu_bias if kf is not None and kf.imu_bias is not None else None

    def _update_recent_real_positions(self, timestamp_ns: int | None, pos: np.ndarray) -> None:
        if timestamp_ns is None:
            return
        self.recent_real_positions.append((timestamp_ns, pos))
        if len(self.recent_real_positions) > self._recent_real_positions_max:
            self.recent_real_positions.pop(0)

    def _estimate_velocity(self, pose_cw: np.ndarray, timestamp_ns: int | None) -> np.ndarray:
        """Least-squares linear-velocity fit over the last few seconds of REAL
        (track/relocalize) anchor positions - including this one - used only to seed IMU
        prediction if a LATER frame needs to dead-reckon off this one (real anchors have no
        proper velocity solve of their own; unlike mapping's tracker.py, there's no factor
        graph running live here). Deliberately fits ONLY against self.recent_real_positions
        (never imu_only positions): the naive version of this used to diff against
        self.last_state regardless of mode, which meant the very first real anchor right
        after an imu_only stretch had its velocity computed against that stretch's OWN
        (potentially off-by-meters, per the 5s-noise-gap measurement) dead-reckoned
        position - seeding the very next imu_only chain with a velocity error inherited
        from the last one. A window (default last 10 real anchors within 2s) also averages
        out per-frame PnP position noise better than a raw 2-point difference would.
        Falls back to zero with fewer than 2 real anchors in the window (e.g. the very
        first anchor of a session) - self-corrects within a few IMU samples during real
        motion, and is exactly the kind of local inaccuracy smooth_session_trajectory fixes
        up after the fact."""
        if timestamp_ns is None:
            return np.zeros(3)
        pts = self.recent_real_positions + [(timestamp_ns, camera_center(pose_cw))]
        pts = [(ts, p) for ts, p in pts if (timestamp_ns - ts) * 1e-9 <= self._velocity_fit_max_age_s]
        if len(pts) < 2:
            return np.zeros(3)
        ts_s = np.array([ts for ts, _ in pts], dtype=np.float64) * 1e-9
        ts_s -= ts_s[-1]  # numerically stable, centered on "now"
        pos = np.array([p for _, p in pts])
        design = np.stack([np.ones_like(ts_s), ts_s], axis=1)
        coeffs, *_ = np.linalg.lstsq(design, pos, rcond=None)
        return coeffs[1]

    def _log_and_advance(
        self, pose_cw: np.ndarray, mode: str, timestamp_ns: int | None,
        velocity: np.ndarray | None = None, bias_vec: np.ndarray | None = None,
        correspondences: list | None = None,
    ) -> None:
        """Common bookkeeping for every localize() return path (track/relocalize/imu_only):
        advances self.last_state (used by the NEXT call's imu_only tier and velocity
        estimate) and appends one entry to self.session_log (consumed once, at the end of
        the session, by smooth_session_trajectory)."""
        if velocity is None:
            velocity = self._estimate_velocity(pose_cw, timestamp_ns)
        if bias_vec is None:
            bias_vec = self.last_state[2] if self.last_state is not None else np.zeros(6)
        self.last_state = (pose_cw, velocity, bias_vec, timestamp_ns)
        if mode != "imu_only":
            self.last_real_state = self.last_state
            self._update_recent_real_positions(timestamp_ns, camera_center(pose_cw))
        self.session_log.append({
            "timestamp_ns": timestamp_ns,
            "mode": mode,
            "pose_cw": pose_cw,
            "velocity": velocity,
            "bias_vec": bias_vec,
            "correspondences": correspondences or [],
        })

    def localize(
        self, rect_img_left: np.ndarray, timestamp_ns: int | None = None,
    ) -> tuple[bool, np.ndarray | None, dict]:
        """`timestamp_ns`: this frame's capture time - required (non-None) for the IMU
        dead-reckoning fallback tier to be available on this call, and for every call's
        session_log entry to be usable by smooth_session_trajectory's IMU chaining. Passing
        None (default) just disables both for that call; track/relocalize still work as
        before."""
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
                self._log_and_advance(
                    pose_cw, "relocalize", timestamp_ns, correspondences=info.get("correspondences"),
                    bias_vec=self._kf_bias(self.ref_kf_id),
                )
                return True, pose_cw, info
            # forced refresh found nothing better - fall through and keep trusting the
            # existing anchor rather than discarding a track state that wasn't actually lost.
            # Reset track_streak regardless of outcome: leaving it >= force_relocalize_every
            # would make force_refresh fire again on every subsequent frame once
            # track-against-ref below re-increments it past the threshold, turning this
            # "periodic cheap check" into an expensive full relocalization every frame.
            self.track_streak = 0

        if self.ref_kf_id is not None:
            result = self._track_against_ref(query_feats, kpts_q)
            if result is not None:
                pose_cw, num_inliers, correspondences = result
                jump_m = None
                if self.max_track_jump_m is not None and self.last_pos is not None:
                    jump_m = float(np.linalg.norm(camera_center(pose_cw) - self.last_pos))
                if jump_m is None or jump_m <= self.max_track_jump_m:
                    self.n_track += 1
                    self.track_streak += 1
                    self.last_pos = camera_center(pose_cw)
                    self._log_and_advance(
                        pose_cw, "track", timestamp_ns, correspondences=correspondences,
                        bias_vec=self._kf_bias(self.ref_kf_id),
                    )
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
            self._log_and_advance(
                pose_cw, "relocalize", timestamp_ns, correspondences=info.get("correspondences"),
                bias_vec=self._kf_bias(self.ref_kf_id),
            )
            return True, pose_cw, info

        # Both track and relocalize failed. Before reporting "lost" outright, try the IMU
        # dead-reckoning fallback - see the class docstring: once a session has any real
        # anchor, it must never report ok=False again. Clear self.ref_kf_id/track_streak
        # here (unlike the max_track_jump_m rejection path above, which leaves them alone):
        # an imu_only frame isn't a visually-verified anchor, so the NEXT call must be
        # forced through a fresh, independent relocalization rather than resuming
        # _track_against_ref against a stale pre-gap reference keyframe - trusting that
        # reference again with zero visual corroboration is exactly the silent-bad-match
        # risk max_track_jump_m/force_relocalize_every exist to guard against elsewhere.
        # (Bug fix: this used to be a comment-only claim with no code behind it - measured
        # on a real MH01 5s-noise-gap test, recovery silently resumed "track" mode against
        # the 5s-stale reference instead of relocalizing, exactly the failure this is meant
        # to prevent.)
        imu_result = self._dead_reckon_imu(timestamp_ns) if timestamp_ns is not None else None
        if imu_result is not None:
            pose_cw, velocity = imu_result
            self.n_imu_only += 1
            self.last_pos = camera_center(pose_cw)
            bias_vec = self.last_state[2] if self.last_state is not None else np.zeros(6)
            self._log_and_advance(pose_cw, "imu_only", timestamp_ns, velocity=velocity, bias_vec=bias_vec)
            self.ref_kf_id = None
            self.track_streak = 0
            return True, pose_cw, {"mode": "imu_only"}

        self.ref_kf_id = None
        self.track_streak = 0
        return False, None, info


def smooth_session_trajectory(
    localizer: "ContinuousLocalizer", k_rect: np.ndarray, pixel_sigma: float = 1.0,
) -> list[np.ndarray]:
    """One-shot, end-of-session correction of `localizer.session_log` (every frame this
    session, whatever mode produced it): a single fixed-map GTSAM factor graph with one
    X(i)/Y(i)/V(i)/B(i) state per logged frame (camera pose / IMU body pose / velocity /
    bias - same X vs. Y split as mapping/local_ba.py, for the same reason: this build's
    GenericProjectionFactorCal3_S2 supports a fixed camera<->body extrinsic implicitly by
    keeping the reprojection factors on X directly, while IMU factors need the body-frame
    convention on a separate symbol), tied together per-frame by a tight
    BetweenFactorPose3 using the known extrinsic. Map points touched by any REAL
    (track/relocalize) frame's correspondences are held FIXED via a tight PriorFactorPoint3
    (mirrors local_ba.py's "fixed_touched" boundary-keyframe pinning pattern, applied here
    to points instead of keyframes) - this is a session-trajectory correction, not a
    mapping pass, so the map itself must never move. CombinedImuFactor chains every
    consecutive pair of logged frames for which real IMU samples exist between their
    timestamps (works across imu_only stretches exactly as well as across two real
    frames - the factor doesn't care which produced the endpoints, only that a genuine IMU
    integral bridges them), which is what pulls a whole imu_only stretch back toward
    consistency with the real, map-anchored frames on both sides of it once this is
    optimized - the live/streaming poses recorded into session_log along the way stay
    approximate; only this corrected list is meant to be trusted afterward.

    Returns one pose_cw per session_log entry, same order/length as the input. Safe to call
    with an all-real (no imu_only frames, even no IMU at all) session too - it degrades to
    a plain fixed-map bundle adjustment over the session's own frames in that case."""
    session_log = localizer.session_log
    n = len(session_log)
    if n == 0:
        return []
    world_map = localizer.world_map
    calib = Cal3_S2(k_rect[0, 0], k_rect[1, 1], 0.0, k_rect[0, 2], k_rect[1, 2])

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    fixed_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))
    point_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1e-6)
    proj_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345), gtsam.noiseModel.Isotropic.Sigma(2, pixel_sigma),
    )
    loose_vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    loose_bias_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 0.1))

    have_imu = localizer.imu_enabled
    extrinsic_pose = matrix_to_gtsam_pose3(localizer.imu_calib.T_cam0_body) if have_imu else None

    pinned_points: set[int] = set()

    def _pin_point(pid: int) -> bool:
        if pid not in world_map.map_points:
            return False
        if pid not in pinned_points:
            pinned_points.add(pid)
            initial.insert(L(pid), Point3(world_map.map_points[pid].position))
            graph.add(PriorFactorPoint3(L(pid), initial.atPoint3(L(pid)), point_prior_noise))
        return True

    n_reproj_factors = 0
    for i, entry in enumerate(session_log):
        initial.insert(X(i), pose_cw_to_gtsam(entry["pose_cw"]))
        for pid, uv in entry["correspondences"]:
            if _pin_point(pid):
                graph.add(GenericProjectionFactorCal3_S2(Point2(uv[0], uv[1]), proj_noise, X(i), L(pid), calib))
                n_reproj_factors += 1
        if have_imu:
            initial.insert(Y(i), pose_cw_to_body_gtsam(entry["pose_cw"], localizer.imu_calib.T_cam0_body))
            initial.insert(V(i), np.asarray(entry["velocity"]))
            initial.insert(B(i), bias_from_vector(entry["bias_vec"]))
            graph.add(BetweenFactorPose3(X(i), Y(i), extrinsic_pose, fixed_prior_noise))

    if not pinned_points:
        # No real correspondences logged at all (e.g. IMU disabled and every frame somehow
        # still returned ok - shouldn't happen by construction, but keep the graph solvable
        # rather than throwing) - anchor the gauge on frame 0 directly instead.
        graph.add(PriorFactorPose3(X(0), initial.atPose3(X(0)), fixed_prior_noise))

    n_imu_factors = 0
    if have_imu:
        for i in range(n - 1):
            ts_a, ts_b = session_log[i]["timestamp_ns"], session_log[i + 1]["timestamp_ns"]
            if ts_a is None or ts_b is None:
                continue
            ts_col = localizer.imu_measurements[:, 0]
            start, end = np.searchsorted(ts_col, ts_a, side="right"), np.searchsorted(ts_col, ts_b, side="right")
            if end <= start + 1:
                continue
            bias_ref = bias_from_vector(session_log[i]["bias_vec"])
            preint = preintegrate(localizer.imu_measurements[start:end], bias_ref, localizer.imu_params)
            graph.add(CombinedImuFactor(Y(i), V(i), Y(i + 1), V(i + 1), B(i), B(i + 1), preint))
            n_imu_factors += 1
        graph.add(PriorFactorVector(V(0), initial.atVector(V(0)), loose_vel_noise))
        graph.add(PriorFactorConstantBias(B(0), initial.atConstantBias(B(0)), loose_bias_noise))

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, gtsam.LevenbergMarquardtParams())
    result = optimizer.optimize()
    return [gtsam_pose_to_cw(result.atPose3(X(i))) for i in range(n)]


def write_session_trajectory(
    session_log: list[dict], corrected_poses: list[np.ndarray], out_path: str,
) -> None:
    """Writes the final, corrected per-frame trajectory to disk - `out_path`'s extension
    picks the format: `.csv` (TUM-style: timestamp_ns, tx, ty, tz, qx, qy, qz, qw, mode) or
    `.npz` (arrays: timestamp_ns, pose_cw [Nx4x4], mode). This is the ONE trajectory a
    pure-localization session is meant to be judged by - the live per-frame poses returned
    from localize() during the run are allowed to be off during imu_only stretches; this
    file is the corrected version, written once at program end."""
    from splg_slam.geometry.pose_utils import camera_center

    timestamps = np.array([e["timestamp_ns"] if e["timestamp_ns"] is not None else -1 for e in session_log])
    modes = np.array([e["mode"] for e in session_log])
    poses = np.stack(corrected_poses, axis=0)

    if out_path.endswith(".npz"):
        np.savez(out_path, timestamp_ns=timestamps, pose_cw=poses, mode=modes)
        return

    def _rot_to_quat_xyzw(rot: np.ndarray) -> tuple[float, float, float, float]:
        # Standard Shepperd's-method rotation-matrix -> quaternion conversion (numerically
        # stable across all rotations, unlike the naive trace-only formula) - kept
        # dependency-free (no gtsam/scipy) since this is just an output-format detail.
        m = rot
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
        return float(qx), float(qy), float(qz), float(qw)

    with open(out_path, "w") as f:
        f.write("# timestamp_ns tx ty tz qx qy qz qw mode\n")
        for ts, pose_cw, mode in zip(timestamps, poses, modes):
            pose_wc = np.linalg.inv(pose_cw)
            t = pose_wc[:3, 3]
            qx, qy, qz, qw = _rot_to_quat_xyzw(pose_wc[:3, :3])
            f.write(f"{int(ts)} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f} {mode}\n")
