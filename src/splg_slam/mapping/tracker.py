import threading

import cv2
import gtsam
import numpy as np
import torch

from splg_slam.data.euroc import ImuCalibration
from splg_slam.features.person_detector import PersonDetector
from splg_slam.features.splg import SPLG
from splg_slam.geometry.mono_init import try_initialize
from splg_slam.geometry.pnp import solve_pnp_ransac
from splg_slam.geometry.pose_utils import camera_center, pose_delta
from splg_slam.geometry.stereo import StereoDepthEstimator, StereoRectifier
from splg_slam.geometry.triangulation import triangulate_points
from splg_slam.localization.retrieval import GlobalDescriptorIndex
from splg_slam.map.frame import Frame
from splg_slam.map.keyframe import KeyFrame
from splg_slam.map.world_map import WorldMap
from splg_slam.mapping.gtsam_utils import gtsam_pose_to_cw, matrix_to_gtsam_pose3, pose_cw_to_body_gtsam
from splg_slam.mapping.imu_init import choose_imu_init_mode
from splg_slam.mapping.imu_preintegration import (
    bias_from_vector,
    gravity_alignment_rotation,
    make_preintegration_params,
    preintegrate,
)
from splg_slam.mapping.local_map import gather_local_map_point_ids, project_points, search_local_map


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
        depth_lookup: dict | None = None, mono_mode: bool = False,
    ):
        self.cfg = cfg
        self.rectifier = rectifier
        # mono_mode: no right image/depth at all - new map points are born by two-view
        # triangulation against the previous reference keyframe (see
        # _triangulate_new_mono_points) instead of single-view depth backproject, and the
        # very first two keyframes need a dedicated essential-matrix bootstrap (see
        # try_initialize / mono_bootstrap_pending below) since there's no map yet to PnP
        # against. depth_est is still constructed (cheap) but never called in mono_mode.
        self.mono_mode = mono_mode
        self.mono_bootstrap_pending = mono_mode
        self.depth_est = StereoDepthEstimator(
            rectifier,
            min_disp=cfg.stereo.min_disp,
            num_disp=cfg.stereo.num_disp,
            block_size=cfg.stereo.block_size,
        )
        # Sensor-provided depth (e.g. a RealSense's onboard depth output, already
        # registered to cam0's raw pixel grid) sidesteps our own stereo-baseline
        # calibration entirely - see load_depth_lookup. None (the common case) keeps the
        # existing computed-SGBM-disparity path unchanged.
        self.depth_lookup = depth_lookup
        self._depth_ts_sorted = np.array(sorted(depth_lookup)) if depth_lookup else None
        # stereo.max_depth_m auto-estimation: an explicit config value always wins (fixed
        # forever, exactly the old behavior). Otherwise, start from a purely geometric
        # ceiling - fx*baseline/MIN_RELIABLE_DISPARITY_PX, MIN_RELIABLE_DISPARITY_PX=3px a
        # universal (not per-dataset) noise-floor constant, same role as e.g. nMinObs
        # elsewhere in this file - and narrow it once real data comes in: this value is NOT
        # purely a sensor-noise floor (confirmed by comparing it against real calibration:
        # EuRoC's fx*baseline is ~48, KITTI's ~386 - an 8x difference - yet their own
        # hand-tuned max_depth_m only differed 2x, 20m vs 40m) - it's closer to "how far
        # does useful scene content actually extend in THIS environment" (EuRoC's indoor
        # machine-hall room vs KITTI's open highway), which calibration geometry alone can't
        # tell you. So: use the geometric ceiling as a generous, always-safe upper bound
        # during a short bootstrap window (depth values beyond it are noise almost
        # regardless of scene), then narrow down to the 90th percentile of what the scene
        # actually shows once enough frames have been seen - adapts to indoor vs outdoor
        # automatically instead of needing a hand-picked meters constant per dataset.
        self._depth_ceiling_m = (
            self.rectifier.fx_rect * self.rectifier.baseline / 3.0
            if not mono_mode and hasattr(self.rectifier, "baseline") else None
        )
        explicit_max_depth_m = getattr(cfg.stereo, "max_depth_m", None) if hasattr(cfg, "stereo") else None
        self._max_depth_m_fixed = explicit_max_depth_m is not None
        # Mono has no stereo geometric ceiling (_depth_ceiling_m is None) and no metric
        # scale of its own - the map's whole scale is whatever mono_init_assumed_depth_m
        # bootstrapped it to. A flat stereo-tuned fallback (e.g. 20m) would be arbitrarily
        # right or wrong depending on that assumption, so scale the fallback with it
        # instead - 5x keeps today's default (assumed depth 4.0m) at the same 20m this
        # used to be hardcoded to, for identical out-of-the-box behavior.
        _mono_fallback_max_depth_m = 5.0 * getattr(cfg.tracking, "mono_init_assumed_depth_m", 4.0)
        self._effective_max_depth_m = (
            explicit_max_depth_m if self._max_depth_m_fixed
            else (
                self._depth_ceiling_m if self._depth_ceiling_m is not None
                else (_mono_fallback_max_depth_m if mono_mode else 20.0)
            )
        )
        # tracking.keyframe_translation_fallback_m auto-estimation: the baseline needed for
        # reliable two-view triangulation parallax (baseline ~= depth * tan(min_parallax_angle)),
        # scaled by stereo.max_depth_m (self._effective_max_depth_m - itself already scene-
        # adaptive, see _update_depth_estimates) rather than a separate scene-median statistic.
        #
        # TRIED and REVERTED: using the pure GEOMETRIC ceiling (_depth_ceiling_m, a fixed
        # calibration-only quantity, never adapting to the actual scene) instead of any scene
        # statistic at all - motivated by a shallow-scene realsense recording (~4m median
        # depth) computing a much tighter fallback (0.14m) than EuRoC's own hand-validated
        # 0.3m purely because its rooms are physically smaller, not because it needs denser
        # keyframes. Measured directly on a real, already-marginal (frequent tracking loss)
        # realsense recording: widening the fallback this way (0.11m -> 0.50m, via the
        # geometric ceiling ignoring the scene entirely) more than HALVED keyframe density but
        # made tracking measurably LESS robust through the exact same rough stretch - lost
        # frames 76->133, Atlas re-inits 2->6 (four extra fragmentations in one stretch that
        # previously survived as continuous tracking), and one segment that previously found a
        # genuine visual merge instead needed an unverified IMU-bridge fallback stitch. Sparser
        # reference keyframes mean more appearance/viewpoint drift accumulates before the next
        # one is inserted, which hurts exactly the marginal, already-prone-to-loss stretches
        # the density is most needed for. Reverted to a scene-adaptive basis (max_depth_m) -
        # still wider than the old raw-scene-median basis (max_depth_m's own 90th-pct x1.2 is
        # larger than a straight median) so it still meaningfully loosens density in shallow
        # scenes, but doesn't strand a rough stretch just because the room happens to be small.
        #
        # keyframe_fallback_parallax_deg is its OWN constant, deliberately NOT the same as
        # mono_init_min_parallax_deg (tried reusing that first - see git history - it's
        # tuned for a different purpose, mono two-view bootstrap triangulation quality, not
        # this one, tracking-density robustness, and using its 1.5deg value overshot real
        # KITTI density badly: 4242 keyframes inserted vs the ~2600 its own explicitly-
        # validated 1.0m fallback needs, expensive enough to blow out disk space saving the
        # resulting map). Back-solving the angle implied by each dataset's own independently
        # hand-validated fallback value against its SCENE median depth (the old basis): EuRoC
        # (0.3m at ~9.6m scene depth) implies ~1.79deg, KITTI (1.0m at ~17.3m depth) implies
        # ~3.31deg - these don't fully agree either (plausibly because KITTI's near-pure-
        # forward highway motion accumulates less effective parallax per meter of baseline
        # than EuRoC's more varied 6DOF flight, a second-order factor this depth-only model
        # doesn't capture), but 2.0deg reproduces EuRoC's own value almost exactly (0.34m vs
        # 0.3m) while meaningfully loosening KITTI's density relative to the old 1.5deg guess
        # (0.60m vs 0.45m) - the safer direction to err on, since UNDER-dense has a bounded
        # downside (this codebase's existing translation/rotation fallback and content-driven
        # signals still catch genuinely necessary insertions) while OVER-dense has an
        # UNbounded one (BA/memory/disk cost scales with keyframe count, and this repo has
        # already hit a real disk-full failure from an over-dense run once). max_depth_m's
        # own 90th-pct-based basis is somewhat larger than a straight scene median, so 2.0deg
        # against it lands a bit looser than the numbers quoted above - not re-validated
        # against EuRoC/KITTI's own hand-tuned values as part of switching the basis.
        explicit_kf_fallback_m = getattr(cfg.tracking, "keyframe_translation_fallback_m", None)
        self._kf_translation_fallback_fixed = explicit_kf_fallback_m is not None
        self._effective_keyframe_translation_fallback_m = (
            explicit_kf_fallback_m if self._kf_translation_fallback_fixed else 0.0
        )

        # ROLLING window, not a one-time bootstrap-then-freeze (that was this mechanism's
        # original design - reverted after a real failure: two realsense bags whose opening
        # ~60 frames happened to see an atypical close-up view (median depth 4.2m and 1.1m
        # respectively, vs a more representative few-meters-to-open-room mix later in the
        # same recording) got BOTH max_depth_m and keyframe_translation_fallback_m locked
        # onto values derived from that one unrepresentative snapshot for the entire rest of
        # the run - the exact same "early data isn't the whole story" failure mode
        # max_pose_jump_m's own bootstrap-then-freeze design hit on KITTI, before being
        # redesigned as a continuously-recomputed rolling window there too. Recomputed every
        # depth_recompute_every_frames (default 30, not every frame - the window can hold
        # tens of thousands of raw depth samples, too much to re-sort/percentile that often)
        # from the most recent depth_window_frames (default 60) frames' own valid depths.
        self._depth_window_frames: list[np.ndarray] = []
        self._depth_window_max_frames = getattr(cfg.stereo, "depth_window_frames", 60) if hasattr(cfg, "stereo") else 60
        self._depth_recompute_every = getattr(cfg.stereo, "depth_recompute_every_frames", 30) if hasattr(cfg, "stereo") else 30
        self._depth_frames_since_recompute = 0
        self._depth_estimates_active = not (self._max_depth_m_fixed and self._kf_translation_fallback_fixed)
        self._depth_last_printed: tuple[float, float] | None = None

        # tracking.max_pose_jump_m auto-estimation, take 3 - SPEED-based, not distance-
        # based. Two earlier attempts (both DISTANCE thresholds - see git history/prior
        # comment here) failed for the same root reason: "distance since ref_keyframe" for
        # a legitimately dropped/blurred-frame gap is large mainly because MORE TIME
        # elapsed, not because the camera moved implausibly fast - measured directly on the
        # two real KITTI00 failures a distance-based threshold rejected: "v=10.05 m/s over
        # 3.11s gap" and "v=8.49 m/s over 2.49s gap" - both a perfectly ordinary highway
        # speed, just sustained over a longer-than-usual gap (30-40 raw meters), which any
        # FIXED distance threshold tuned for "normal-length gap at normal speed" will always
        # misread as implausible. Dividing by the actual elapsed wall-clock time (available
        # from timestamp_ns, which a raw meters check was throwing away) removes that
        # confound entirely: implied SPEED barely varies whether a gap is one dropped frame
        # or ten, so a robust statistic of RECENT implied speed is a stable, physically
        # meaningful quantity in a way recent raw distance never was. Rejects only when the
        # CURRENT frame's implied speed itself is implausible relative to recent normal
        # speed, regardless of how much distance/time that spans.
        self._effective_max_speed_mps: float = 1.0e6  # permissive until enough samples
        self._speed_window: list[float] = []
        self._speed_last_printed: float | None = None
        explicit_pose_jump_m = getattr(cfg.tracking, "max_pose_jump_m", None)
        self._pose_jump_m_fixed = explicit_pose_jump_m is not None
        self._effective_max_pose_jump_m = explicit_pose_jump_m  # only meaningful when fixed
        # max_pose_jump_deg is NOT bootstrapped: every dataset this session has ever needed
        # (EuRoC/KITTI/realsense, walking/hover/car speeds alike) converged on the exact same
        # 90.0 value without any tuning - it's already a de-facto universal constant (a
        # genuinely implausible per-frame ROTATION looks the same regardless of how fast the
        # camera is translating), so it gets a plain code default instead of a bootstrap
        # mechanism that would just rediscover the same number with extra moving parts.
        self.max_pose_jump_deg = getattr(cfg.tracking, "max_pose_jump_deg", 90.0)
        self.splg = SPLG(max_keypoints=cfg.features.max_keypoints)
        self.global_extractor = global_extractor
        # DynamicVINS-style robustness: discard SuperPoint keypoints landing on a
        # detected person before they ever reach PnP/triangulation/matching (see
        # _mask_dynamic_keypoints) - off by default, existing configs unaffected.
        dyn_cfg = getattr(cfg, "dynamic_objects", None)
        self.person_detector = (
            PersonDetector(score_threshold=dyn_cfg.score_threshold)
            if dyn_cfg and dyn_cfg.enabled else None
        )
        self.world_map = WorldMap()

        self.ref_keyframe: KeyFrame | None = None
        self.ref_feats: dict | None = None
        self.last_frame: Frame | None = None
        # Last 1-2 successfully-tracked poses (keyframe or not), for the motion-model
        # tracking fallback's constant-velocity prediction (see _predict_pose). Reset to a
        # single entry after a relocalization jump (not real continuous motion), left
        # untouched while mono's bootstrap is still waiting for parallax.
        self._pose_history: list[np.ndarray] = []
        # Atlas-style multi-segment mapping (mono and stereo): consecutive fully-lost
        # frames (both tracking tiers AND relocalization failed). Once this exceeds
        # tracking.reinit_after_lost_frames, a NEW map segment is started from the
        # current frame (see _start_new_map_segment) instead of waiting - possibly forever -
        # for the camera to wander back into already-mapped territory.
        self._consecutive_lost = 0
        self._bootstrap_wait_frames = 0
        self.n_map_segments = 1  # total distinct segments ever created (for reporting only)
        # Which segment new keyframes get tagged with (see KeyFrame.segment_id) - distinct
        # from n_map_segments because a successful cross-segment map merge (build_map.py)
        # reassigns this back to the surviving segment's id, so keyframes created *after*
        # the merge join that segment directly instead of endlessly re-spawning "segment N"
        # one keyframe at a time until each one individually gets merged too.
        self._current_segment_id = 1
        # Per-frame camera speed (map units/frame) captured from the dead-reckoning chain at
        # the moment a segment was re-seeded, used to give the new segment's two-view
        # bootstrap a baseline in the *existing* map's units - see _imu_bootstrap_scale.
        self._segment_seed_speed: float | None = None
        self._next_frame_id = 0
        self.n_keyframes_inserted = 0
        # Which _need_new_keyframe condition fired for each inserted keyframe (stereo:
        # "ref_ratio", "close_points", "ref_ratio+close_points"; mono: "mono_geometric") -
        # diagnostic only, for understanding what's actually driving insertion on a given
        # dataset (see build_map.py's "Done." summary line).
        self.kf_insert_reason_counts: dict[str, int] = {}
        self.n_relocalizations = 0
        self.track_stats: list[dict] = []

        self.reloc_index = GlobalDescriptorIndex() if global_extractor is not None else None
        self.reloc_feats_cache: dict[int, dict] = {}

        self.imu_enabled = bool(
            getattr(cfg, "imu", None) and cfg.imu.enabled and imu_measurements is not None and imu_calib is not None
        )
        # tight_fusion=False: use the IMU only to gravity-align the very first keyframe from
        # a static leading window (choose_imu_init_mode/gravity_alignment_rotation below) -
        # no preintegration factors in BA, no dynamic/periodic reinit, no per-keyframe
        # velocity/bias estimation. A lighter-weight alternative to full VIO for a stereo run
        # that just wants a gravity-aligned Z axis (map scale/robustness stay exactly as
        # vision-only stereo's, since nothing IMU-derived ever touches the map afterward).
        # If the leading window isn't actually static, gravity can't be determined this way
        # (that's what tight_fusion's dynamic-init path is for) - the run proceeds with an
        # unaligned (arbitrary Z) orientation instead, same as imu_enabled=False.
        self.imu_tight_fusion = self.imu_enabled and bool(getattr(cfg.imu, "tight_fusion", True))
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
        # Optional pre-calibrated bias (e.g. from a lab/turntable calibration of this
        # specific IMU unit) - a 6-vector [accel_bias(3), gyro_bias(3)], same layout as
        # KeyFrame.imu_bias. Used as: (1) subtracted from the raw accelerometer average
        # before static-window gravity alignment (that method has no other way to tell
        # true gravity direction apart from sensor bias - see gravity_alignment_rotation),
        # and (2) the starting guess / Tikhonov-prior center for the dynamic solve
        # (_solve_and_realign) instead of an assumed-zero bias, for every keyframe that
        # doesn't yet have a better (previously-estimated) value of its own. None (the
        # default) keeps the original zero-based behavior. Bias INSTABILITY still means a
        # calibrated value isn't permanently valid across power cycles - see this
        # session's IMU-init discussion - so this is a better starting prior, not a
        # replacement for the online solve.
        accel_bias_prior = getattr(cfg.imu, "accel_bias_prior", None) if self.imu_enabled else None
        gyro_bias_prior = getattr(cfg.imu, "gyro_bias_prior", None) if self.imu_enabled else None
        self._imu_bias_prior = (
            np.concatenate([
                np.zeros(3) if accel_bias_prior is None else np.asarray(accel_bias_prior, dtype=np.float64),
                np.zeros(3) if gyro_bias_prior is None else np.asarray(gyro_bias_prior, dtype=np.float64),
            ])
            if self.imu_tight_fusion else None
        )

    def _pose_jump_threshold_m(self, timestamp_ns: int) -> float:
        """Current max_pose_jump_m to check `translation` (relative to ref_keyframe)
        against - a fixed config value if given, else effective_max_speed_mps (see
        _update_speed_threshold) times the actual elapsed time since ref_keyframe, floored
        at pose_jump_min_floor_m so a very small dt doesn't produce a degenerately tight
        threshold before the speed estimate itself has stabilized."""
        if self._pose_jump_m_fixed:
            return self._effective_max_pose_jump_m
        dt_s = max((timestamp_ns - self.ref_keyframe.timestamp_ns) / 1.0e9, 1.0e-3)
        floor_m = getattr(self.cfg.tracking, "pose_jump_min_floor_m", 0.5)
        return max(self._effective_max_speed_mps * dt_s, floor_m)

    def _update_depth_estimates(self, depths: np.ndarray) -> None:
        """Rolling-window re-estimation of stereo.max_depth_m and tracking.keyframe_
        translation_fallback_m (whichever isn't explicitly fixed) from this frame's own
        valid depths - see __init__'s comment for why this recomputes continuously from a
        recent window instead of a one-time early-sequence snapshot. Bounded to the
        geometric noise-floor ceiling before being added to the window, so a stray bad
        disparity spike can't blow out the percentile the same way it would blow out a mean.
        Recomputes only every depth_recompute_every_frames frames (not every frame) since
        the window can hold tens of thousands of raw depth samples - too much to re-sort/
        percentile that often for negligible extra freshness.

        keyframe_translation_fallback_m is derived from new_max_depth (not a separate scene-
        median statistic) - see __init__'s comment on why a scene-adaptive basis, not the
        pure geometric ceiling, was kept."""
        valid = depths[~np.isnan(depths)]
        if self._depth_ceiling_m is not None:
            valid = valid[valid <= self._depth_ceiling_m]
        if valid.size > 0:
            self._depth_window_frames.append(valid)
            if len(self._depth_window_frames) > self._depth_window_max_frames:
                self._depth_window_frames.pop(0)
        self._depth_frames_since_recompute += 1
        if self._depth_frames_since_recompute < self._depth_recompute_every or not self._depth_window_frames:
            return
        self._depth_frames_since_recompute = 0
        all_samples = np.concatenate(self._depth_window_frames)
        new_max_depth, new_fallback = self._effective_max_depth_m, self._effective_keyframe_translation_fallback_m
        if not self._max_depth_m_fixed:
            scene_p90 = float(np.percentile(all_samples, 90))
            new_max_depth = min(self._depth_ceiling_m, scene_p90 * 1.2)
        if not self._kf_translation_fallback_fixed:
            fallback_parallax_deg = getattr(self.cfg.tracking, "keyframe_fallback_parallax_deg", 2.0)
            new_fallback = new_max_depth * np.tan(np.radians(fallback_parallax_deg))
        # Print only the first estimate and any later >=20% swing in either value - a real
        # scene-depth regime change (e.g. exiting a narrow corridor into an open room) worth
        # seeing in the log, without spamming every recompute.
        prev = self._depth_last_printed
        swung = prev is None or abs(new_max_depth - prev[0]) > 0.2 * max(prev[0], 1e-6) or abs(new_fallback - prev[1]) > 0.2 * max(prev[1], 1e-6)
        if swung:
            if not self._max_depth_m_fixed:
                print(
                    f"  stereo.max_depth_m auto-estimated: {new_max_depth:.1f}m (scene's own "
                    f"recent 90th-pct depth x1.2, geometric ceiling {self._depth_ceiling_m:.1f}m, "
                    f"over last {len(self._depth_window_frames)} frames) - override with "
                    f"stereo.max_depth_m if unreliable"
                )
            if not self._kf_translation_fallback_fixed:
                print(
                    f"  tracking.keyframe_translation_fallback_m auto-estimated: {new_fallback:.2f}m "
                    f"(stereo.max_depth_m x tan(keyframe_fallback_parallax_deg), over "
                    f"last {len(self._depth_window_frames)} frames) - override with "
                    f"tracking.keyframe_translation_fallback_m if unreliable"
                )
            self._depth_last_printed = (new_max_depth, new_fallback)
        self._effective_max_depth_m, self._effective_keyframe_translation_fallback_m = new_max_depth, new_fallback

    def _update_speed_threshold(self, implied_speed_mps: float) -> None:
        """Called on every accepted (non-rejected) tracked frame with translation/dt since
        ref_keyframe - see __init__'s comment on why speed, not raw distance. Rolling window
        (speed_window_frames, default 150) robust statistic: median + 10x a MAD-based sigma
        estimate, floored at speed_min_floor_mps. 10x and the 1.4826 MAD-to-sigma factor are
        universal statistics constants, not per-dataset tuning. The floor guards against a
        degenerate near-zero threshold during a near-stationary stretch (e.g. parked/
        hovering) - genuinely implausible-speed detection should never get THIS tight
        regardless of how quiet a recent stretch was. Sampling every frame (not just at
        keyframe insertion) is safe here, unlike the earlier raw-distance version: implied
        speed doesn't saw-tooth with insertion timing the way accumulated distance does."""
        window_size = getattr(self.cfg.tracking, "speed_window_frames", 150)
        self._speed_window.append(implied_speed_mps)
        if len(self._speed_window) > window_size:
            self._speed_window.pop(0)
        min_samples = getattr(self.cfg.tracking, "speed_min_samples", 20)
        if len(self._speed_window) < min_samples:
            return
        speeds = np.array(self._speed_window)
        median_v = float(np.median(speeds))
        robust_sigma_v = 1.4826 * float(np.median(np.abs(speeds - median_v)))
        floor_mps = getattr(self.cfg.tracking, "speed_min_floor_mps", 1.0)
        new_threshold = max(median_v + 10.0 * robust_sigma_v, floor_mps)
        # Print only the first stabilization and any later >=20% swing - a real speed-
        # regime change (e.g. KITTI's residential-street start vs its later highway
        # stretches) worth seeing in the log, without spamming every single update.
        if self._speed_last_printed is None or abs(new_threshold - self._speed_last_printed) > 0.2 * self._speed_last_printed:
            print(
                f"  tracking.max_pose_jump_m (speed-based) auto-estimated: {new_threshold:.2f}m/s "
                f"(median recent speed {median_v:.3f}m/s + 10x robust sigma {robust_sigma_v:.3f}m/s "
                f"over last {len(speeds)} frames) - set tracking.max_pose_jump_m explicitly if unreliable"
            )
            self._speed_last_printed = new_threshold
        self._effective_max_speed_mps = new_threshold

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

    def _record_tracked_pose(self, pose_cw: np.ndarray, reset: bool = False) -> None:
        if reset:
            self._pose_history = [pose_cw]
            return
        self._pose_history.append(pose_cw)
        if len(self._pose_history) > 2:
            self._pose_history.pop(0)

    def _predict_pose(self) -> np.ndarray | None:
        """Constant-velocity extrapolation from the last two successfully-tracked poses -
        used by _try_motion_model_tracking. None until at least two prior poses exist (the
        first couple of frames), in which case that tier is simply skipped."""
        if len(self._pose_history) < 2:
            return None
        pose_prev, pose_last = self._pose_history[-2], self._pose_history[-1]
        c_prev, c_last = camera_center(pose_prev), camera_center(pose_last)
        r_rel = pose_last[:3, :3] @ pose_prev[:3, :3].T
        r_pred = r_rel @ pose_last[:3, :3]
        c_pred = c_last + (c_last - c_prev)
        pose_pred = np.eye(4)
        pose_pred[:3, :3] = r_pred
        pose_pred[:3, 3] = -r_pred @ c_pred
        return pose_pred

    def _depths_from_sensor(self, kpts: np.ndarray, timestamp_ns: int) -> np.ndarray:
        """Nearest-timestamp-match the frame to a sensor depth image (raw uint16 mm, on
        cam0's unrectified pixel grid), warp it through cam0's own rectification map so it
        lines up with kpts (which are in the rectified frame, same as StereoDepthEstimator's
        output), and sample at each keypoint."""
        j = int(np.searchsorted(self._depth_ts_sorted, timestamp_ns))
        j = min(max(j, 0), len(self._depth_ts_sorted) - 1)
        if j > 0 and abs(self._depth_ts_sorted[j - 1] - timestamp_ns) < abs(self._depth_ts_sorted[j] - timestamp_ns):
            j -= 1
        depth_entry = self.depth_lookup[int(self._depth_ts_sorted[j])]
        # rosbag2 mode's load_depth_lookup keeps already-decoded arrays in memory instead
        # of file paths (there's no depth0/ on disk to read back) - euroc's file-based
        # depth_lookup is unaffected, still a {timestamp_ns: Path} dict.
        depth_mm = depth_entry if isinstance(depth_entry, np.ndarray) else cv2.imread(str(depth_entry), cv2.IMREAD_UNCHANGED)
        depth_rect = cv2.remap(depth_mm, self.rectifier.map_l[0], self.rectifier.map_l[1], cv2.INTER_NEAREST)

        h, w = depth_rect.shape
        xi = np.round(kpts[:, 0]).astype(np.int64)
        yi = np.round(kpts[:, 1]).astype(np.int64)
        in_bounds = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        depths = np.full(len(kpts), np.nan, dtype=np.float32)
        d_mm = np.zeros(len(kpts), dtype=np.float32)
        d_mm[in_bounds] = depth_rect[yi[in_bounds], xi[in_bounds]].astype(np.float32)
        valid = in_bounds & (d_mm > 0) & (d_mm < 65535)
        depths[valid] = d_mm[valid] / 1000.0
        return depths

    def process_stereo_pair(self, img_left: np.ndarray, img_right: np.ndarray,
                             timestamp_ns: int, image_path: str | None = None):
        """Returns (frame_or_keyframe, is_keyframe). frame_or_keyframe is None if tracking failed."""
        rect_l, rect_r = self.rectifier.rectify(img_left, img_right)

        if self.depth_lookup is not None:
            feats = self.splg.extract(rect_l)
            feats = self._mask_dynamic_keypoints(feats, rect_l)
            kpts, desc = SPLG.to_frame_arrays(feats)
            depths = self._depths_from_sensor(kpts, timestamp_ns)
        else:
            # SuperPoint extraction (GPU) and SGBM disparity (CPU) only share rect_l/rect_r
            # as input - neither touches world_map or any other shared mutable state - so
            # they're run concurrently on a helper thread instead of back-to-back. Both
            # OpenCV's C++ StereoMatcher.compute and torch's CUDA calls release the GIL
            # while waiting on the device/kernel, so this achieves real wall-clock overlap
            # despite Python's GIL (measured: see local_bundle_adjustment's docstring for
            # the much larger, separate periodic-BA convergence tuning - this SGBM/extract
            # overlap is the smaller, safe-by-construction half of the two "engineering"
            # asks; a full async Tracking/LocalMapping thread split was scoped and rejected -
            # see build_map.py's main() docstring note on why).
            disp_result: dict = {}

            def _compute_disp():
                disp_result["disp"] = self.depth_est.compute_disparity(rect_l, rect_r)

            disp_thread = threading.Thread(target=_compute_disp)
            disp_thread.start()
            feats = self.splg.extract(rect_l)
            feats = self._mask_dynamic_keypoints(feats, rect_l)
            kpts, desc = SPLG.to_frame_arrays(feats)
            disp_thread.join()
            depths = self.depth_est.depths_at_points(disp_result["disp"], kpts)

        if self._depth_estimates_active:
            self._update_depth_estimates(depths)

        depths[depths > self._effective_max_depth_m] = np.nan

        return self._process_frame(rect_l, feats, kpts, desc, depths, timestamp_ns, image_path)

    def process_mono_pair(self, img_left: np.ndarray, timestamp_ns: int, image_path: str | None = None):
        """Same as process_stereo_pair but with no right image/depth at all - new map
        points come from two-view triangulation (see _triangulate_new_mono_points) instead
        of single-view depth backproject. Returns (frame_or_keyframe, is_keyframe)."""
        rect_l, _ = self.rectifier.rectify(img_left, img_left)
        feats = self.splg.extract(rect_l)
        feats = self._mask_dynamic_keypoints(feats, rect_l)
        kpts, desc = SPLG.to_frame_arrays(feats)
        depths = np.full(len(kpts), np.nan, dtype=np.float32)
        return self._process_frame(rect_l, feats, kpts, desc, depths, timestamp_ns, image_path)

    def _mask_dynamic_keypoints(self, feats: dict, rect_l: np.ndarray) -> dict:
        """DynamicVINS-style filter: drops any keypoint landing inside a detected
        person's bounding box (padded by dynamic_objects.bbox_margin_px, guarding
        against detector box imprecision/motion blur at the silhouette edge) before it
        can reach PnP, triangulation, local-map matching, or keyframe storage - a moving
        person violates the static-scene assumption all of those rely on."""
        if self.person_detector is None:
            return feats
        boxes = self.person_detector.detect_person_boxes(rect_l)
        if len(boxes) == 0:
            return feats

        kpts = feats["keypoints"][0].cpu().numpy()
        margin = self.cfg.dynamic_objects.bbox_margin_px
        inside = np.zeros(len(kpts), dtype=bool)
        for x1, y1, x2, y2 in boxes:
            inside |= (
                (kpts[:, 0] >= x1 - margin) & (kpts[:, 0] <= x2 + margin)
                & (kpts[:, 1] >= y1 - margin) & (kpts[:, 1] <= y2 + margin)
            )
        if not inside.any():
            return feats

        keep = torch.from_numpy(~inside).to(feats["keypoints"].device)
        filtered = dict(feats)
        for key in ("keypoints", "keypoint_scores", "descriptors"):
            filtered[key] = feats[key][:, keep]
        return filtered

    def _process_frame(self, rect_l: np.ndarray, feats: dict, kpts: np.ndarray, desc: np.ndarray,
                        depths: np.ndarray, timestamp_ns: int, image_path: str | None):
        frame_id = self._next_frame_id
        self._next_frame_id += 1

        if self.imu_tight_fusion:
            self._pull_imu_samples(timestamp_ns)

        if self.ref_keyframe is None:
            pose_cw0 = np.eye(4)
            if self.imu_enabled:
                # World frame is defined by this bootstrap keyframe, so its orientation
                # fixes gravity's direction in world frame for every later IMU factor (or,
                # tight_fusion=False, simply the map's own Z axis - nothing reads it again
                # afterward). EuRoC sequences are typically handled/moved before takeoff
                # (not static), so check the actual leading IMU data instead of assuming
                # either case.
                self.imu_init_mode, accel_samples = choose_imu_init_mode(
                    self.imu_measurements, self.cfg.imu.init_static_samples,
                    search_samples=10 * self.cfg.imu.init_static_samples,
                    gyro_static_threshold=self.cfg.imu.init_gyro_static_threshold,
                    gravity_norm=self.cfg.imu.gravity_norm,
                )
                if self.imu_init_mode == "static":
                    r_world_body0 = gravity_alignment_rotation(
                        accel_samples,
                        accel_bias=self._imu_bias_prior[:3] if self._imu_bias_prior is not None else None,
                    )
                    r_cam0_body = self.imu_calib.T_cam0_body[:3, :3]
                    pose_cw0[:3, :3] = r_cam0_body @ r_world_body0.T
                elif not self.imu_tight_fusion:
                    print("  warning: tight_fusion=False and the leading IMU window isn't "
                          "static - can't gravity-align without either; proceeding with an "
                          "arbitrary (non-gravity-aligned) orientation, same as imu.enabled=false")
                if self.imu_tight_fusion:
                    # Unify onto one authoritative path regardless of whether the static
                    # check above succeeded: that crude accel-average alignment (if any)
                    # never separates true gravity direction from sensor bias (a single
                    # static window has no rotational diversity to do that - see
                    # gravity_alignment_rotation) and is only ever a provisional seed.
                    # run_dynamic_imu_init (called from build_map.py once enough keyframes/
                    # IMU data accumulate) jointly solves gyro bias + accel bias + gravity +
                    # velocities from real, vision-cross-validated motion and retroactively
                    # re-aligns everything - schedule it unconditionally so it's always the
                    # one that actually settles this segment's orientation, rather than only
                    # running (and this segment being stuck on the cruder guess) when the
                    # leading window happened to not be static.
                    self.imu_init_pending = True
            kf = KeyFrame(
                frame_id=frame_id, timestamp_ns=timestamp_ns,
                keypoints=kpts, descriptors=desc, depths=depths,
                pose_cw=pose_cw0, image_path=str(image_path) if image_path else None,
                velocity=np.zeros(3) if self.imu_tight_fusion else None,
                imu_bias=self._imu_bias_prior,
                segment_id=self._current_segment_id,
            )
            self._insert_keyframe(kf, feats, rect_l)
            self.last_frame = kf
            self._record_tracked_pose(kf.pose_cw)
            return kf, True

        match = self.splg.match(self.ref_feats, feats)
        matches = match["matches"]

        if self.mono_mode and self.mono_bootstrap_pending:
            result = try_initialize(
                self.ref_keyframe.keypoints, kpts, matches, self.rectifier.K_rect,
                min_matches=getattr(self.cfg.tracking, "mono_init_min_matches", 100),
                min_parallax_deg=getattr(self.cfg.tracking, "mono_init_min_parallax_deg", 1.5),
                min_triangulated=getattr(self.cfg.tracking, "mono_init_min_triangulated", 50),
                assumed_scene_depth_m=getattr(self.cfg.tracking, "mono_init_assumed_depth_m", 4.0),
            )
            if result is None:
                # Not enough matches/parallax yet against the first keyframe - keep waiting
                # for more camera motion (mirrors ORB-SLAM's monocular bootstrap loop).
                self._bootstrap_wait_frames += 1
                max_wait = getattr(self.cfg.tracking, "mono_bootstrap_max_wait_frames", 60)
                if self.n_map_segments > 1 and self._bootstrap_wait_frames >= max_wait:
                    # A re-init seed keyframe that never manages to bootstrap (e.g. it was
                    # itself a blurred frame with poor features) would otherwise stall the
                    # whole run - re-seed the new segment from the current frame instead.
                    return self._start_new_map_segment(kpts, desc, depths, rect_l, feats, frame_id, timestamp_ns, image_path)
                frame = Frame(
                    frame_id=frame_id, timestamp_ns=timestamp_ns,
                    keypoints=kpts, descriptors=desc, depths=depths, pose_cw=self.ref_keyframe.pose_cw.copy(),
                )
                self.last_frame = frame
                return frame, False

            # try_initialize works in the reference keyframe's own frame (ref at identity):
            # compose its relative pose onto the ref's absolute pose so the new keyframe lands
            # in the shared world frame - required both for a re-initialized map segment (ref
            # seeded mid-flight, far from the origin) and for an IMU-gravity-aligned first
            # keyframe (ref rotation != identity).
            pose_cw_b = result.pose_cw_b.copy()
            imu_scale = self._imu_bootstrap_scale(pose_cw_b)
            if imu_scale is not None:
                # Re-initialized segment: rescale so its baseline matches the IMU-measured
                # (or dead-reckoned) motion across the gap, keeping it in the same units as
                # the rest of the map. The map points triangulated from this pair are scaled
                # to match inside _triangulate_new_mono_points, which uses these poses.
                pose_cw_b[:3, 3] *= imu_scale
            kf = KeyFrame(
                frame_id=frame_id, timestamp_ns=timestamp_ns,
                keypoints=kpts, descriptors=desc, depths=depths, pose_cw=pose_cw_b @ self.ref_keyframe.pose_cw,
                image_path=str(image_path) if image_path else None,
                velocity=np.zeros(3) if self.imu_tight_fusion else None,
                # This mono segment's first keyframe (self.ref_keyframe) already carries
                # whatever bias applies (a prior or a previously-estimated value - see
                # __init__'s _imu_bias_prior) - inherit it rather than restarting from zero.
                imu_bias=(
                    self.ref_keyframe.imu_bias if self.ref_keyframe.imu_bias is not None else self._imu_bias_prior
                ) if self.imu_tight_fusion else None,
                segment_id=self._current_segment_id,
            )
            self._insert_keyframe(kf, feats, rect_l, ref_matches=matches)
            self.mono_bootstrap_pending = False
            self._bootstrap_wait_frames = 0
            self.last_frame = kf
            self._record_tracked_pose(kf.pose_cw)
            if self.n_map_segments > 1:
                print(f"  [atlas] map segment {self.n_map_segments} bootstrapped at kf{kf.frame_id} "
                      f"({result.num_points} init points, parallax {result.median_parallax_deg:.1f}deg"
                      f"{', IMU-scaled' if imu_scale is not None else ''})")
            return kf, True

        tracked = self._try_track_against_ref(matches, kpts, desc, rect_l)
        used_motion_model = False
        used_imu_bridge_prediction = False
        predicted = None
        if tracked is None:
            # Tier 2: LightGlue matching against the single reference keyframe failed
            # (typical cause: motion blur / large appearance change during a fast
            # segment) - fall back to projecting the existing local map using a
            # constant-velocity-predicted pose instead, before giving up to full
            # relocalization. Tolerates appearance change much better since it only
            # needs "a similar descriptor near where geometry predicts", not a LightGlue
            # graph match between two full images.
            predicted = self._predict_pose()
            tracked = self._try_motion_model_tracking(kpts, desc, rect_l, predicted, depths)
            used_motion_model = tracked is not None
            if tracked is None and self.imu_tight_fusion:
                # Tier 2b (tight_fusion only): constant-velocity assumes the camera keeps
                # doing whatever it did over the last tracked frame interval, which is
                # exactly wrong during the sustained fast/rotating motion this tier exists
                # for - once _pose_history starts getting fed predictions instead of real
                # observations (see the dead-reckoning update below), that stale guess
                # compounds frame over frame. _imu_bridge_pose_cw() integrates the actual
                # measured rotation/acceleration since ref_keyframe instead of assuming
                # constant motion, so retry the same matching tier with that prediction
                # before giving up to full relocalization.
                #
                # Deliberately tried SECOND, not first: its integration window spans back
                # to ref_keyframe (_pending_imu_samples only resets on keyframe insertion,
                # so this can be many frames old), while _predict_pose's baseline is just
                # the single freshest tracked frame - for an isolated one-frame miss amid
                # otherwise-good tracking, that longer IMU window is more exposed to bias
                # error and would likely be worse, so it only comes in once the fresher,
                # cheaper constant-velocity guess has already failed.
                imu_predicted = self._imu_bridge_pose_cw()
                if imu_predicted is not None:
                    tracked = self._try_motion_model_tracking(kpts, desc, rect_l, imu_predicted, depths)
                    if tracked is not None:
                        used_motion_model = True
                        used_imu_bridge_prediction = True
                        predicted = imu_predicted
        if tracked is None:
            result = self._on_tracking_failure(rect_l, feats, kpts, desc, depths, frame_id, timestamp_ns)
            if result[0] is not None:
                self._consecutive_lost = 0
                return result
            self._consecutive_lost += 1
            if predicted is not None:
                # Total loss this frame (relocalization also failed): still advance the
                # dead-reckoning chain using this frame's own prediction, so the NEXT
                # frame's constant-velocity guess keeps pace with elapsed time instead of
                # staying frozen at the last *validated* pose.
                self._record_tracked_pose(predicted)
            reinit_after = getattr(self.cfg.tracking, "reinit_after_lost_frames", 15)
            if self._consecutive_lost >= reinit_after:
                if getattr(self.cfg.tracking, "continuous_imu_tracking", False):
                    # EXPERIMENTAL, opt-in, off by default: rather than cutting into a new
                    # Atlas segment, keep tracking going via pure IMU prediction (see
                    # _insert_imu_only_keyframe) - deliberately does NOT reset
                    # _consecutive_lost, so if vision is still lost next frame too, this
                    # same branch fires again immediately (one IMU-only keyframe per lost
                    # frame) instead of needing another full reinit_after-frame buildup each
                    # time.
                    imu_only_result = self._insert_imu_only_keyframe(
                        kpts, desc, depths, rect_l, feats, frame_id, timestamp_ns, image_path
                    )
                    if imu_only_result is not None:
                        return imu_only_result
                    # IMU state isn't usable (or the predicted pose failed the sanity
                    # check) - fall through to the Atlas safety net below, same as when
                    # continuous_imu_tracking is off.
                # ORB-SLAM3 Atlas-style recovery: a sustained loss almost always means the
                # camera moved into territory the map never covered (that's *why* neither
                # tracking tier nor relocalization can find anything) - waiting for it to
                # wander back is hopeless, so start a fresh map segment right here. Stereo
                # segments need no bootstrap wait (see _start_new_map_segment) since a
                # single stereo frame already has metric depth.
                return self._start_new_map_segment(kpts, desc, depths, rect_l, feats, frame_id, timestamp_ns, image_path)
            return result
        self._consecutive_lost = 0
        pose_cw, point_ids, kp_idx_list, inlier_mask, num_local_map_matches = tracked

        translation, rotation_deg = pose_delta(self.ref_keyframe.pose_cw, pose_cw)
        if (
            translation > self._pose_jump_threshold_m(timestamp_ns)
            or rotation_deg > self.max_pose_jump_deg
        ):
            # A "valid" (enough inliers, low reprojection error) but physically
            # implausible PnP solution - e.g. a near-degenerate point configuration in a
            # repetitive scene. Silently accepting this creates a keyframe at a nonsense
            # position that every later frame then tracks forward from. Treat it as a
            # tracking failure instead of a keyframe.
            return self._on_tracking_failure(rect_l, feats, kpts, desc, depths, frame_id, timestamp_ns)

        if not self._pose_jump_m_fixed:
            dt_s = max((timestamp_ns - self.ref_keyframe.timestamp_ns) / 1.0e9, 1.0e-3)
            self._update_speed_threshold(translation / dt_s)

        frame = Frame(
            frame_id=frame_id, timestamp_ns=timestamp_ns,
            keypoints=kpts, descriptors=desc, depths=depths, pose_cw=pose_cw,
        )
        self.last_frame = frame
        self._record_tracked_pose(pose_cw)

        tracked_ratio = int(inlier_mask.sum()) / max(len(kpts), 1)

        need_new_kf, kf_reason = self._need_new_keyframe(
            translation, rotation_deg, inlier_mask, kp_idx_list, tracked_ratio, depths,
        )
        self.track_stats.append({
            "frame_id": frame_id, "translation": translation, "rotation_deg": rotation_deg,
            "tracked_ratio": tracked_ratio, "num_matched_with_mp": len(point_ids) - num_local_map_matches,
            "num_local_map_matches": num_local_map_matches, "used_motion_model": used_motion_model,
            "used_imu_bridge_prediction": used_imu_bridge_prediction,
            "num_inliers": int(inlier_mask.sum()), "total_kpts": len(kpts),
        })
        if not need_new_kf:
            return frame, False
        self.kf_insert_reason_counts[kf_reason] = self.kf_insert_reason_counts.get(kf_reason, 0) + 1

        kf = KeyFrame(
            frame_id=frame_id, timestamp_ns=timestamp_ns,
            keypoints=kpts, descriptors=desc, depths=depths, pose_cw=pose_cw,
            image_path=str(image_path) if image_path else None,
            segment_id=self._current_segment_id,
        )
        for pid, kp_i, is_inlier in zip(point_ids, kp_idx_list, inlier_mask):
            if is_inlier:
                kf.map_point_ids[kp_i] = pid
        self._insert_keyframe(kf, feats, rect_l, ref_matches=matches)
        return kf, True

    def _imu_bridge_pose_cw(self) -> np.ndarray | None:
        """tight_fusion, after IMU init: propagates the last keyframe's state through every
        IMU sample accumulated since it (self._pending_imu_samples - i.e. across a tracking
        gap) and returns the predicted current camera pose_cw. Unlike constant-velocity
        extrapolation this captures a sudden rotation (the gyro measures it directly) and is
        metric. Returns None when the IMU state isn't usable yet. Read-only: leaves
        _pending_imu_samples for _insert_keyframe to turn into the bridging IMU factor.

        Two very different uses, both fine: (1) _insert_imu_only_keyframe
        (tracking.continuous_imu_tracking) calls this every frame while vision is lost,
        continuing an already-tracking NavState one frame at a time - each gap is a single
        frame interval, so double-integration error stays small per step, same as any
        tightly-coupled VIO's normal per-frame IMU propagation. (2) _imu_bootstrap_scale
        uses it only as a magnitude (how far did we move), not a pose, which is far more
        forgiving of integration error than using it as an actual seed pose.
        Deliberately NOT used as _start_new_map_segment's seed pose (removed this session) -
        that use bridges an *unknown-length, possibly-long* gap into a *new, otherwise
        unrelated* coordinate frame, where double-integration error compounds over
        however long tracking had already been lost before giving up - measured to make
        Atlas-reinit recovery worse, not better."""
        if not (self.imu_tight_fusion and not self.imu_init_pending and self.ref_keyframe is not None):
            return None
        if self.ref_keyframe.velocity is None or not self._pending_imu_samples:
            return None
        samples = np.concatenate(self._pending_imu_samples, axis=0)
        if len(samples) < 2:
            return None
        bias = bias_from_vector(self.ref_keyframe.imu_bias)
        preint = preintegrate(samples, bias, self._imu_params)
        body_pose = pose_cw_to_body_gtsam(self.ref_keyframe.pose_cw, self.imu_calib.T_cam0_body)
        predicted = preint.predict(gtsam.NavState(body_pose, self.ref_keyframe.velocity), bias)
        t_cam_body = matrix_to_gtsam_pose3(self.imu_calib.T_cam0_body)
        cam_wc = predicted.pose().compose(t_cam_body.inverse())
        return gtsam_pose_to_cw(cam_wc)

    def _insert_imu_only_keyframe(
        self, kpts: np.ndarray, desc: np.ndarray, depths: np.ndarray, rect_l: np.ndarray,
        feats: dict, frame_id: int, timestamp_ns: int, image_path: str | None,
    ) -> tuple[KeyFrame, bool] | None:
        """tracking.continuous_imu_tracking (opt-in, default off): instead of cutting into a
        new independent-coordinate-frame Atlas segment on sustained vision loss, keep the
        SAME segment going with a pure-IMU-predicted keyframe - tinynav-style "never truly
        lose tracking", but only ever a single-frame IMU step at a time (see
        _imu_bridge_pose_cw's docstring on why that's safe here but wasn't as a segment seed).
        _insert_keyframe's existing machinery handles everything else unchanged: the
        CombinedImuFactor to the previous keyframe (with its own correctly-growing
        preintegration covariance - no separate bookkeeping needed), and stereo depth
        backprojected into fresh map points exactly as for any normal keyframe, so a later
        loop closure still has something to visually match against once the trajectory
        revisits familiar territory. Returns None (caller falls back to
        _start_new_map_segment as the safety net) if IMU state isn't usable yet, or if the
        predicted pose fails the same max_pose_jump_m/deg sanity check normal tracking
        already uses - an IMU-only chain should still have a backstop rather than run away
        indefinitely on a bad prediction."""
        predicted_pose = self._imu_bridge_pose_cw()
        if predicted_pose is None:
            return None
        translation, rotation_deg = pose_delta(self.ref_keyframe.pose_cw, predicted_pose)
        if translation > self._pose_jump_threshold_m(timestamp_ns) or rotation_deg > self.max_pose_jump_deg:
            return None
        kf = KeyFrame(
            frame_id=frame_id, timestamp_ns=timestamp_ns,
            keypoints=kpts, descriptors=desc, depths=depths, pose_cw=predicted_pose,
            image_path=str(image_path) if image_path else None,
            segment_id=self._current_segment_id,
            imu_only=True,
        )
        self._insert_keyframe(kf, feats, rect_l)
        self.last_frame = kf
        self._record_tracked_pose(predicted_pose)
        return kf, True

    def _gravity_aligned_origin_pose(self, timestamp_ns: int) -> np.ndarray:
        """A fresh, independent world origin: identity translation, gravity-aligned Z axis
        (horizontal heading arbitrary) - the same thing the very first keyframe of the
        whole run gets (top of _process_frame), just re-anchored to timestamp_ns instead of
        the start of the recording. Used by _start_new_map_segment: rather than guessing
        how a re-initialized segment relates to the rest of the map (constant-velocity or
        IMU-integrated dead reckoning across a tracking-loss gap - which can be badly
        wrong, e.g. if the camera kept turning while it had nothing to track against),
        ORB-SLAM3's Atlas starts a re-init segment as its own independent coordinate frame
        and only ties it to the rest of the map later, once a real (visually verified)
        connection is found (see build_map.py's cross-segment map-merge handling) - a
        one-time rigid transform then, not a guess now. Falls back to an arbitrary
        (non-gravity-aligned) orientation, with a warning, if the IMU window right after
        timestamp_ns isn't static enough to read gravity's direction off it this way - which
        in practice is most of the time, since a re-init is usually triggered BY the camera
        moving (often why tracking failed in the first place).

        TRIED: a rotation-only gyro bridge as a fallback here (integrate just the rotation
        delta, via preintegration, from the stale ref_keyframe's own established orientation
        across the gap - deliberately narrower than a full position+velocity bridge, which
        was already found to hurt: see _start_new_map_segment's docstring). Measured WORSE on
        the same test case this was designed for (z_range 0.641m -> 0.741m, same fragmentation
        into 3 segments either way). Root cause: whatever orientation a re-init segment starts
        with is later overwritten wholesale anyway, either by a genuine visually-verified merge
        (a real SE3, not a guess) or by the fallback-stitch blend once no such merge is found -
        so a better *initial* guess only affects that segment's own transient tracking/BA
        quality before it gets superseded, not the final result, and on this dataset it just
        added noise. Reverted - see the session's IMU-init discussion for the full comparison
        across all these variants."""
        pose = np.eye(4)
        if not self.imu_enabled:
            return pose
        start = np.searchsorted(self.imu_measurements[:, 0], timestamp_ns)
        imu_init_mode, accel_samples = choose_imu_init_mode(
            self.imu_measurements[start:], self.cfg.imu.init_static_samples,
            search_samples=10 * self.cfg.imu.init_static_samples,
            gyro_static_threshold=self.cfg.imu.init_gyro_static_threshold,
            gravity_norm=self.cfg.imu.gravity_norm,
        )
        if imu_init_mode == "static":
            r_world_body0 = gravity_alignment_rotation(
                accel_samples,
                accel_bias=self._imu_bias_prior[:3] if self._imu_bias_prior is not None else None,
            )
            r_cam0_body = self.imu_calib.T_cam0_body[:3, :3]
            pose[:3, :3] = r_cam0_body @ r_world_body0.T
        else:
            print("  [atlas] new segment's leading IMU window isn't static - can't "
                  "gravity-align this segment's origin, proceeding with an arbitrary orientation")
        if self.imu_tight_fusion:
            # Unify onto the same authoritative path as the very first bootstrap (see
            # _process_frame): this crude accel-average guess (if any) never separates
            # true gravity direction from sensor bias, and an "arbitrary orientation"
            # fallback is even further off - schedule the same joint gravity+bias+
            # velocity dynamic solve (run_dynamic_imu_init, unconditional for stereo)
            # unconditionally, so THIS segment's real orientation always gets settled by
            # the same, more complete method instead of staying stuck on whatever this
            # method could (or couldn't) produce for its entire life until a visual merge
            # or fallback stitch happens to come along.
            self.imu_init_pending = True
        return pose

    def _imu_bootstrap_scale(self, pose_cw_b_relative: np.ndarray) -> float | None:
        """Scale factor for a two-view bootstrap, so a *re-initialized* map segment lands in
        the same units as the rest of the map instead of getting a fresh arbitrary scale from
        assumed_scene_depth_m (segments that disagree on scale can't be reconciled by any
        single global Sim3 - it shows up directly as ATE).

        Reference baseline between the seed keyframe and now, best source first:
          - mono+IMU: IMU propagation across the gap (metric, so the segment is metric too)
          - vision-only: the constant-velocity dead reckoning that seeded the segment, which
            carries the *previous* segment's scale forward
        Returns None during the very first bootstrap (no prior scale to inherit - that one
        legitimately defines the arbitrary unit via assumed_scene_depth_m) or when the motion
        is too small for the ratio to be trustworthy."""
        reference_pose_cw = self._imu_bridge_pose_cw()
        if reference_pose_cw is not None:
            ref_baseline = float(np.linalg.norm(camera_center(reference_pose_cw) - camera_center(self.ref_keyframe.pose_cw)))
        elif self.n_map_segments > 1 and self._segment_seed_speed:
            # Vision-only: the seed keyframe's pose came from dead reckoning at a known
            # per-frame speed (in the existing map's units), so the distance covered since
            # then is speed * frames elapsed. _predict_pose() can't be used here - the
            # segment restart deliberately reset the pose history to a single entry.
            frames_elapsed = max(1, self._bootstrap_wait_frames)
            ref_baseline = self._segment_seed_speed * frames_elapsed
        else:
            return None
        vision_baseline = float(np.linalg.norm(camera_center(pose_cw_b_relative)))
        min_baseline = getattr(self.cfg.tracking, "mono_imu_scale_min_baseline_m", 0.05)
        if ref_baseline < min_baseline or vision_baseline < 1e-6:
            return None
        return ref_baseline / vision_baseline

    def _start_new_map_segment(self, kpts, desc, depths, rect_l, feats, frame_id, timestamp_ns, image_path):
        """Atlas-style recovery: seed a fresh map segment from the current frame and rerun the
        two-view mono bootstrap against it. Seed pose is always a fresh, independent,
        gravity-aligned origin (see _gravity_aligned_origin_pose) - ORB-SLAM3's own
        Tracking::CreateMapInAtlas() does the same thing (verified directly against its
        source): it resets mpImuPreintegratedFromLastKF to a brand-new, zero-bias
        IMU::Preintegrated and clears mpLastKeyFrame/mpReferenceKF/mState entirely - a clean
        break, not a bridge. Don't guess how a re-init segment relates to the rest of the
        map; tie it back on later via a one-time rigid transform once a real connection is
        found (see build_map.py's cross-segment map-merge handling).

        Previously (tight_fusion only) this preferred _imu_bridge_pose_cw() - dead-reckoning
        the pose across the gap via double IMU integration - over the gravity-aligned
        origin, on the reasoning that it's "a real metric measurement, not a guess". Reverted:
        that's exactly the "guess how the segments relate" ORB-SLAM3 deliberately avoids, and
        double integration compounds any bias/gravity error quadratically in position over an
        unknown-length gap. Measured concretely on one real dataset: with the bridge
        preferred, a segment that (bridge-free) survives 388->593 as one continuous piece
        instead fractured into two (383->405, 405->593) - the bridged restart landed the
        tracker somewhere bad enough that it lost again almost immediately. Clearing
        _pending_imu_samples below (so _insert_keyframe's IMU-factor block sees nothing to
        add) matches the other half of ORB-SLAM3's reset: no factor spans the segment
        boundary either, since the two segments' poses aren't in the same coordinate frame to
        begin with - a "bridging" IMU factor between them would be constraining a relative
        pose that isn't physically meaningful until a genuine merge unifies the frames."""
        seed = self._gravity_aligned_origin_pose(timestamp_ns)
        seed_src = "gravity-aligned-independent-origin"
        # Save (don't just discard) the raw IMU samples spanning this transition, keyed by
        # (old ref_keyframe, this new segment's seed keyframe) - NOT added to
        # world_map.imu_factors (that would wrongly treat the two segments as already
        # sharing a coordinate frame - see this method's docstring on why the spanning
        # factor was removed), but kept separately so build_map.py's fallback-stitch pass
        # can use the real IMU physics, if this segment never finds a genuine visual
        # merge, to estimate the still-unknown transform between the two segments instead
        # of a blind constant-velocity guess.
        #
        # CHAIN-THROUGH-SHORT-LIVED-SEGMENTS: if the segment we're abandoning right now
        # (self.ref_keyframe's own segment) never grew past a couple of keyframes before
        # ALSO losing tracking, don't keep it as its own throwaway segment sandwiched
        # between two real anchors - it and whatever comes right before it (which the
        # keyframe count check below would need to look up) can end up reconciled through
        # two totally unrelated mechanisms (this one via fallback-stitch's blind IMU/CV
        # guess, the next segment via a completely independent, much-later real visual
        # merge onto a different part of the map), with nothing ever cross-checking that
        # they agree - measured directly as a ~2m visible seam in the final trajectory
        # exactly at this kind of boundary (a 1-keyframe segment immediately followed by
        # another tracking-loss into yet another segment). Discard the short-lived
        # segment's own keyframe(s) and chain the transition record back through it to
        # the real anchor before it, so whichever segment comes next bridges the WHOLE
        # combined gap in one shot instead of through a spurious middleman.
        short_segment_max_kf = getattr(self.cfg.tracking, "short_segment_discard_max_kf", 1)
        chained_anchor_id = None
        chained_samples = None
        if (
            self.imu_tight_fusion and self.ref_keyframe is not None
            and self.ref_keyframe.segment_id != 1 and short_segment_max_kf > 0
        ):
            abandoned_segment_id = self.ref_keyframe.segment_id
            abandoned_kf_ids = [
                kf_id for kf_id, kf in self.world_map.keyframes.items() if kf.segment_id == abandoned_segment_id
            ]
            if len(abandoned_kf_ids) <= short_segment_max_kf:
                prior_transition = next(
                    (
                        (old_id, samples) for old_id, new_id, samples in self.world_map.segment_transition_imu_samples
                        if new_id == min(abandoned_kf_ids)
                    ),
                    None,
                )
                if prior_transition is not None:
                    prior_anchor_id, prior_samples = prior_transition
                    gap_samples = (
                        [np.concatenate(self._pending_imu_samples, axis=0)] if self._pending_imu_samples else []
                    )
                    chained_samples = np.concatenate([prior_samples, *gap_samples], axis=0)
                    chained_anchor_id = prior_anchor_id
                    for kf_id in abandoned_kf_ids:
                        self.world_map.remove_keyframe(kf_id)
                    print(
                        f"  [atlas] segment {abandoned_segment_id} never grew past "
                        f"{len(abandoned_kf_ids)} keyframe(s) before losing tracking again - "
                        f"discarding it and chaining the fallback-stitch gap back to kf{prior_anchor_id}"
                    )
        if chained_anchor_id is not None:
            if len(chained_samples) >= 2:
                self.world_map.add_segment_transition_imu_samples(chained_anchor_id, frame_id, chained_samples)
        elif self.imu_tight_fusion and self.ref_keyframe is not None and self._pending_imu_samples:
            samples = np.concatenate(self._pending_imu_samples, axis=0)
            if len(samples) >= 2:
                self.world_map.add_segment_transition_imu_samples(self.ref_keyframe.frame_id, frame_id, samples)
        self._pending_imu_samples = []

        # Capture the dead-reckoning speed before resetting the pose history: the new
        # segment's bootstrap uses it to land in the existing map's units instead of a fresh
        # arbitrary scale (see _imu_bootstrap_scale).
        if len(self._pose_history) >= 2:
            self._segment_seed_speed = float(
                np.linalg.norm(camera_center(self._pose_history[-1]) - camera_center(self._pose_history[-2]))
            )
        else:
            self._segment_seed_speed = None

        if self.mono_bootstrap_pending and self.ref_keyframe is not None and not np.any(self.ref_keyframe.map_point_ids >= 0):
            # Re-seeding a segment whose seed keyframe never bootstrapped: drop that orphan
            # (zero map points) instead of leaving a stray pose in the trajectory.
            self.world_map.remove_keyframe(self.ref_keyframe.frame_id)

        self.n_map_segments += 1
        self._current_segment_id = self.n_map_segments
        kf = KeyFrame(
            frame_id=frame_id, timestamp_ns=timestamp_ns,
            keypoints=kpts, descriptors=desc, depths=depths, pose_cw=seed,
            image_path=str(image_path) if image_path else None,
            velocity=np.zeros(3) if self.imu_tight_fusion else None,
            # Inherited from the stale pre-loss ref_keyframe (not reset to zero) - TRIED
            # zero-reset first, reasoning it should match ORB-SLAM3's CreateMapInAtlas()
            # resetting to a fresh IMU::Bias(); reverted after measurement showed it made
            # things WORSE (one more tracking-loss segment on the same test dataset, in a
            # region that was fine in every other variant). Unlike pose (which genuinely
            # doesn't carry across an unknown-length, unknown-motion gap) and unlike the
            # cross-segment IMU factor (which would wrongly assume the two segments share a
            # coordinate frame), accelerometer/gyro bias is a slowly-varying property of the
            # physical sensor itself, not of the trajectory - carrying the last known
            # estimate forward is a reasonable warm start for the new segment's own
            # dynamic/periodic reinit to refine, and measurably better than forcing it to
            # relearn from an exact-zero start.
            imu_bias=(self.ref_keyframe.imu_bias if self.ref_keyframe is not None and self.ref_keyframe.imu_bias is not None
                      else self._imu_bias_prior),
            segment_id=self._current_segment_id,
        )
        self._insert_keyframe(kf, feats, rect_l)
        if self.mono_mode:
            # Mono needs a two-view essential-matrix bootstrap before this seed has any
            # map points - stereo doesn't: _insert_keyframe just above already
            # backprojected this keyframe's own stereo depth into real map points, so
            # normal per-frame tracking against it can resume next frame.
            self.mono_bootstrap_pending = True
        self._bootstrap_wait_frames = 0
        self._consecutive_lost = 0
        self.last_frame = kf
        self._record_tracked_pose(seed, reset=True)
        print(f"  [atlas] tracking lost - starting map segment {self.n_map_segments} at frame {frame_id} (seed pose from {seed_src})")
        return kf, True

    def _need_new_keyframe(
        self, translation: float, rotation_deg: float, inlier_mask: np.ndarray,
        kp_idx_list: list[int], tracked_ratio: float, depths: np.ndarray,
    ) -> tuple[bool, str]:
        """Ported from ORB-SLAM3's Tracking::NeedNewKeyFrame() (stereo/RGBD branch), ORed
        with mono's own untouched geometric-threshold rule (see below) - insertion is
        driven by how much of the current view the *map* still explains, not by raw
        displacement since the last keyframe. The original bug this replaces: a fixed
        `translation > 0.3m or rotation_deg > 8deg` rule fires on nearly every single
        frame during a fast in-place turn (rotation easily exceeds 8deg frame-to-frame),
        producing a burst of near-duplicate keyframes a few centimeters apart - measured in
        practice inserting keyframes 0.05s apart (20/s, i.e. every camera frame) during a
        turnaround, vs. a 0.5-0.6s median elsewhere. Those bursts were also the direct
        cause of ~90% of this dataset's "loop closures" actually spanning under 2m of real
        arc length - two keyframes 30+ ids apart that never actually left the same spot.

        CORRECTED (previously this docstring claimed ORB-SLAM3's mMaxFrames ceiling - c1a -
        was a missing content-agnostic "force a keyframe periodically" mechanism, and this
        method grew a keyframe_max_frames_ceiling OR-branch to restore it - checked directly
        against this repo's own vendored ORB-SLAM3 source, /home/dm/slam_ws/src/ORB_SLAM3/
        src/Tracking.cc:3610-3770, and that reading was WRONG): real ORB-SLAM3's actual gate
        is `((c1a||c1b||c1c) && c2) || c3 || c4` where c3/c4 are IMU-only conditions (always
        false for stereo/RGBD without IMU) and c1b = `(mnId >= mnLastKeyFrameId + mMinFrames)
        && bLocalMappingIdle` with mMinFrames=0 (confirmed via grep). In THIS pipeline, local
        mapping runs synchronously in the same call, i.e. it is unconditionally "idle" -
        making c1b unconditionally true for any frame after the last keyframe, which makes
        the WHOLE `(c1a||c1b||c1c)` clause unconditionally true regardless of c1a. c2 (the
        content-driven ratio/close-points test below) is therefore the ONLY thing that ever
        gates insertion in real ORB-SLAM3 too, once local mapping never falls behind -
        keyframe_max_frames_ceiling was accordingly removed again: it wasn't restoring
        missing ORB-SLAM3 behavior, it was adding behavior ORB-SLAM3 itself doesn't have
        under these conditions.

        This repo's OWN vendored ORB-SLAM3 has a prior, empirically-tuned "SPLG FullReplace"
        experiment (Frame::mbSPLGFullReplace, Tracking.cc:3688) that raises thRefRatio from
        the literal upstream 0.75 to 0.9 specifically when running on SuperPoint+LightGlue
        features, on the theory that LightGlue's wide-baseline matching keeps a frame
        tracking well against an older reference keyframe for longer than ORB's matching
        would, so the ratio test needs a tighter (higher) bar to still discriminate.
        keyframe_ref_ratio's default below is raised to 0.9 to match that prior art.

        BUT (measured directly on KITTI00 via temporary instrumentation logging n_inliers/
        n_ref_matches per frame, both with the map already sparse and with density
        artificially seeded via keyframe_translation_fallback_m to rule out a cold-start
        effect): 0.9 does NOT fix KITTI's under-firing, and neither does a much larger value.
        With a healthy, externally-seeded map, n_inliers sits at ~1.87x n_ref_matches on
        average (e.g. medians 432 vs 226) - so ANY ratio at or below ~0.9 structurally never
        triggers (0/643 samples), and even sweeping the ratio up to the observed 1.87 mean
        only triggers ~52% of frames, DROPPING OFF SMOOTHLY either side of that value with no
        sign of a genuine bimodal "map explains this view / doesn't" split. That shape means
        n_inliers/n_ref_matches behaves like noise scattered around a roughly constant mean
        on this dataset+front-end combination, not a signal that discriminates "just moved
        past this keyframe's useful range" from "still well inside it" the way ORB-SLAM3's
        design intends - no fixed threshold search over this ratio can turn it into a
        reliable trigger here. KITTI therefore still needs keyframe_translation_fallback_m as
        a working density floor (see configs/kitti_00_run.yaml) - this is a documented, known
        limitation of the content-driven signal on this dataset, not a threshold-tuning gap.

        nMinObs is a literal ORB-SLAM3 constant, not exposed as config - same as upstream,
        it's an algorithm internal rather than something a dataset config is expected to
        retune. The close-point-starvation constants are NOT ported as upstream's literal
        absolute 100/70, though - see need_close_points below.

        Stereo used to have a separate keyframe_legacy_geometric_only opt-in that bypassed
        all of the above for a raw geometric-threshold rule (kept around after this method
        was first ported, as a reproduction tool for diagnosing a dataset regression against
        pre-port behavior). Removed once that investigation was done and stereo had no
        remaining case where it beat the content-driven+fallback rule below - mono keeps its
        own geometric rule since, unlike stereo, its keyframe cadence directly gates how much
        parallax is available to triangulate new map points, not just density/robustness."""
        if self.mono_mode:
            need = (
                translation > self.cfg.tracking.keyframe_translation_m
                or rotation_deg > self.cfg.tracking.keyframe_rotation_deg
                or tracked_ratio < self.cfg.tracking.keyframe_min_tracked_ratio
            )
            return need, ("mono_geometric" if need else "none")

        n_kfs = len(self.world_map.keyframes)
        min_obs = 2 if n_kfs <= 2 else 3
        n_ref_matches = sum(
            1 for pid in self.ref_keyframe.map_point_ids
            if pid >= 0 and int(pid) in self.world_map.map_points
            and self.world_map.map_points[int(pid)].num_observations() >= min_obs
        )
        n_inliers = int(inlier_mask.sum())

        inlier_kp_idx = {kp_idx_list[i] for i in range(len(kp_idx_list)) if inlier_mask[i]}
        valid_depth_idx = set(np.flatnonzero(~np.isnan(depths)).tolist())
        n_tracked_close = len(inlier_kp_idx & valid_depth_idx)
        n_non_tracked_close = len(valid_depth_idx - inlier_kp_idx)
        # ORB-SLAM3's literal thresholds here are ABSOLUTE counts (100/70). Measured on
        # this pipeline: they're satisfied on ~99.8% of insertions (close-point starvation
        # has effectively replaced the coverage-ratio signal as the dominant trigger,
        # rather than being the secondary/occasional one ORB-SLAM3 intends) - so tried
        # scaling both relative to how many close-depth keypoints this frame actually
        # offers (10%/7%, same ratio the literal 100/70 implies against ORB-SLAM3's own
        # ~1000-feature assumption). That backfired: keyframes dropped 414->191, but so did
        # real trajectory quality - path length 207.7m->157.1m, Z range 4.9m->8.0m *worse*,
        # and the final keyframe (near the very end of the recording) ended up positioned
        # back near the start's climb area instead of out at the trajectory's real distal
        # end - a much bigger, real drift/coverage regression, not just sparser sampling of
        # the same path. Reverted to the literal ORB-SLAM3 constants: even though they're
        # almost always satisfied here, they still produce more, better-placed keyframes
        # than the "improved" relative version - denser triangulation support evidently
        # matters more for this pipeline's tracking accuracy than matching ORB-SLAM3's
        # original *proportion* of frames that trigger on this specific signal.
        need_close_points = n_tracked_close < 100 and n_non_tracked_close > 70

        # 0.9, not upstream's literal 0.75 - see this method's docstring (this repo's own
        # vendored ORB-SLAM3 already validated 0.9 specifically for SuperPoint+LightGlue via
        # Frame::mbSPLGFullReplace/Tracking.cc:3688 - LightGlue's wide-baseline robustness
        # keeps mnMatchesInliers above the ratio bar for longer than ORB's matching would,
        # so the literal ORB-tuned 0.75 under-triggers on this feature front-end).
        ref_ratio = 0.4 if n_kfs < 2 else getattr(self.cfg.tracking, "keyframe_ref_ratio", 0.9)
        ref_ratio_triggered = n_inliers < n_ref_matches * ref_ratio
        need = n_inliers > self.cfg.tracking.min_inlier_matches and (ref_ratio_triggered or need_close_points)
        # Density floor, OFF by default (0 disables each). Content-based insertion alone
        # starved tracking robustness on a fast-motion dataset: sparser reference keyframes
        # meant Tier1/Tier2 lost tracking more often during a hard excursion, tripling how
        # many Atlas re-inits fired (1 -> 3) and leaving 2 of the 3 resulting segments
        # permanently unmerged (no other segment ever found a matching candidate for them)
        # - confirmed by a direct A/B: OR-ing the old geometric thresholds back in as a
        # diagnostic dropped re-inits back to 1 and let it merge cleanly.
        #
        # Translation-only fallback ALONE didn't reproduce that fix (tried it first: barely
        # changed anything, 172->173 keyframes) - the old rotation_deg threshold was
        # apparently doing most of the actual work for this dataset (a hard section that's
        # turning while moving, not just translating).
        #
        # Unguarded (no minimum-translation requirement) as of this session: the original
        # port had gated this on also exceeding a small minimum translation, out of concern
        # for the port's own original bug - a stationary in-place turn (e.g. a tripod/gimbal
        # with a fixed optical center) blowing past a fixed rotation threshold on nearly
        # every single frame, 20 keyframes/s measured, since a fast enough constant angular
        # rate re-triggers "rotated > threshold since ref_keyframe" again on the very next
        # frame once ref_keyframe resets to whatever was just inserted. Re-tested directly on
        # a real handheld realsense recording with the gate removed and the threshold at
        # 15deg: 26 rotation_fallback insertions over 1080 frames, none consecutive, every
        # one alongside genuine non-zero translation (0.9-21cm) - handheld motion essentially
        # never holds a truly fixed optical center, so the runaway case never materialized in
        # practice here. The theoretical risk remains real for a genuinely fixed-pivot
        # rotation (tripod/gimbal) - if this ever regresses on such a setup, the fix is to
        # reintroduce a translation (or, more robustly, a max-expansions-style per-check
        # cap) gate, not to lower keyframe_rotation_fallback_deg, which doesn't address the
        # root cause (see this file's git history around this comment).
        translation_fallback_m = self._effective_keyframe_translation_fallback_m
        translation_triggered = translation_fallback_m > 0 and translation > translation_fallback_m
        rotation_fallback_deg = getattr(self.cfg.tracking, "keyframe_rotation_fallback_deg", 0.0)
        rotation_triggered = rotation_fallback_deg > 0 and rotation_deg > rotation_fallback_deg
        need = need or translation_triggered or rotation_triggered
        if not need:
            reason = "none"
        elif (translation_triggered or rotation_triggered) and not (ref_ratio_triggered or need_close_points):
            reason = "translation_fallback" if translation_triggered else "rotation_fallback"
        elif ref_ratio_triggered and need_close_points:
            reason = "ref_ratio+close_points"
        elif ref_ratio_triggered:
            reason = "ref_ratio"
        else:
            reason = "close_points"
        return need, reason

    def _try_track_against_ref(self, matches, kpts: np.ndarray, desc: np.ndarray, rect_l: np.ndarray):
        """Tier 1 (primary): PnP against the reference keyframe's LightGlue matches, then
        expand correspondences via the local map (search_local_map) and re-solve. Returns
        (pose_cw, point_ids, kp_idx_list, inlier_mask, num_local_map_matches), or None if
        any stage fails - the caller then tries the motion-model fallback tier.

        Deliberately does NOT require the reference-keyframe-only correspondence count to
        already clear min_inlier_matches before attempting the initial PnP (only
        solve_pnp_ransac's own much lower absolute floor - 6 points, the minimum PnP needs
        at all - gates this first solve; the real min_inlier_matches bar is still enforced
        below, after local-map expansion). ORB-SLAM3's own primary tracking path
        (TrackWithMotionModel/TrackReferenceKeyFrame) works the same way: get *a* rough
        pose cheaply first, then let TrackLocalMap's full local-map projection (not just
        one keyframe's own points) do the real work of building a robust correspondence
        set. Gating the expansion behind the reference keyframe's own point count first -
        this function's original design - makes a single thin reference keyframe a single
        point of failure: measured in practice, a reference keyframe whose own map-point
        count had been quietly declining for the prior ~10 keyframes (from ~160 down to
        41, as the scene's texture thinned out) failed this gate outright and never got a
        chance to lean on the local map window's much richer point set, which is exactly
        the situation TrackLocalMap's unconditional expansion exists to cover."""
        obj_pts, img_pts, point_ids, kp_idx_list = [], [], [], []
        for ref_i, cur_i in matches:
            mp_id = self.ref_keyframe.map_point_ids[ref_i]
            if mp_id >= 0:
                obj_pts.append(self.world_map.map_points[mp_id].position)
                img_pts.append(kpts[cur_i])
                point_ids.append(int(mp_id))
                kp_idx_list.append(int(cur_i))

        ok, pose_init, _ = solve_pnp_ransac(
            np.asarray(obj_pts), np.asarray(img_pts), self.rectifier.K_rect,
            reproj_threshold_px=self.cfg.tracking.pnp_reproj_threshold_px,
        )
        if not ok:
            return None

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
            return None
        return pose_cw, point_ids, kp_idx_list, inlier_mask, len(lm_obj)

    def _try_motion_model_tracking(self, kpts: np.ndarray, desc: np.ndarray, rect_l: np.ndarray, predicted: np.ndarray | None, depths: np.ndarray | None = None):
        """Tier 2 (fallback): projects the existing local map into the frame using a
        constant-velocity-predicted pose (_predict_pose, computed by the caller so it can
        also be used to advance the dead-reckoning chain on total failure - see
        _process_frame) and matches by projected position + descriptor similarity
        (search_local_map), instead of LightGlue graph-matching against the single
        reference keyframe. Tolerates motion blur / large appearance change much better
        (needs only "a similar descriptor near where geometry predicts", not a full
        LightGlue match between two images) - but still needs the local map to actually
        cover roughly where the camera is, so if the reference keyframe has near-zero
        covisibility with genuinely new/unmapped territory, this naturally yields too few
        candidates and returns None, falling through to relocalization exactly as before
        this tier existed. Returns the same shape as _try_track_against_ref."""
        debug = getattr(self.cfg.tracking, "motion_model_debug", False)
        if predicted is None:
            if debug:
                print("    [motion-model] no pose history yet")
            return None
        if debug:
            pred_trans, pred_rot = pose_delta(self.ref_keyframe.pose_cw, predicted)
            print(f"    [motion-model] predicted vs ref_kf{self.ref_keyframe.frame_id}: translation={pred_trans:.4f} rotation={pred_rot:.2f}deg (n_history={len(self._pose_history)})")

        window = self.world_map.covisible_window(
            self.ref_keyframe.frame_id, window_size=self.cfg.tracking.local_map_window_size,
            min_shared=self.cfg.tracking.local_map_min_shared,
        )
        point_ids = gather_local_map_point_ids(self.world_map, window)
        if not point_ids:
            if debug:
                print(f"    [motion-model] ref_kf{self.ref_keyframe.frame_id}: covisible window empty (window={window})")
            return None
        if debug:
            positions = np.array([self.world_map.map_points[pid].position for pid in point_ids])
            uv, valid = project_points(predicted, positions, self.rectifier.K_rect)
            w, h = rect_l.shape[1], rect_l.shape[0]
            in_bounds = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            print(f"    [motion-model] {len(point_ids)} candidates, {int(in_bounds.sum())} project in-bounds under predicted pose")

        # Search radius derived from geometry instead of a hand-picked per-dataset pixel
        # constant: project how far the predicted step's own translation would move a
        # point at the scene's current typical depth (apparent_px_motion =
        # step_translation_m * fx / depth_m), then add a safety margin - this is exactly
        # why KITTI (fast, ~10-15m/s motion) needed a much wider radius than EuRoC (walking-
        # pace/hover) with the old fixed constants: apparent pixel motion scales with real-
        # world speed for the same focal length/depth, so deriving it from the actual
        # predicted step reproduces that difference automatically instead of needing a
        # separate hand-tuned constant per dataset. 1.5x/+8px/[8,50] are universal safety-
        # margin/sanity-rail constants (not re-tuned per dataset - same role as e.g.
        # nMinObs elsewhere in this file), not scene-scale tuning.
        step_trans_m = pose_delta(self._pose_history[-1], predicted)[0] if self._pose_history else 0.0
        valid_depths = depths[~np.isnan(depths)] if depths is not None else np.empty(0)
        scene_depth_m = float(np.median(valid_depths)) if valid_depths.size > 0 else 10.0
        apparent_px_motion = step_trans_m * self.rectifier.fx_rect / max(scene_depth_m, 0.5)
        base_radius = float(np.clip(apparent_px_motion * 1.5 + 8.0, 8.0, 50.0))
        growth = 0.25
        max_radius = min(base_radius * 3.0, 150.0)
        radius_px = min(max_radius, base_radius * (1.0 + growth * self._consecutive_lost))

        used_kp_mask = np.zeros(len(kpts), dtype=bool)
        mm_obj, mm_img, mm_pids, mm_kpidx = search_local_map(
            self.world_map, point_ids, predicted, kpts, desc, self.rectifier.K_rect,
            image_size=(rect_l.shape[1], rect_l.shape[0]), used_kp_mask=used_kp_mask,
            radius_px=radius_px,
            desc_threshold=getattr(self.cfg.tracking, "motion_model_desc_threshold", 0.75),
        )
        if len(mm_obj) < self.cfg.tracking.min_inlier_matches:
            if debug:
                print(f"    [motion-model] ref_kf{self.ref_keyframe.frame_id}: {len(point_ids)} candidates, only {len(mm_obj)} matched (need {self.cfg.tracking.min_inlier_matches})")
            return None

        ok, pose_cw, inlier_mask = solve_pnp_ransac(
            np.asarray(mm_obj), np.asarray(mm_img), self.rectifier.K_rect,
            reproj_threshold_px=self.cfg.tracking.pnp_reproj_threshold_px,
        )
        if not ok or inlier_mask.sum() < self.cfg.tracking.min_inlier_matches:
            if debug:
                print(f"    [motion-model] PnP failed or too few inliers ({0 if not ok else int(inlier_mask.sum())}/{len(mm_obj)})")
            return None
        if debug:
            print(f"    [motion-model] SUCCESS: {int(inlier_mask.sum())}/{len(mm_obj)} inliers")
        return pose_cw, mm_pids, mm_kpidx, inlier_mask, len(mm_obj)

    def _insert_keyframe(
        self, kf: KeyFrame, feats: dict, rect_left_img: np.ndarray,
        ref_matches: list[tuple[int, int]] | None = None,
    ) -> None:
        if self.imu_tight_fusion and self.ref_keyframe is not None and self._pending_imu_samples:
            samples = np.concatenate(self._pending_imu_samples, axis=0)
            prev_bias_vec = self.ref_keyframe.imu_bias
            kf.imu_bias = prev_bias_vec if prev_bias_vec is not None else np.zeros(6)
            # Only propagate a velocity when this segment's world frame has an actual
            # gravity-validated "up" direction (imu_init_pending False) and the previous
            # keyframe already carries a real (not fabricated) velocity - preint.predict()
            # subtracts a fixed n_gravity=(0,0,-g) assuming the pose passed in is expressed
            # in a frame where Z really is up, which is exactly NOT guaranteed while pending
            # (see _gravity_aligned_origin_pose/_start_new_map_segment: a fresh segment's
            # origin can be an arbitrary, unverified orientation until dynamic init succeeds).
            # Confirmed as a real bug on one dataset: a 2-keyframe orphan Atlas segment never
            # reached init_dynamic_window_kf, so imu_init_pending stayed True its whole life,
            # yet this block still ran unconditionally on its 2nd keyframe - some of the
            # true gravity vector wasn't cancelled in that unverified frame and leaked into
            # the integrated velocity (2.96 m/s, well above plausible walking speed), which
            # then got fed straight into build_map.py's fallback-stitch IMU bridge as the
            # starting velocity for the NEXT segment, producing a spurious ~1.3m Z jump over
            # a 0.6s gap. Leaving kf.velocity at its default (None) here is the correct,
            # honest "not yet known" signal - every consumer (this class's own
            # _imu_bridge_pose_cw, build_map.py's fallback-stitch bridge, local_ba's
            # _ensure_imu_state) already treats a None velocity as "not usable yet" rather
            # than silently trusting a fabricated zero or a gravity-corrupted guess.
            if not self.imu_init_pending and self.ref_keyframe.velocity is not None:
                prev_bias = bias_from_vector(prev_bias_vec)
                preint = preintegrate(samples, prev_bias, self._imu_params)
                prev_body_pose = pose_cw_to_body_gtsam(self.ref_keyframe.pose_cw, self.imu_calib.T_cam0_body)
                prev_state = gtsam.NavState(prev_body_pose, self.ref_keyframe.velocity)
                predicted = preint.predict(prev_state, prev_bias)
                kf.velocity = predicted.velocity()
            self.world_map.add_imu_factor(self.ref_keyframe.frame_id, kf.frame_id, samples)
        self._pending_imu_samples = []

        if self.global_extractor is not None:
            kf.global_descriptor = self.global_extractor.extract(rect_left_img)

        self.world_map.add_keyframe(kf)
        self.n_keyframes_inserted += 1

        if kf.image_path is None:
            # No on-disk file to re-read for a future relocalization attempt (bag-sourced
            # frame) - cache the features already extracted for this keyframe now instead
            # of _cached_keyframe_feats's usual lazy re-extract-from-disk path.
            self.reloc_feats_cache[kf.frame_id] = feats

        for kp_idx, mp_id in enumerate(kf.map_point_ids):
            if mp_id >= 0:
                self.world_map.add_observation(int(mp_id), kf.frame_id, kp_idx)

        if self.mono_mode:
            if ref_matches is not None and self.ref_keyframe is not None:
                self._triangulate_new_mono_points(kf, ref_matches)
        else:
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

    def _triangulate_new_mono_points(self, kf: KeyFrame, ref_matches: list[tuple[int, int]]) -> None:
        """Monocular new-map-point creation: two-view-triangulates every ref<->cur match
        that isn't already a map point on either side, using the two keyframes' own
        (already PnP/essential-matrix-estimated) poses - the direct replacement for the
        stereo/depth-sensor path's single-view backproject, since mono has no per-frame
        depth at all."""
        ref_kf = self.ref_keyframe
        ref_idx, cur_idx = [], []
        for ref_i, cur_i in ref_matches:
            if ref_kf.map_point_ids[ref_i] >= 0 or kf.map_point_ids[cur_i] >= 0:
                continue
            ref_idx.append(ref_i)
            cur_idx.append(cur_i)
        if not ref_idx:
            return
        ref_idx = np.array(ref_idx)
        cur_idx = np.array(cur_idx)

        points_world, valid = triangulate_points(
            ref_kf.pose_cw, kf.pose_cw, self.rectifier.K_rect,
            ref_kf.keypoints[ref_idx], kf.keypoints[cur_idx],
            min_parallax_deg=getattr(self.cfg.tracking, "mono_min_parallax_deg", 1.0),
        )
        if getattr(self.cfg.tracking, "mono_debug_triangulation", False):
            translation = float(np.linalg.norm(kf.pose_wc()[:3, 3] - ref_kf.pose_wc()[:3, 3]))
            print(f"    [mono new-points] kf{kf.frame_id}<-kf{ref_kf.frame_id}: candidates={len(ref_idx)} accepted={int(valid.sum())} baseline={translation:.4f}")
        for local_i in np.nonzero(valid)[0]:
            ri, ci = int(ref_idx[local_i]), int(cur_idx[local_i])
            point_id = self.world_map.new_map_point(
                points_world[local_i], kf.descriptors[ci], created_at_kf_count=self.n_keyframes_inserted,
            )
            self.world_map.add_observation(point_id, ref_kf.frame_id, ri)
            self.world_map.add_observation(point_id, kf.frame_id, ci)
            ref_kf.map_point_ids[ri] = point_id
            kf.map_point_ids[ci] = point_id

    def _cached_keyframe_feats(self, kf_id: int) -> dict:
        if kf_id not in self.reloc_feats_cache:
            kf = self.world_map.keyframes[kf_id]
            img = cv2.imread(kf.image_path, cv2.IMREAD_GRAYSCALE)
            rect_l, _ = self.rectifier.rectify(img, img)
            self.reloc_feats_cache[kf_id] = self._mask_dynamic_keypoints(self.splg.extract(rect_l), rect_l)
        return self.reloc_feats_cache[kf_id]

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
            # EPnP + RANSAC can occasionally converge to a wildly wrong pose despite
            # reporting a healthy inlier count and ratio - a repetitive/near-degenerate
            # point configuration (e.g. a section of corridor prone to perceptual aliasing)
            # can admit more than one self-consistent-looking pose. Seen in practice on one
            # keyframe that alternately relocalized correctly (<2m from itself) and wildly
            # wrong (16-39m from itself) across consecutive frames, all past a `> 50m from
            # cand_kf` sanity check that was originally sized for catching a *literally*
            # nonsensical solve (millions of meters away), not this more subtle failure.
            # Two checks, tightened from that first attempt:
            #  1. distance-from-candidate bounded by the stereo pipeline's own valid-depth
            #     range (self.cfg.stereo.max_depth_m) - a genuine PnP solve against
            #     cand_kf's map points puts the camera within roughly that range of them,
            #     and cand_kf's own camera is roughly at the near end of what it observes.
            #  2. inlier ratio (not just count) - mirrors verify_loop_candidate's own
            #     min_inlier_ratio check, for the same reason: a healthy absolute count can
            #     still be a small, cherry-picked slice of a much larger, noisier attempted
            #     correspondence set.
            # Neither is a complete fix for genuine aliasing (a wrong-but-plausible solve
            # can still pass both), but together they catch most of what was observed.
            if float(np.linalg.norm(camera_center(pose_cw) - camera_center(cand_kf.pose_cw))) > self._effective_max_depth_m:
                continue
            if n_inliers / len(obj_pts) < 0.3:
                continue

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
        self.world_map.add_relocalization_event(frame_id, cand_kf.frame_id, pose_cw)
        if self.imu_tight_fusion:
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
        # Reset (not append): this jump is non-local in the map, not continuous motion -
        # a constant-velocity prediction spanning old-position -> jumped-to-position would
        # be nonsense. _predict_pose() stays disabled until a second real frame accumulates
        # after the jump.
        self._record_tracked_pose(pose_cw, reset=True)
        self.ref_keyframe = cand_kf
        self.ref_feats = cand_feats
        return frame, False
