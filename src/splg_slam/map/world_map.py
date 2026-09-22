from collections import defaultdict

import numpy as np

from splg_slam.map.keyframe import KeyFrame
from splg_slam.map.map_point import MapPoint


class WorldMap:
    """Container for keyframes and 3D map points, with a covisibility graph
    (keyframe_id -> {other_keyframe_id: shared_point_count}) used by local BA
    windowing and loop-closure candidate filtering."""

    def __init__(self):
        self.keyframes: dict[int, KeyFrame] = {}
        self.map_points: dict[int, MapPoint] = {}
        self.covisibility: dict[int, dict[int, int]] = defaultdict(dict)
        self.loop_edges: list[tuple[int, int, np.ndarray, int]] = []  # (kf_a, kf_b, rel, num_inliers)
        self.imu_factors: list[tuple[int, int, np.ndarray]] = []  # (kf_a, kf_b, Nx7 [t_s,wx,wy,wz,ax,ay,az])
        # Raw IMU samples spanning an Atlas re-init transition: (old_kf_id, new_kf_id, Nx7
        # samples) - kept SEPARATE from imu_factors (never added there) because the two
        # keyframes are in different, not-yet-related coordinate frames; imu_factors'
        # consumer (local_bundle_adjustment) adds a same-frame CombinedImuFactor for every
        # entry, which would be a meaningless (or actively wrong) constraint here. Recorded
        # so build_map.py's fallback-stitch pass can use the real IMU physics (not a
        # constant-velocity guess) to estimate the still-unknown merge transform between the
        # two segments when a genuine visual re-identification never succeeds - see
        # tracker.py's _start_new_map_segment for where this gets populated.
        self.segment_transition_imu_samples: list[tuple[int, int, np.ndarray]] = []
        # Every successful relocalization jump: (frame_id, anchor_kf_id, rel), where rel is
        # the pose *relative to* anchor_kf_id (the matched candidate keyframe) at the
        # moment of relocalization - not an absolute pose, and not necessarily a keyframe
        # itself, since relocalization only re-anchors ref_keyframe and doesn't insert one.
        # Stored relative to anchor_kf_id (not absolute) so that if anchor_kf_id's own pose
        # later gets corrected (pose-graph optimization, a loop-edge merge, global BA - all
        # of which happen far more often than relocalization itself), recomputing
        # `rel @ anchor_kf.pose_cw` with the anchor's *current* pose keeps this event
        # positioned consistently with the rest of the map instead of going stale as a
        # frozen snapshot. Kept for trajectory-status visualization (plot_xy.py).
        self.relocalization_events: list[tuple[int, int, np.ndarray]] = []
        # Every Atlas cross-segment merge: (new_kf_id, cand_kf_id) - the two keyframes
        # whose visual match triggered transform_segment's rigid weld (see build_map.py's
        # try_close_loops). Deliberately NOT added to loop_edges: a merge has no residual
        # "drift" for pose-graph optimization to correct (the whole segment was already
        # rigidly re-placed to match exactly), so it isn't a BetweenFactor-style edge - it's
        # a distinct event kind, kept separately for trajectory-status visualization
        # (plot_xy.py, colored apart from same-segment loop closures).
        self.segment_merges: list[tuple[int, int]] = []
        # Every low-confidence fallback stitch: (anchor_kf_id, [stitched_kf_ids]) - applied
        # to an Atlas segment that, even after exhaustive orphan-segment reconciliation
        # (see build_map.py), never found a genuine visual re-identification match. Rather
        # than leave it permanently isolated in its own unrelated coordinate frame (losing
        # continuity of the final trajectory), it's welded on with a *zero-displacement*
        # guess - "assume the camera was still roughly at anchor_kf_id's position when this
        # segment's first keyframe was captured" - the same "keep going on a degraded
        # estimate rather than stop" philosophy a plain accumulating VO/pose-graph backend
        # (no multi-segment concept at all) is forced into by construction. This is NOT a
        # verified closure: no visual evidence backs the assumed position, only continuity.
        # The full stitched kf_id list is recorded (not just the first one) because
        # transform_segment relabels segment_id to the surviving one, so segment_id alone
        # can no longer distinguish these from originally-main-segment keyframes
        # afterward. Kept separately from segment_merges so trajectory-status
        # visualization (plot_xy.py) can flag it as distinctly lower-confidence.
        self.fallback_stitches: list[tuple[int, list[int]]] = []
        self.point_probation: dict[int, int] = {}  # point_id -> keyframe count at creation
        self.frame_processing_times_s: list[float] = []  # per-input-frame wall-clock time (seconds)
        self._next_point_id = 0

    def add_loop_edge(self, kf_id_a: int, kf_id_b: int, relative_pose_a_from_b: np.ndarray, num_inliers: int) -> None:
        self.loop_edges.append((kf_id_a, kf_id_b, relative_pose_a_from_b, num_inliers))

    def add_segment_merge(self, new_kf_id: int, cand_kf_id: int) -> None:
        self.segment_merges.append((new_kf_id, cand_kf_id))

    def add_fallback_stitch(self, anchor_kf_id: int, stitched_kf_ids: list[int]) -> None:
        self.fallback_stitches.append((anchor_kf_id, list(stitched_kf_ids)))

    def add_imu_factor(self, kf_id_a: int, kf_id_b: int, samples: np.ndarray) -> None:
        self.imu_factors.append((kf_id_a, kf_id_b, samples))

    def add_segment_transition_imu_samples(self, old_kf_id: int, new_kf_id: int, samples: np.ndarray) -> None:
        self.segment_transition_imu_samples.append((old_kf_id, new_kf_id, samples))

    def add_relocalization_event(self, frame_id: int, anchor_kf_id: int, pose_cw: np.ndarray) -> None:
        """`pose_cw` is the absolute pose solved at relocalization time - stored relative to
        `anchor_kf_id` (the matched candidate keyframe)'s pose at that same moment, so it can
        be recomputed against the anchor's *current* pose later (see relocalization_events'
        docstring above)."""
        from splg_slam.geometry.pose_utils import invert_pose

        anchor_pose_cw = self.keyframes[anchor_kf_id].pose_cw
        rel = pose_cw @ invert_pose(anchor_pose_cw)
        self.relocalization_events.append((frame_id, anchor_kf_id, rel))

    def relocalization_event_pose(self, event: tuple[int, int, np.ndarray]) -> np.ndarray | None:
        """Recomputes an event's absolute pose against anchor_kf_id's *current* pose - see
        relocalization_events' docstring. Returns None if the anchor keyframe no longer
        exists (e.g. culled)."""
        _frame_id, anchor_kf_id, rel = event
        anchor_kf = self.keyframes.get(anchor_kf_id)
        if anchor_kf is None:
            return None
        return rel @ anchor_kf.pose_cw

    def add_keyframe(self, kf: KeyFrame) -> None:
        self.keyframes[kf.frame_id] = kf

    def transform_keyframes(self, kf_ids: set[int], transform: np.ndarray) -> None:
        """Bulk rigid-transforms every keyframe pose in `kf_ids` and every map point whose
        observations are a strict subset of `kf_ids` by `transform` (4x4, maps a point/pose
        from `kf_ids`'s own local world into the target world). Shared by transform_segment
        (whole Atlas segment) and a same-segment long-arc loop closure that welds just the
        unanchored excursion between the closure's two endpoints (see build_map.py's
        try_close_loops) - both are "two chunks of map that each internally agree with
        themselves but not with each other yet" problems, just with `kf_ids` picked
        differently (by segment_id vs. by keyframe-id range since the last loop anchor).

        A map point only moves if *every* keyframe observing it is in `kf_ids` - one already
        fused across the boundary (mixed observations) is left alone, since a single rigid
        transform can't correctly move a point that's simultaneously correct in two
        different coordinate frames without first being fused into one."""
        from splg_slam.geometry.pose_utils import invert_pose

        transform_inv = invert_pose(transform)
        for kf_id in kf_ids:
            kf = self.keyframes[kf_id]
            kf.pose_cw = kf.pose_cw @ transform_inv
        for mp in self.map_points.values():
            observers = set(mp.observations.keys())
            if observers and observers <= kf_ids:
                mp.position = (transform[:3, :3] @ mp.position) + transform[:3, 3]

    def transform_segment(self, segment_id: int, transform: np.ndarray, merge_into_segment_id: int) -> None:
        """Bulk rigid-transforms every keyframe pose and map point position belonging to
        `segment_id` (an Atlas map segment - see KeyFrame.segment_id) by `transform` (4x4,
        maps a point/pose from that segment's own local world into the target world), then
        relabels them as `merge_into_segment_id`.

        Used to weld a segment onto the rest of the map in one shot once a real (visually
        verified) connection to it is found, ORB-SLAM3-Atlas-style - rather than nudging it
        into place via a loop-closure pose-graph edge, which only works for correcting
        *drift* (both sides already agree on roughly where they are). An Atlas segment's
        first keyframe never claimed to share a coordinate frame with the rest of the map
        (see OfflineMapper._start_new_map_segment) - there's no "drift" to speak of, just
        two unrelated coordinate frames that need a one-time rigid alignment."""
        seg_kf_ids = {kf_id for kf_id, kf in self.keyframes.items() if kf.segment_id == segment_id}
        self.transform_keyframes(seg_kf_ids, transform)
        for kf_id in seg_kf_ids:
            self.keyframes[kf_id].segment_id = merge_into_segment_id

    def new_map_point(self, position: np.ndarray, descriptor: np.ndarray, created_at_kf_count: int | None = None) -> int:
        point_id = self._next_point_id
        self.map_points[point_id] = MapPoint(position=position, descriptor=descriptor)
        self._next_point_id += 1
        if created_at_kf_count is not None:
            self.point_probation[point_id] = created_at_kf_count
        return point_id

    def merge_map_points(self, keep_id: int, remove_id: int) -> None:
        """Reassigns every observation of `remove_id` onto `keep_id` (skipping a keyframe
        that already sees `keep_id` through a different keypoint - a rare local mismatch,
        not worth resolving here) and deletes `remove_id`. Used by loop-closure fusion,
        where the two sides of a closed loop turn out to have independently triangulated
        the same physical point as two different MapPoints."""
        if keep_id == remove_id:
            return
        keep_mp = self.map_points.get(keep_id)
        remove_mp = self.map_points.get(remove_id)
        if keep_mp is None or remove_mp is None:
            return
        for kf_id, kp_idx in list(remove_mp.observations.items()):
            if kf_id in keep_mp.observations:
                continue
            self.add_observation(keep_id, kf_id, kp_idx)
        self.remove_map_point(remove_id)

    def remove_observation(self, point_id: int, keyframe_id: int) -> None:
        """Drops just one keyframe's observation of a point (unlike remove_map_point,
        which deletes the whole point). Used for post-BA outlier pruning: an observation
        whose reprojection error is still large after optimization is almost certainly a
        mismatch, but the point itself may still be well-supported by its other
        observations. If this was the last observation, the point is deleted too."""
        mp = self.map_points.get(point_id)
        if mp is None or keyframe_id not in mp.observations:
            return
        kp_idx = mp.observations.pop(keyframe_id)
        kf = self.keyframes.get(keyframe_id)
        if kf is not None and kf.map_point_ids[kp_idx] == point_id:
            kf.map_point_ids[kp_idx] = -1
        for other_kf in mp.observations:
            self._decrement_covisibility(keyframe_id, other_kf)
            self._decrement_covisibility(other_kf, keyframe_id)
        if not mp.observations:
            self.map_points.pop(point_id, None)
            self.point_probation.pop(point_id, None)

    def remove_map_point(self, point_id: int) -> None:
        mp = self.map_points.pop(point_id, None)
        self.point_probation.pop(point_id, None)
        if mp is None:
            return
        kf_ids = list(mp.observations.keys())
        for kf_id, kp_idx in mp.observations.items():
            kf = self.keyframes.get(kf_id)
            if kf is not None and kf.map_point_ids[kp_idx] == point_id:
                kf.map_point_ids[kp_idx] = -1
        for i, kf_a in enumerate(kf_ids):
            for kf_b in kf_ids[i + 1:]:
                self._decrement_covisibility(kf_a, kf_b)
                self._decrement_covisibility(kf_b, kf_a)

    def _decrement_covisibility(self, kf_a: int, kf_b: int) -> None:
        if kf_b not in self.covisibility.get(kf_a, {}):
            return
        self.covisibility[kf_a][kf_b] -= 1
        if self.covisibility[kf_a][kf_b] <= 0:
            del self.covisibility[kf_a][kf_b]

    def cull_probationary_points(self, current_kf_count: int, probation_kf: int = 5, min_observations: int = 3) -> int:
        """A newly created point earns a permanent place in the map only if it's still
        being observed `min_observations` times once `probation_kf` more keyframes have
        passed - otherwise it's almost certainly noisy stereo depth or a bad match, and
        gets discarded before it can pollute BA / PnP with a phantom 3D point."""
        resolved = [
            pid for pid, created in self.point_probation.items()
            if current_kf_count - created >= probation_kf
        ]
        culled = 0
        for pid in resolved:
            mp = self.map_points.get(pid)
            if mp is None or mp.num_observations() < min_observations:
                if mp is not None:
                    self.remove_map_point(pid)
                culled += 1
            else:
                del self.point_probation[pid]
        return culled

    def add_observation(self, point_id: int, keyframe_id: int, kp_idx: int) -> None:
        mp = self.map_points[point_id]
        existing_kf_ids = [kf_id for kf_id in mp.observations if kf_id != keyframe_id]
        mp.add_observation(keyframe_id, kp_idx)
        self.keyframes[keyframe_id].map_point_ids[kp_idx] = point_id
        self._increment_covisibility_for_new_observer(keyframe_id, existing_kf_ids)

    def _increment_covisibility_for_new_observer(self, new_kf_id: int, existing_kf_ids: list[int]) -> None:
        """Bumps the shared-point count between `new_kf_id` and each keyframe that
        already observed this point, once - not all pairs among the point's full
        observer set, which would re-count already-counted pairs every time a further
        observation is added to a long-lived point."""
        for other_kf in existing_kf_ids:
            self.covisibility[new_kf_id][other_kf] = self.covisibility[new_kf_id].get(other_kf, 0) + 1
            self.covisibility[other_kf][new_kf_id] = self.covisibility[other_kf].get(new_kf_id, 0) + 1

    def remove_keyframe(self, kf_id: int) -> None:
        """Deletes a keyframe and drops its observation from every map point it saw (a
        point left with no observers afterward is deleted too). Used for redundant-
        keyframe culling: once almost everything a keyframe sees is already covered by
        several other keyframes, it adds nothing further and only grows the covisibility
        graph / BA problem size for no benefit."""
        kf = self.keyframes.pop(kf_id, None)
        if kf is None:
            return
        for mp_id in kf.map_point_ids:
            if mp_id < 0:
                continue
            mp = self.map_points.get(int(mp_id))
            if mp is None:
                continue
            mp.observations.pop(kf_id, None)
            if not mp.observations:
                self.map_points.pop(int(mp_id), None)
                self.point_probation.pop(int(mp_id), None)
        for other_id in list(self.covisibility.get(kf_id, {}).keys()):
            self.covisibility[other_id].pop(kf_id, None)
        self.covisibility.pop(kf_id, None)

    def find_redundant_keyframes(
        self, candidate_kf_ids: list[int], min_observers: int = 3, redundancy_ratio: float = 0.9
    ) -> list[int]:
        """Among `candidate_kf_ids`, returns those where at least `redundancy_ratio` of
        the map points they observe are each *also* seen by >= `min_observers` other
        keyframes - i.e. keyframes that add essentially no unique coverage beyond their
        covisible neighbors (ORB-SLAM's local-keyframe-culling criterion)."""
        redundant = []
        for kf_id in candidate_kf_ids:
            kf = self.keyframes.get(kf_id)
            if kf is None:
                continue
            mp_ids = [int(mp_id) for mp_id in kf.map_point_ids if mp_id >= 0]
            if not mp_ids:
                continue
            n_redundant = 0
            for mp_id in mp_ids:
                mp = self.map_points.get(mp_id)
                if mp is not None and len(mp.observations) - 1 >= min_observers:
                    n_redundant += 1
            if n_redundant / len(mp_ids) >= redundancy_ratio:
                redundant.append(kf_id)
        return redundant

    def keyframe_ids_sorted(self) -> list[int]:
        return sorted(self.keyframes.keys())

    def covisible_window(self, keyframe_id: int, window_size: int = 10, min_shared: int = 15) -> list[int]:
        """`keyframe_id` plus its strongest covisibility neighbors (most shared map
        points first), regardless of how long ago they were inserted - so once a loop
        has reconnected a previously visited area, its (not-recent) keyframes still show
        up here and anchor the local optimization instead of being invisible to it."""
        if keyframe_id not in self.keyframes:
            return []
        neighbors = self.covisibility.get(keyframe_id, {})
        ranked = sorted(neighbors.items(), key=lambda kv: -kv[1])
        window = [keyframe_id]
        for kf_id, shared in ranked:
            if shared < min_shared or len(window) >= window_size:
                break
            window.append(kf_id)
        return window
