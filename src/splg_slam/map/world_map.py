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
        self.point_probation: dict[int, int] = {}  # point_id -> keyframe count at creation
        self.frame_processing_times_s: list[float] = []  # per-input-frame wall-clock time (seconds)
        self._next_point_id = 0

    def add_loop_edge(self, kf_id_a: int, kf_id_b: int, relative_pose_a_from_b: np.ndarray, num_inliers: int) -> None:
        self.loop_edges.append((kf_id_a, kf_id_b, relative_pose_a_from_b, num_inliers))

    def add_imu_factor(self, kf_id_a: int, kf_id_b: int, samples: np.ndarray) -> None:
        self.imu_factors.append((kf_id_a, kf_id_b, samples))

    def add_keyframe(self, kf: KeyFrame) -> None:
        self.keyframes[kf.frame_id] = kf

    def new_map_point(self, position: np.ndarray, descriptor: np.ndarray, created_at_kf_count: int | None = None) -> int:
        point_id = self._next_point_id
        self.map_points[point_id] = MapPoint(point_id=point_id, position=position, descriptor=descriptor)
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
        mp.add_observation(keyframe_id, kp_idx)
        self.keyframes[keyframe_id].map_point_ids[kp_idx] = point_id
        self._update_covisibility_for(point_id)

    def _update_covisibility_for(self, point_id: int) -> None:
        kf_ids = list(self.map_points[point_id].observations.keys())
        for i, kf_a in enumerate(kf_ids):
            for kf_b in kf_ids[i + 1:]:
                self.covisibility[kf_a][kf_b] = self.covisibility[kf_a].get(kf_b, 0) + 1
                self.covisibility[kf_b][kf_a] = self.covisibility[kf_b].get(kf_a, 0) + 1

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
