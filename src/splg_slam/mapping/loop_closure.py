import cv2
import numpy as np

from splg_slam.features.splg import SPLG
from splg_slam.geometry.pnp import solve_pnp_ransac
from splg_slam.geometry.stereo import StereoRectifier
from splg_slam.localization.retrieval import GlobalDescriptorIndex
from splg_slam.map.world_map import WorldMap


class LoopConsistencyTracker:
    """Requires a loop-candidate area to be independently re-verified across several
    separate detection cycles before it's trusted - mirrors ORB-SLAM's requirement that
    a candidate persist across consecutive keyframes rather than being accepted off a
    single hit. Guards against perceptual aliasing (two different-but-similar-looking
    places) producing one convincing-but-wrong geometric verification."""

    def __init__(self, required_confirmations: int = 2, group_radius_kf: int = 20, pending_timeout_kf: int = 40):
        self.required_confirmations = required_confirmations
        self.group_radius_kf = group_radius_kf
        self.pending_timeout_kf = pending_timeout_kf
        self._pending: list[dict] = []

    def observe(self, candidate_kf_id: int, current_kf_id: int, rel: np.ndarray) -> bool:
        """rel: the relative pose (candidate_from_current) just verified for this hit.
        Returns True once the same area has been confirmed `required_confirmations` times.

        (Tried grouping hits by covisibility instead of plain id-distance, so a genuinely
        repeated hit far apart in id would still count - it backfired badly: MH01 RMSE
        went 0.044m -> 1.19m, because ordinary nearby-in-the-map keyframes sharing *any*
        point let unrelated candidates get grouped together, letting a still-unverified
        one "confirm" off borrowed count instead of a real independent re-observation.
        Reverted to plain id-distance, which is stricter and safer by construction.)"""
        self._pending = [
            e for e in self._pending if current_kf_id - e["last_current_kf"] <= self.pending_timeout_kf
        ]

        for entry in self._pending:
            if abs(entry["candidate_kf"] - candidate_kf_id) <= self.group_radius_kf:
                entry["count"] += 1
                entry["last_current_kf"] = current_kf_id
                entry["candidate_kf"] = candidate_kf_id
                entry["rel"] = rel
                if entry["count"] >= self.required_confirmations:
                    self._pending.remove(entry)
                    return True
                return False

        self._pending.append({
            "candidate_kf": candidate_kf_id, "last_current_kf": current_kf_id, "count": 1, "rel": rel,
        })
        return False


def detect_loop_candidates(
    world_map: WorldMap,
    index: GlobalDescriptorIndex,
    current_kf_id: int,
    min_id_gap: int = 30,
    top_k: int = 3,
    min_similarity: float = 0.5,
) -> list[tuple[int, float]]:
    """Retrieval-only candidate recall: excludes keyframes within `min_id_gap` of the
    current one so ordinary local covisibility isn't mistaken for a loop.

    (Tried rejecting the whole cycle when top1 didn't lead top2 by a margin, on the
    theory that a near-tie means repetitive-structure ambiguity - backfired hard on all
    three datasets: confirmed loops collapsed from 40-49 down to 8 everywhere, RMSE
    40-50% worse across the board. These scenes get traversed many times on purpose, so
    several keyframes from different earlier passes over the *same* real place routinely
    score near-identically - that's an expected multi-match, not aliasing, and the margin
    check couldn't tell the two apart. Reverted.)"""
    query_desc = world_map.keyframes[current_kf_id].global_descriptor
    if query_desc is None:
        return []
    exclude = {kf_id for kf_id in world_map.keyframes if abs(kf_id - current_kf_id) < min_id_gap}
    return [
        (kf_id, sim)
        for kf_id, sim in index.query(query_desc, top_k=top_k, exclude=exclude)
        if sim >= min_similarity
    ]


def _features_for_keyframe(world_map: WorldMap, rectifier: StereoRectifier, splg: SPLG, kf_id: int, cache: dict):
    if kf_id not in cache:
        kf = world_map.keyframes[kf_id]
        img = cv2.imread(kf.image_path, cv2.IMREAD_GRAYSCALE)
        rect_l, _ = rectifier.rectify(img, img)
        cache[kf_id] = splg.extract(rect_l)
    return cache[kf_id]


def verify_loop_candidate(
    world_map: WorldMap,
    rectifier: StereoRectifier,
    splg: SPLG,
    current_kf_id: int,
    candidate_kf_id: int,
    min_inliers: int = 60,
    min_inlier_ratio: float = 0.0,
    pnp_reproj_threshold_px: float = 3.0,
    feats_cache: dict | None = None,
) -> tuple[bool, np.ndarray | None, int, np.ndarray]:
    """LightGlue-matches the candidate's stored features against the current keyframe's,
    gathers 2D-3D correspondences via the candidate's map points, and solves PnP+RANSAC.
    Returns (accepted, pose_cw_of_current_via_candidates_map, num_inliers, matches), where
    `matches` are the raw LightGlue (candidate_kp_idx, current_kp_idx) pairs - callers can
    pass these to fuse_loop_matches() once the loop is confirmed, to merge whatever map
    points the two sides had independently created for the same physical points.

    `min_inliers` alone lets a high *count* through even when it's a small slice of a much
    larger, noisier correspondence set (e.g. 600 inliers out of 6000 attempted - only
    10%). `min_inlier_ratio` (inliers / attempted 2D-3D correspondences) catches that: a
    candidate must be both absolutely and proportionally well-supported."""
    feats_cache = feats_cache if feats_cache is not None else {}
    candidate_feats = _features_for_keyframe(world_map, rectifier, splg, candidate_kf_id, feats_cache)
    current_feats = _features_for_keyframe(world_map, rectifier, splg, current_kf_id, feats_cache)

    matches = splg.match(candidate_feats, current_feats)["matches"]
    kpts_current, _ = SPLG.to_frame_arrays(current_feats)

    candidate_kf = world_map.keyframes[candidate_kf_id]
    obj_pts, img_pts = [], []
    for cand_i, cur_i in matches:
        mp_id = candidate_kf.map_point_ids[cand_i]
        if mp_id >= 0:
            obj_pts.append(world_map.map_points[mp_id].position)
            img_pts.append(kpts_current[cur_i])

    if len(obj_pts) < min_inliers:
        return False, None, len(obj_pts), matches

    ok, pose_cw_loop, inlier_mask = solve_pnp_ransac(
        np.asarray(obj_pts), np.asarray(img_pts), rectifier.K_rect,
        reproj_threshold_px=pnp_reproj_threshold_px,
    )
    if not ok:
        return False, None, 0, matches

    num_inliers = int(inlier_mask.sum())
    inlier_ratio = num_inliers / len(obj_pts)
    accepted = num_inliers >= min_inliers and inlier_ratio >= min_inlier_ratio
    return accepted, pose_cw_loop, num_inliers, matches


def fuse_loop_matches(world_map: WorldMap, candidate_kf_id: int, current_kf_id: int, matches: np.ndarray) -> int:
    """For every matched keypoint pair where BOTH sides already have their own map point,
    the two keyframes independently triangulated the same physical point on their own
    pass through the scene - merge them into one (keeping the candidate's, since it comes
    from the earlier, already-settled part of the map) so later BA isn't fighting two
    slightly-different 3D points for what should be a single, better-constrained one."""
    candidate_kf = world_map.keyframes[candidate_kf_id]
    current_kf = world_map.keyframes[current_kf_id]
    n_fused = 0
    for cand_i, cur_i in matches:
        keep_id = int(candidate_kf.map_point_ids[cand_i])
        remove_id = int(current_kf.map_point_ids[cur_i])
        if keep_id < 0 or remove_id < 0 or keep_id == remove_id:
            continue
        if keep_id not in world_map.map_points or remove_id not in world_map.map_points:
            continue
        world_map.merge_map_points(keep_id, remove_id)
        n_fused += 1
    return n_fused
