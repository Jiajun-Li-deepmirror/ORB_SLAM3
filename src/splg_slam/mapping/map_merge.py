import cv2
import numpy as np

from splg_slam.geometry.pose_utils import invert_pose
from splg_slam.localization.relocalizer import Relocalizer
from splg_slam.map.world_map import WorldMap


def _transform_diff(t1: np.ndarray, t2: np.ndarray) -> tuple[float, float]:
    d_trans = float(np.linalg.norm(t1[:3, 3] - t2[:3, 3]))
    r_rel = t1[:3, :3].T @ t2[:3, :3]
    cos_angle = np.clip((np.trace(r_rel) - 1) / 2, -1.0, 1.0)
    d_rot = float(np.degrees(np.arccos(cos_angle)))
    return d_trans, d_rot


def _average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    """Averages a small, mutually-agreeing cluster of rigid transforms: mean translation,
    and mean rotation via SVD-orthogonalization of the mean rotation matrix (fine for a
    small-spread cluster - not a proper manifold mean, but the cluster is already filtered
    to be tightly agreeing before this is called)."""
    mean_trans = np.mean([t[:3, 3] for t in transforms], axis=0)
    mean_rot_raw = np.mean([t[:3, :3] for t in transforms], axis=0)
    u, _, vt = np.linalg.svd(mean_rot_raw)
    mean_rot = u @ vt
    if np.linalg.det(mean_rot) < 0:
        u[:, -1] *= -1
        mean_rot = u @ vt
    t_final = np.eye(4)
    t_final[:3, :3] = mean_rot
    t_final[:3, 3] = mean_trans
    return t_final


def register_map_b_into_a(
    relocalizer: Relocalizer, map_b: WorldMap, rectifier, sample_stride: int = 5,
    agreement_trans_m: float = 0.5, agreement_rot_deg: float = 5.0, min_cluster_size: int = 3,
) -> dict:
    """Finds the rigid transform T_worldA_worldB (4x4) bringing map_b's coordinate frame into
    map_a's (map_a is whatever `relocalizer` was built against), by cross-localizing a sample
    of map_b's own keyframe images against map_a - reusing Relocalizer exactly as it's used
    for live query localization. Each successful cross-localization gives the SAME physical
    camera pose expressed in both frames (pose_cw_a from the relocalizer, pose_cw_b already
    stored on the map_b keyframe), which is enough to solve for the one shared rigid
    transform between the two maps:

        pose_cw_a @ p_worldA = pose_cw_b @ p_worldB  (same instant, same camera)
        => T_worldA_worldB = pose_wc_a @ pose_cw_b

    Robust to individual bad cross-map matches: collects every candidate transform, keeps
    only the largest mutually-agreeing cluster (by translation/rotation tolerance), and
    averages those - a single spurious match (repetitive-structure aliasing) won't be in the
    dominant cluster and gets discarded rather than corrupting the whole registration.

    Returns a dict with "accepted": bool; when True, also "T_worldA_worldB", "num_candidates",
    "cluster_size".
    """
    candidates = []
    for kf_id in map_b.keyframe_ids_sorted()[::sample_stride]:
        kf_b = map_b.keyframes[kf_id]
        if kf_b.image_path is None:
            continue
        img = cv2.imread(kf_b.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        rect_l, _ = rectifier.rectify(img, img)
        ok, pose_cw_a, info = relocalizer.localize(rect_l)
        if not ok:
            continue
        t_wa_wb = invert_pose(pose_cw_a) @ kf_b.pose_cw
        candidates.append((t_wa_wb, info["num_inliers"], kf_id))

    if len(candidates) < min_cluster_size:
        return {"accepted": False, "reason": f"only {len(candidates)} cross-map matches found", "num_candidates": len(candidates)}

    best_cluster: list[int] = []
    for i, (t_i, _, _) in enumerate(candidates):
        cluster = [i]
        for j, (t_j, _, _) in enumerate(candidates):
            if i == j:
                continue
            d_trans, d_rot = _transform_diff(t_i, t_j)
            if d_trans < agreement_trans_m and d_rot < agreement_rot_deg:
                cluster.append(j)
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    if len(best_cluster) < min_cluster_size:
        return {
            "accepted": False,
            "reason": f"no agreeing cluster (best size {len(best_cluster)}/{len(candidates)})",
            "num_candidates": len(candidates),
        }

    t_final = _average_transforms([candidates[i][0] for i in best_cluster])
    return {
        "accepted": True,
        "T_worldA_worldB": t_final,
        "num_candidates": len(candidates),
        "cluster_size": len(best_cluster),
    }
