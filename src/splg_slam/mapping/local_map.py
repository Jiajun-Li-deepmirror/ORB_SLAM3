import numpy as np

from splg_slam.map.world_map import WorldMap


def project_points(pose_cw: np.ndarray, points_world: np.ndarray, k: np.ndarray):
    """points_world: Nx3. Returns (uv Nx2 [NaN where invalid], valid_mask N bool)."""
    p_cam = (pose_cw[:3, :3] @ points_world.T).T + pose_cw[:3, 3]
    valid = p_cam[:, 2] > 1e-3

    uv = np.full((len(points_world), 2), np.nan)
    uv_h = (k @ p_cam[valid].T).T
    uv[valid] = uv_h[:, :2] / uv_h[:, 2:3]
    return uv, valid


def gather_local_map_point_ids(world_map: WorldMap, keyframe_ids: list[int], exclude: set[int] | None = None) -> list[int]:
    exclude = exclude or set()
    point_ids = set()
    for kf_id in keyframe_ids:
        for mp_id in world_map.keyframes[kf_id].map_point_ids:
            if mp_id >= 0 and int(mp_id) not in exclude:
                point_ids.add(int(mp_id))
    return list(point_ids)


def search_local_map(
    world_map: WorldMap,
    point_ids: list[int],
    pose_cw: np.ndarray,
    kpts: np.ndarray,
    descriptors: np.ndarray,
    k_rect: np.ndarray,
    image_size: tuple[int, int],
    used_kp_mask: np.ndarray,
    radius_px: float = 6.0,
    desc_threshold: float = 0.8,
):
    """Projects `point_ids` into the current frame with `pose_cw` and greedily matches
    them (best cosine similarity first) against not-yet-used current-frame keypoints
    within `radius_px`. Returns (obj_pts, img_pts, matched_point_ids, matched_kp_idx),
    all as plain lists, disjoint from whatever produced `used_kp_mask`."""
    if not point_ids:
        return [], [], [], []

    positions = np.array([world_map.map_points[pid].position for pid in point_ids])
    mp_descs = np.array([world_map.map_points[pid].descriptor for pid in point_ids])

    uv, valid = project_points(pose_cw, positions, k_rect)
    w, h = image_size
    in_bounds = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)

    cand_idx = np.nonzero(~used_kp_mask)[0]
    m_idx = np.nonzero(in_bounds)[0]
    if len(cand_idx) == 0 or len(m_idx) == 0:
        return [], [], [], []

    uv_m = uv[m_idx]
    desc_m = mp_descs[m_idx]
    kpts_c = kpts[cand_idx]
    desc_c = descriptors[cand_idx]

    dist = np.linalg.norm(uv_m[:, None, :] - kpts_c[None, :, :], axis=2)
    sim = desc_m @ desc_c.T
    sim = np.where(dist < radius_px, sim, -1.0)

    order = np.argsort(-sim, axis=None)
    used_m, used_n = set(), set()
    obj_pts, img_pts, matched_point_ids, matched_kp_idx = [], [], [], []
    n_cols = sim.shape[1]
    for flat_i in order:
        i, j = divmod(int(flat_i), n_cols)
        if sim[i, j] < desc_threshold:
            break
        if i in used_m or j in used_n:
            continue
        used_m.add(i)
        used_n.add(j)
        obj_pts.append(positions[m_idx[i]])
        img_pts.append(kpts_c[j])
        matched_point_ids.append(point_ids[m_idx[i]])
        matched_kp_idx.append(int(cand_idx[j]))

    return obj_pts, img_pts, matched_point_ids, matched_kp_idx
