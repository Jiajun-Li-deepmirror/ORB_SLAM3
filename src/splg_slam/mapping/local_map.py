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

    # Bucket current-frame keypoints into a radius_px-sized grid so each projected map
    # point only gets compared against spatially nearby keypoints, instead of building and
    # fully sorting a dense M-by-N distance+similarity matrix against every keypoint in the
    # frame regardless of how sparse genuine radius_px matches actually are - this local
    # map window can hold thousands of points, and this runs every tracked frame. A single
    # ring of 3x3 neighbor cells is always sufficient to find every keypoint within
    # radius_px of a point, since the cell size equals radius_px.
    cell = max(radius_px, 1e-6)
    buckets: dict[tuple[int, int], list[int]] = {}
    for local_j, (cx, cy) in enumerate(np.floor(kpts_c / cell).astype(np.int64)):
        buckets.setdefault((int(cx), int(cy)), []).append(local_j)

    pair_i, pair_j = [], []
    for local_i, (px, py) in enumerate(np.floor(uv_m / cell).astype(np.int64)):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for local_j in buckets.get((int(px) + dx, int(py) + dy), ()):
                    pair_i.append(local_i)
                    pair_j.append(local_j)

    if not pair_i:
        return [], [], [], []

    pair_i = np.array(pair_i)
    pair_j = np.array(pair_j)
    dist = np.linalg.norm(uv_m[pair_i] - kpts_c[pair_j], axis=1)
    sim = np.sum(desc_m[pair_i] * desc_c[pair_j], axis=1)
    sim = np.where(dist < radius_px, sim, -1.0)

    order = np.argsort(-sim)
    used_m, used_n = set(), set()
    obj_pts, img_pts, matched_point_ids, matched_kp_idx = [], [], [], []
    for flat_i in order:
        if sim[flat_i] < desc_threshold:
            break
        i, j = int(pair_i[flat_i]), int(pair_j[flat_i])
        if i in used_m or j in used_n:
            continue
        used_m.add(i)
        used_n.add(j)
        obj_pts.append(positions[m_idx[i]])
        img_pts.append(kpts_c[j])
        matched_point_ids.append(point_ids[m_idx[i]])
        matched_kp_idx.append(int(cand_idx[j]))

    return obj_pts, img_pts, matched_point_ids, matched_kp_idx
