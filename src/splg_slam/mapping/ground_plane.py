"""Optional observed ground landmarks shared by BA and pose-graph optimization.

Camera-frame normals/centroids remain immutable across map realignments. Plane
variables are rebuilt in the current world gauge at each solve. Different ground
patches/floors have independent heights; camera height itself is never fixed.
"""
from dataclasses import dataclass, fields, replace

import gtsam
import numpy as np
from gtsam.symbol_shorthand import X
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.spatial.transform import Rotation

from splg_slam.mapping.gtsam_utils import pose_cw_to_gtsam


@dataclass(frozen=True)
class GroundPlaneOptions:
    enabled: bool = False
    depth_topic: str = "/camera/camera/depth/image_rect_raw"
    depth_info_topic: str = "/camera/camera/depth/camera_info"
    depth_scale: float = 0.001
    max_depth_dt_ms: float = 20.0
    pixel_stride: int = 6
    roi_top: float = 0.45
    min_depth_m: float = 0.4
    max_depth_m: float = 4.0
    min_camera_height_m: float = 0.5
    max_camera_height_m: float = 2.5
    max_tilt_deg: float = 20.0
    distance_threshold_m: float = 0.035
    min_inliers: int = 100
    min_inlier_ratio: float = 0.12
    min_area_m2: float = 0.6
    min_span_m: float = 0.45
    ransac_iterations: int = 200
    association_distance_m: float = 0.12
    association_angle_deg: float = 8.0
    overlap_distance_m: float = 0.25
    min_overlap_ratio: float = 0.15
    distance_sigma_m: float = 0.06
    normal_sigma_deg: float = 3.0
    gravity_sigma_deg: float = 5.0
    min_observations: int = 3
    # Sensor bias of the depth-measured floor normal as a camera-frame rotation [about x
    # (pitch), about z (roll)] in degrees, removed from every observation. Calibrate it on a
    # level floor once gravity is right: median of cross(map +Z in camera, floor normal).
    normal_bias_deg: tuple = (0.0, 0.0)

    @classmethod
    def from_config(cls, config):
        values = {} if config is None else (config if isinstance(config, dict) else vars(config))
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown ground_plane options: {sorted(unknown)}")
        result = cls(**values)
        for f in fields(cls):
            value = getattr(result, f.name)
            if f.type in (float, int) and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"ground_plane.{f.name} must be finite and positive")
            if f.type is int and (isinstance(value, bool) or int(value) != value):
                raise ValueError(f"ground_plane.{f.name} must be an integer")
        for name in ("roi_top", "min_inlier_ratio", "min_overlap_ratio"):
            if getattr(result, name) >= 1:
                raise ValueError(f"ground_plane.{name} must be below 1")
        if result.min_depth_m >= result.max_depth_m or result.min_camera_height_m >= result.max_camera_height_m:
            raise ValueError("Ground depth/height bounds must be ordered")
        if max(result.max_tilt_deg, result.association_angle_deg, result.gravity_sigma_deg) >= 45:
            raise ValueError("Ground angle limits must be below 45 degrees")
        if result.min_observations < 2:
            raise ValueError("Ground constraints need at least two observations")
        bias = np.asarray(result.normal_bias_deg, dtype=float)
        if bias.shape != (2,) or not np.isfinite(bias).all() or np.abs(bias).max() >= 5:
            raise ValueError("ground_plane.normal_bias_deg must be two finite angles below 5 degrees")
        return replace(result, normal_bias_deg=(float(bias[0]), float(bias[1])))


def detect_ground(points, up_camera, options, seed=0):
    """Constrained deterministic RANSAC; select the lowest sufficiently wide plane.

    This is geometric ground detection, not semantic segmentation. Small stair
    treads, walls and ceiling planes are rejected by area, direction and sidedness.
    A sufficiently large tabletop can remain ambiguous; diagnostics expose this.
    """
    p = np.asarray(points, dtype=float)
    p = p[np.isfinite(p).all(axis=1)]
    up = np.asarray(up_camera, dtype=float)
    up /= np.linalg.norm(up)
    if len(p) < options.min_inliers:
        return None, {"reason": "too_few_depth_points", "points": len(p)}
    rng = np.random.default_rng(seed)
    working = p
    candidates = []
    cosine = np.cos(np.radians(options.max_tilt_deg))
    for _ in range(3):
        if len(working) < options.min_inliers:
            break
        best = None
        for _ in range(options.ransac_iterations):
            sample = working[rng.choice(len(working), 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            length = np.linalg.norm(normal)
            if length < 1e-8:
                continue
            normal /= length
            if normal @ up < 0:
                normal = -normal
            if normal @ up < cosine:
                continue
            height = -normal @ sample[0]
            if not options.min_camera_height_m <= height <= options.max_camera_height_m:
                continue
            mask = np.abs(working @ normal + height) < options.distance_threshold_m
            if best is None or mask.sum() > best.sum():
                best = mask
        if best is None or best.sum() < options.min_inliers:
            break
        inliers = working[best]
        centroid = np.mean(inliers, axis=0)
        _, _, axes = np.linalg.svd(inliers - centroid, full_matrices=False)
        normal = axes[-1]
        if normal @ up < 0:
            normal = -normal
        height = -normal @ centroid
        mask = np.abs(p @ normal + height) < options.distance_threshold_m
        inliers = p[mask]
        xy = (inliers - centroid) @ axes[:2].T
        try:
            area = float(ConvexHull(xy).volume)
        except QhullError:
            area = 0.0
        span = np.percentile(xy, 95, axis=0) - np.percentile(xy, 5, axis=0)
        rms = float(np.sqrt(np.mean((inliers @ normal + height) ** 2)))
        if (normal @ up >= cosine and options.min_camera_height_m <= height <= options.max_camera_height_m
                and mask.sum() >= options.min_inliers and mask.mean() >= options.min_inlier_ratio
                and area >= options.min_area_m2 and min(span) >= options.min_span_m):
            # A small spatial sample suffices for overlap checks and audit plots.
            take = rng.choice(len(inliers), min(96, len(inliers)), replace=False)
            candidates.append(dict(normal=normal.copy(), centroid=np.mean(inliers, axis=0),
                                   support=inliers[take].copy(), height_m=float(height),
                                   inliers=int(mask.sum()), inlier_ratio=float(mask.mean()),
                                   area_m2=area, rms_m=rms))
        working = working[~best]
    if not candidates:
        return None, {"reason": "no_supported_ground", "points": len(p)}
    observation = max(candidates, key=lambda c: c["height_m"])
    if any(options.normal_bias_deg):
        pitch, roll = np.radians(options.normal_bias_deg)
        normal = Rotation.from_rotvec([-pitch, 0.0, -roll]).as_matrix() @ observation["normal"]
        observation["normal"] = normal
        observation["height_m"] = float(-normal @ observation["centroid"])
    return observation, {"reason": "accepted", **{k: observation[k] for k in
                        ("height_m", "inliers", "inlier_ratio", "area_m2", "rms_m")}}


def world_observation(kf):
    obs = kf.ground_plane
    rotation = kf.pose_cw[:3, :3].T
    translation = -rotation @ kf.pose_cw[:3, 3]
    return (rotation @ obs["normal"], rotation @ obs["centroid"] + translation,
            obs["support"] @ rotation.T + translation)


def associate_ground(world, kf, observation, options):
    kf.ground_plane = observation
    normal, center, support = world_observation(kf)
    best = None
    for other in world.keyframes.values():
        if other.frame_id == kf.frame_id or other.segment_id != kf.segment_id or not getattr(other, "ground_plane", None):
            continue
        n, c, s = world_observation(other)
        angle = np.degrees(np.arccos(np.clip(normal @ n, -1, 1)))
        distance = max(abs(n @ (center - c)), abs(normal @ (center - c)))
        if angle > options.association_angle_deg or distance > options.association_distance_m:
            continue
        # Actual 3D footprint overlap, not a height-only floor classification.
        close = cKDTree(s).query(support)[0] < options.overlap_distance_m
        reverse = cKDTree(support).query(s)[0] < options.overlap_distance_m
        overlap = max(float(close.mean()), float(reverse.mean()))
        if overlap < options.min_overlap_ratio:
            continue
        score = distance / options.association_distance_m + angle / options.association_angle_deg - overlap
        if best is None or score < best[0]:
            best = (score, other.ground_plane["group_id"], other.frame_id)
    if best is None:
        group = getattr(world, "_next_ground_group", 0)
        world._next_ground_group = group + 1
        matched = None
    else:
        _, group, matched = best
    observation["group_id"] = group
    observation["matched_keyframe"] = matched
    return {"group_id": group, "matched_keyframe": matched}


def _skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def plane_error(pose, plane, normal_c, centroid_c, jacobians=False):
    """Plane z=a*x+b*y+h; analytic Jacobians for GTSAM's right Pose3 retract."""
    a, b, height = plane
    v = np.array([-a, -b, 1.])
    norm = np.linalg.norm(v)
    n = v / norm
    rotation = pose.rotation().matrix()
    point = rotation @ centroid_c + pose.translation()
    residual = np.r_[rotation @ normal_c - n, n @ point - height * n[2]]
    if not jacobians:
        return residual
    j_pose = np.zeros((4, 6))
    j_pose[:3, :3] = -rotation @ _skew(normal_c)
    j_pose[3, :3] = -n @ rotation @ _skew(centroid_c)
    j_pose[3, 3:] = n @ rotation
    dn = -(np.eye(3) - np.outer(n, n))[:, :2] / norm
    j_plane = np.zeros((4, 3))
    j_plane[:3, :2] = -dn
    j_plane[3, :2] = (point - np.array([0., 0., height])) @ dn
    j_plane[3, 2] = -n[2]
    return residual, j_pose, j_plane


def make_plane_factor(pose_key, plane_key, observation, noise):
    normal = observation["normal"].copy()
    centroid = observation["centroid"].copy()

    def error(_factor, values, jacobians):
        pose, plane = values.atPose3(pose_key), values.atVector(plane_key)
        if jacobians is None:
            return plane_error(pose, plane, normal, centroid)
        residual, jp, jl = plane_error(pose, plane, normal, centroid, True)
        jacobians[0], jacobians[1] = jp, jl
        return residual
    return gtsam.CustomFactor(noise, [pose_key, plane_key], error)


def add_ground_factors(world, graph, initial, free_ids, fixed_ids):
    """Adds shared plane landmarks and, in local BA, up to two boundary observers.

    Returning immediately when disabled preserves the previous graph and avoids
    any depth I/O. No optimized world-plane state is cached, so rejected solves,
    culled keyframes, Atlas merges and IMU gauge rotations cannot leave stale planes.
    """
    options = getattr(world, "ground_plane_options", None)
    if options is None or not options.enabled:
        return {"ground_plane_factors": 0, "ground_plane_groups": 0}
    groups = {}
    for kid, kf in world.keyframes.items():
        obs = getattr(kf, "ground_plane", None)
        if obs is not None:
            groups.setdefault((kf.segment_id, obs["group_id"]), []).append(kid)
    sigmas = np.array([np.radians(options.normal_sigma_deg)] * 3 + [options.distance_sigma_m])
    noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(1.345),
                                          gtsam.noiseModel.Diagonal.Sigmas(sigmas))
    anchor_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)
    count, used = 0, 0
    for (segment, group), members in sorted(groups.items()):
        active = sorted(set(members) & free_ids)
        if not active or len(members) < options.min_observations:
            continue
        boundary = sorted(set(members) - free_ids)
        selected = sorted(set(active + ([boundary[0], boundary[-1]] if boundary else [])))
        if len(selected) < 2:
            continue
        # Initialize the plane from current poses. Fixed boundary observations take
        # precedence; in global BA every measurement participates in initialization.
        init_members = boundary if boundary else members
        normals, centers = [], []
        for kid in init_members:
            n, c, _ = world_observation(world.keyframes[kid])
            normals.append(n)
            centers.append(c)
        normal = np.mean(normals, axis=0)
        normal /= np.linalg.norm(normal)
        if normal[2] < 0.5:
            continue  # segment is not gravity aligned; do not constrain it
        used += 1
        key = gtsam.symbol('g', used)
        height = float(np.median(np.asarray(centers) @ normal) / normal[2])
        initial.insert(key, np.array([-normal[0] / normal[2], -normal[1] / normal[2], height]))

        def slope_error(factor, values, jacobians):
            if jacobians is not None:
                jacobians[0] = np.array([[1., 0., 0.], [0., 1., 0.]])
            return values.atVector(factor.keys()[0])[:2]
        graph.add(gtsam.CustomFactor(gtsam.noiseModel.Isotropic.Sigma(
            2, np.tan(np.radians(options.gravity_sigma_deg))), [key], slope_error))
        for kid in selected:
            if not initial.exists(X(kid)):
                initial.insert(X(kid), pose_cw_to_gtsam(world.keyframes[kid].pose_cw))
                graph.add(gtsam.PriorFactorPose3(X(kid), initial.atPose3(X(kid)), anchor_noise))
                fixed_ids.add(kid)
            graph.add(make_plane_factor(X(kid), key, world.keyframes[kid].ground_plane, noise))
            count += 1
    return {"ground_plane_factors": count, "ground_plane_groups": used}
