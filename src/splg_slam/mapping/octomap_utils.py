import numpy as np
import open3d as o3d

from splg_slam.geometry.pose_utils import camera_center, invert_pose
from splg_slam.geometry.stereo import StereoDepthEstimator, StereoRectifier


def insert_keyframe_into_octree(
    tree, img_left_gray: np.ndarray, img_right_gray: np.ndarray, pose_cw: np.ndarray,
    rectifier: StereoRectifier, depth_est: StereoDepthEstimator, pixel_xy: np.ndarray | None,
    stride: int, max_depth_m: float, outlier_nb_neighbors: int = 20, outlier_std_ratio: float = 1.5,
    r_align: np.ndarray | None = None, min_depth_m: float = 0.0,
) -> np.ndarray:
    """Ray-casts one keyframe's dense stereo depth into a persistent Octomap OcTree, marking
    traversed voxels FREE and endpoints OCCUPIED - insertPointCloud() distinguishes "seen
    through, empty" from "never observed", which a plain point cloud can't. This is the same
    per-keyframe insertion build_octomap.py does in one offline pass over an already-finished
    map; factored out here so an online loop (see scripts/online_plan_loop.py) can call it
    incrementally, right after each keyframe lands in the SLAM map, instead of waiting for the
    whole sequence to finish before the octomap exists at all.

    Returns `pixel_xy` (the strided sample grid) so the caller can reuse it across keyframes
    without recomputing it from the image shape every time - same array in, same array out,
    computed once on the first call (`pixel_xy=None`)."""
    rect_l, rect_r = rectifier.rectify(img_left_gray, img_right_gray)
    disp = depth_est.compute_disparity(rect_l, rect_r)

    if pixel_xy is None:
        h, w = disp.shape
        ys, xs = np.mgrid[0:h:stride, 0:w:stride]
        pixel_xy = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)

    depths = depth_est.depths_at_points(disp, pixel_xy)
    # Below min_depth_m, disparity is saturating at the matcher's own max-disparity search
    # limit rather than measuring anything real - it reports a hard-clamped near-camera depth
    # for pixels it simply failed to match, not a genuine close surface. That spurious "point"
    # then gets ray-cast in as an OCCUPIED voxel sitting right on top of the camera's own
    # trajectory. This is a per-point depth-computation error, unrelated to how accurate the
    # camera pose (trajectory/ATE) itself is.
    valid = np.isfinite(depths) & (depths > min_depth_m) & (depths < max_depth_m)
    if not valid.any():
        return pixel_xy

    pts_cam = rectifier.backproject(pixel_xy[valid], depths[valid])
    pose_wc = invert_pose(pose_cw)
    pts_world = (pose_wc[:3, :3] @ pts_cam.T).T + pose_wc[:3, 3]
    origin = camera_center(pose_cw)

    if r_align is not None:
        pts_world = (r_align @ pts_world.T).T
        origin = r_align @ origin

    if outlier_nb_neighbors > 0 and len(pts_world) > outlier_nb_neighbors:
        frame_cloud = o3d.geometry.PointCloud()
        frame_cloud.points = o3d.utility.Vector3dVector(pts_world)
        frame_cloud, _ = frame_cloud.remove_statistical_outlier(
            nb_neighbors=outlier_nb_neighbors, std_ratio=outlier_std_ratio,
        )
        pts_world = np.asarray(frame_cloud.points)

    if len(pts_world) > 0:
        tree.insertPointCloud(pts_world.astype(np.float64), origin.astype(np.float64), maxrange=max_depth_m, lazy_eval=True)

    return pixel_xy
