from pathlib import Path

from splg_slam.data import euroc, kitti, rosbag2


def dataset_module(cfg):
    """Dispatch on cfg.dataset.kind ('euroc' default, 'kitti', or 'rosbag2') to the
    matching loader module - all expose the same load_stereo_rig(dir)/
    load_stereo_frames(dir)/load_mono_frames(dir) interface, so every downstream script
    (tracker, dense_map, octomap) stays dataset-agnostic. rosbag2 additionally returns
    generators (not lists) from load_stereo_frames/load_mono_frames - see
    splg_slam.data.rosbag2's module docstring; build_octomap/dense_map/localize* aren't
    wired up for it and still expect a file-based (euroc/kitti) dataset."""
    kind = getattr(cfg.dataset, "kind", "euroc")
    if kind == "kitti":
        return kitti
    if kind == "rosbag2":
        return rosbag2
    if kind == "euroc":
        return euroc
    raise ValueError(f"Unknown dataset.kind: {kind!r}")


def dataset_dir(cfg) -> Path:
    kind = getattr(cfg.dataset, "kind", "euroc")
    if kind == "kitti":
        return Path(cfg.dataset.sequence_dir)
    if kind == "rosbag2":
        return Path(cfg.dataset.bag_dir)
    return Path(cfg.dataset.mav0_dir)


def right_image_path(left_path: Path, cfg) -> Path:
    """Keyframes only persist the left image path; reconstruct the right path from the
    dataset's known stereo-directory naming convention."""
    kind = getattr(cfg.dataset, "kind", "euroc")
    if kind == "rosbag2":
        # rosbag2 keyframes/frames carry in-memory image arrays, not file paths (see
        # this module's docstring) - a "/cam0/" -> "/cam1/" string replace on left_path
        # is meaningless here and would silently return a bogus path instead of failing
        # loudly, confusing whichever file-based tool eventually tries to open it.
        raise NotImplementedError("right_image_path is not supported for dataset.kind='rosbag2' (no file paths)")
    s = str(left_path)
    if kind == "kitti":
        return Path(s.replace("/image_0/", "/image_1/"))
    return Path(s.replace("/cam0/", "/cam1/"))
