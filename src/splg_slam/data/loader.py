from pathlib import Path

from splg_slam.data import euroc, kitti


def dataset_module(cfg):
    """Dispatch on cfg.dataset.kind ('euroc' default, or 'kitti') to the matching loader
    module - both expose the same load_stereo_rig(dir)/load_stereo_frames(dir) interface,
    so every downstream script (tracker, dense_map, octomap) stays dataset-agnostic."""
    kind = getattr(cfg.dataset, "kind", "euroc")
    if kind == "kitti":
        return kitti
    if kind == "euroc":
        return euroc
    raise ValueError(f"Unknown dataset.kind: {kind!r}")


def dataset_dir(cfg) -> Path:
    kind = getattr(cfg.dataset, "kind", "euroc")
    if kind == "kitti":
        return Path(cfg.dataset.sequence_dir)
    return Path(cfg.dataset.mav0_dir)


def right_image_path(left_path: Path, cfg) -> Path:
    """Keyframes only persist the left image path; reconstruct the right path from the
    dataset's known stereo-directory naming convention."""
    kind = getattr(cfg.dataset, "kind", "euroc")
    s = str(left_path)
    if kind == "kitti":
        return Path(s.replace("/image_0/", "/image_1/"))
    return Path(s.replace("/cam0/", "/cam1/"))
