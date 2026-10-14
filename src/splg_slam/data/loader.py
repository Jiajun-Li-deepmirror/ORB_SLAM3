from pathlib import Path

import cv2

from splg_slam.data import euroc, kitti, rosbag2


def dataset_module(cfg):
    """Dispatch on cfg.dataset.kind ('euroc' default, 'kitti', or 'rosbag2') to the
    matching loader module - all expose the same load_stereo_rig(dir)/
    load_stereo_frames(dir)/load_mono_frames(dir) interface, so every downstream script
    (tracker, dense_map, octomap) stays dataset-agnostic. rosbag2 additionally returns
    generators (not lists) from load_stereo_frames/load_mono_frames - see
    splg_slam.data.rosbag2's module docstring. Consumers must support streamed entries
    with in-memory images as well as file-backed entries."""
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


def set_dataset_dir(cfg, path: str) -> None:
    kind = getattr(cfg.dataset, "kind", "euroc")
    field = {"euroc": "mav0_dir", "kitti": "sequence_dir", "rosbag2": "bag_dir"}[kind]
    setattr(cfg.dataset, field, path)


def load_dataset_rig(cfg):
    path = dataset_dir(cfg)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}. Update dataset in the config or use --dataset-dir.")
    tf_source = getattr(cfg.dataset, "tf_static_from", None)
    kwargs = {"tf_static_from": tf_source} if tf_source else {}
    return dataset_module(cfg).load_stereo_rig(path, **kwargs)


def read_frame_images(entry, stereo: bool = True):
    left = entry.left_image if hasattr(entry, "left_image") else cv2.imread(str(entry.left_path), cv2.IMREAD_GRAYSCALE)
    right = None
    if stereo:
        right = entry.right_image if hasattr(entry, "right_image") else cv2.imread(str(entry.right_path), cv2.IMREAD_GRAYSCALE)
    if left is None or (stereo and right is None):
        raise FileNotFoundError(f"Cannot read images for frame at {entry.timestamp_ns}")
    return left, right


def iter_keyframe_stereo_images(world_map, cfg, keyframe_ids, color_left: bool = False):
    """Stream only the requested keyframe images, including maps built from rosbag2."""
    if getattr(cfg.dataset, "kind", "euroc") == "rosbag2":
        needed = {}
        for kf_id in keyframe_ids:
            needed.setdefault(world_map.keyframes[kf_id].timestamp_ns, []).append(kf_id)
        if not needed:
            return
        entries = dataset_module(cfg).load_stereo_frames(dataset_dir(cfg))
        try:
            for entry in entries:
                ids = needed.pop(entry.timestamp_ns, ())
                if not ids:
                    continue
                left, right = read_frame_images(entry)
                if color_left:
                    left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
                for kf_id in ids:
                    yield kf_id, left, right
                if not needed:
                    break
        finally:
            close = getattr(entries, "close", None)
            if close is not None:
                close()
        if needed:
            raise ValueError(f"Bag is missing {sum(map(len, needed.values()))} requested keyframe image pair(s); check map/config pairing.")
        return
    for kf_id in keyframe_ids:
        kf = world_map.keyframes[kf_id]
        if kf.image_path is None:
            raise ValueError(f"Keyframe {kf_id} has no image path; use its rosbag2 dataset config.")
        left = cv2.imread(kf.image_path, cv2.IMREAD_COLOR if color_left else cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(right_image_path(Path(kf.image_path), cfg)), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            raise FileNotFoundError(f"Cannot read keyframe {kf_id}: {kf.image_path}")
        yield kf_id, left, right


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
