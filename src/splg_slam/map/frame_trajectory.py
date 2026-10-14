"""Per-input tracking records and reference-keyframe trajectory reconstruction.

Final poses follow a reference keyframe's rigid correction. They are not a new
optimization of every frame. Failed frames remain absent; no interpolation occurs.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from splg_slam.geometry.pose_utils import invert_pose


class FrameTrajectory:
    def __init__(self):
        self.records = []
        # Removed keyframe -> (live/removed parent, T_removed_parent).
        self.retired_anchors = {}

    def append(self, world, frame_id, timestamp_ns, frame, is_keyframe, anchor_id,
               status, diagnostics=None, final_supported=True):
        if self.records and timestamp_ns <= self.records[-1]["timestamp_ns"]:
            raise ValueError("Frame trajectory timestamps must increase")
        pose = None if frame is None else np.array(frame.pose_cw, copy=True)
        valid = pose is not None and np.isfinite(pose).all() and status not in ("lost", "bootstrap_wait")
        anchor = world.keyframes.get(anchor_id)
        self.records.append(dict(
            frame_id=int(frame_id), timestamp_ns=int(timestamp_ns), status=status,
            is_keyframe=bool(is_keyframe), original_anchor_id=anchor_id,
            original_segment_id=None if anchor is None else int(anchor.segment_id),
            pose_online_cw=pose if valid else None,
            relative_cw=(pose @ invert_pose(anchor.pose_cw)) if valid and anchor is not None else None,
            final_supported=bool(final_supported), diagnostics=diagnostics or {},
        ))

    def retire_anchor(self, world, keyframe_id):
        kf = world.keyframes[keyframe_id]
        choices = [k for k in world.keyframes.values()
                   if k.frame_id != keyframe_id and k.segment_id == kf.segment_id]
        if not choices:
            # No same-segment parent: explicitly unresolved instead of borrowing a
            # pose from an unrelated coordinate frame.
            self.retired_anchors[keyframe_id] = (None, None)
            return
        parent = min(choices, key=lambda k: (abs(k.timestamp_ns-kf.timestamp_ns), k.frame_id))
        self.retired_anchors[keyframe_id] = (
            parent.frame_id, kf.pose_cw @ invert_pose(parent.pose_cw))

    def resolve(self, world, record):
        if record["relative_cw"] is None or not record["final_supported"]:
            return None, None, None
        anchor_id = record["original_anchor_id"]
        relative = record["relative_cw"].copy()
        visited = set()
        while anchor_id not in world.keyframes:
            if anchor_id is None or anchor_id in visited or anchor_id not in self.retired_anchors:
                return None, None, None
            visited.add(anchor_id)
            anchor_id, transform = self.retired_anchors[anchor_id]
            if transform is None:
                return None, None, None
            relative = relative @ transform
        anchor = world.keyframes[anchor_id]
        return relative @ anchor.pose_cw, anchor_id, anchor.segment_id

    def export(self, world, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        n = len(self.records)
        online = np.full((n, 4, 4), np.nan)
        final = np.full_like(online, np.nan)
        anchor_ids = np.full(n, -1, dtype=np.int64)
        segments = np.full(n, -1, dtype=np.int64)
        rows = []
        for i, record in enumerate(self.records):
            if record["pose_online_cw"] is not None:
                online[i] = record["pose_online_cw"]
            pose, anchor, segment = self.resolve(world, record)
            if pose is not None:
                final[i], anchor_ids[i], segments[i] = pose, anchor, segment
            row = {k: v for k, v in record.items() if k not in ("pose_online_cw", "relative_cw")}
            row.update(online_valid=bool(np.isfinite(online[i]).all()),
                       final_valid=bool(np.isfinite(final[i]).all()),
                       final_anchor_id=anchor, final_segment_id=segment)
            rows.append(row)
        timestamps = np.array([r["timestamp_ns"] for r in rows], dtype=np.int64)
        np.savez_compressed(directory / "trajectory_frames.npz", timestamp_ns=timestamps,
                            frame_id=np.array([r["frame_id"] for r in rows], dtype=np.int64),
                            status=np.array([r["status"] for r in rows], dtype=str),
                            pose_online_cw=online, pose_final_cw=final,
                            final_anchor_id=anchor_ids, final_segment_id=segments)
        with (directory / "tracking_frames.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        for label, poses in (("online", online), ("final", final)):
            with (directory / f"trajectory_frames_{label}.tum").open("w") as f:
                f.write("# timestamp_s tx ty tz qx qy qz qw; T_world_rectified_cam0\n")
                f.write("# Valid estimates only; consult tracking_frames.jsonl for missing frames/segments.\n")
                for stamp, pose in zip(timestamps, poses):
                    if not np.isfinite(pose).all():
                        continue
                    wc = invert_pose(pose)
                    values = [*wc[:3, 3], *Rotation.from_matrix(wc[:3, :3]).as_quat()]
                    f.write(f"{stamp // 10**9}.{stamp % 10**9:09d} " +
                            " ".join(f"{v:.12f}" for v in values) + "\n")
        metadata = dict(
            frames=n, online_valid=sum(r["online_valid"] for r in rows),
            final_valid=sum(r["final_valid"] for r in rows),
            retired_anchors=len(self.retired_anchors),
            online="Raw pose immediately after frontend tracking; world gauge can change between frames.",
            final="Observed frame-to-reference pose composed with final reference pose; culled references follow recorded parent transforms. Not independently optimized per-frame poses.",
            missing="No interpolation or prediction inserted for lost/bootstrap-wait frames.",
            monocular="Final reconstruction disabled for mono until scale changes are recorded; online records remain available.",
        )
        (directory / "trajectory_frames_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata
