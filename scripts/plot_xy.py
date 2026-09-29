"""Plot estimated XY (top-down) and XZ (side) trajectories with no ground truth available
(e.g. a fresh real recording that hasn't been through lidar-GT post-processing yet). Also
reports each map's Z range vs. XY range as a gravity-alignment sanity check: a roughly
planar loop/walk should have Z varying far less than X/Y once gravity alignment has put the
world frame's Z axis pointing up - a large Z range relative to XY usually means the world
frame is tilted (gravity alignment didn't engage, or the leading window wasn't actually
static). The XZ side view makes exactly that tilt directly visible, not just inferrable
from the printed numbers.
"""
import argparse
import itertools
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splg_slam.geometry.pose_utils import camera_center
from splg_slam.map.io import load_map


def _plot_map_projection(ax, label, line_color, x_vals, y_vals, trusted_idx,
                          plain_mask, loop_mask, merge_mask, fallback_mask,
                          imu_only_mask, orphan_mask, segment_ids, reloc_xy,
                          path_len, z_range, show_labels, connect_start=True):
    """Draws one map's tracking-status-colored trajectory onto `ax`, using `x_vals`/
    `y_vals` (arrays parallel to the map's keyframe_ids_sorted() order) as the plotted
    coordinates - called once each for the XY view, the XZ view, and the Z-vs-time view,
    sharing all the same category masks so the three views are always consistent with
    each other. `reloc_xy` is an (N, 2) array of already-projected (x, y) relocalization
    points, or a length-0 array to omit them (the Z-vs-time view has no natural x
    position for a relocalization event, since only keyframes carry a stored timestamp)."""
    ax.plot(x_vals[trusted_idx], y_vals[trusted_idx], "-", lw=0.6,
             color=line_color, alpha=0.5, zorder=1)
    ax.scatter(x_vals[plain_mask], y_vals[plain_mask], s=6, c="tab:blue", zorder=2,
               label=(f"{label} tracking ({len(trusted_idx)} kf, path {path_len:.1f}m, "
                      f"Z range {z_range:.2f}m)") if show_labels else None)
    if loop_mask.any():
        ax.scatter(x_vals[loop_mask], y_vals[loop_mask], s=24, c="gold",
                   edgecolors="k", linewidths=0.4, zorder=3,
                   label=f"{label} loop closure ({int(loop_mask.sum())} kf)" if show_labels else None)
    if merge_mask.any():
        ax.scatter(x_vals[merge_mask], y_vals[merge_mask], s=30, c="limegreen",
                   edgecolors="k", linewidths=0.4, marker="D", zorder=4,
                   label=f"{label} Atlas merge ({int(merge_mask.sum())} kf)" if show_labels else None)
    if fallback_mask.any():
        # Deliberately blue (like plain tracking) since it IS now part of the one
        # continuous trusted trajectory (tinynav_slam-style: keep the final path whole
        # rather than stranding a chunk of it) - but a distinct marker/shade so it's
        # still visually clear this stretch's position is an unverified guess, not a
        # real measurement.
        ax.scatter(x_vals[fallback_mask], y_vals[fallback_mask], s=26, c="blue",
                   marker="^", edgecolors="k", linewidths=0.3, zorder=3,
                   label=(f"{label} fallback stitch, unverified "
                          f"({int(fallback_mask.sum())} kf)") if show_labels else None)
    if imu_only_mask.any():
        # tracking.continuous_imu_tracking (see tracker.py's _insert_imu_only_keyframe):
        # same segment as the surrounding trusted trajectory (no coordinate-frame jump to
        # mark), just a pose from pure IMU prediction instead of a verified vision
        # correspondence - orange to flag it as lower-confidence without implying it's a
        # disconnected island (fallback_mask's blue) or a discrete welding event
        # (merge_mask's green diamond).
        ax.scatter(x_vals[imu_only_mask], y_vals[imu_only_mask], s=20, c="orange",
                   marker="v", edgecolors="k", linewidths=0.3, zorder=3,
                   label=(f"{label} IMU-only (unverified, {int(imu_only_mask.sum())} kf)")
                         if show_labels else None)
    if len(reloc_xy):
        ax.scatter(reloc_xy[:, 0], reloc_xy[:, 1], s=40, c="red", marker="x", zorder=5,
                   label=f"{label} relocalization ({len(reloc_xy)})" if show_labels else None)

    # Orphan Atlas segment(s): never merged, so their coordinates share no common frame
    # with the trusted trajectory above or with each other. Each one gets its own
    # disconnected line (never bridged to the main trajectory or to a different orphan
    # segment) and a distinct, deliberately "untrusted-looking" style, so it reads as a
    # self-consistent island rather than a continuation of the real path.
    n_orphan_segments = 0
    if orphan_mask.any():
        for seg_id in sorted(set(segment_ids[orphan_mask].tolist())):
            seg_idx = np.flatnonzero(segment_ids == seg_id)
            n_orphan_segments += 1
            ax.plot(x_vals[seg_idx], y_vals[seg_idx], ":", lw=0.8, color="0.6", zorder=1)
            ax.scatter(x_vals[seg_idx], y_vals[seg_idx], s=14, c="none",
                       edgecolors="0.4", linewidths=0.8, marker="s", zorder=2,
                       label=(f"{label} orphan segment(s) ({int(orphan_mask.sum())} kf total, "
                              "unmerged, own coordinate frame")
                             if show_labels and n_orphan_segments == 1 else None)
    if connect_start:
        ax.plot(x_vals[0], y_vals[0], "o", ms=10, color=line_color, mfc="none", zorder=6)
    return n_orphan_segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_png", type=str)
    parser.add_argument("--map", action="append", required=True, metavar="LABEL=PATH",
                        help="repeatable, e.g. --map run1=/path/map.pkl --map run2=/path/map.pkl")
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    fig, (ax_xy, ax_xz, ax_zt) = plt.subplots(1, 3, figsize=(22, 7))

    # Tracking-status colors, shared across every --map given: normal tracking (blue),
    # a keyframe that is one endpoint of a verified loop-closure edge (yellow), and the
    # pose at a successful relocalization jump (red) - the latter isn't necessarily a
    # keyframe itself (relocalization only re-anchors ref_keyframe, see
    # OfflineMapper._on_tracking_failure), so it's plotted from world_map.relocalization_events
    # rather than off keyframe_ids_sorted().
    line_colors = itertools.cycle(["0.55", "0.35", "0.7", "0.2", "0.05", "0.85"])
    for spec, line_color in zip(args.map, line_colors):
        label, _, map_path = spec.partition("=")
        world_map = load_map(map_path)
        kf_ids = world_map.keyframe_ids_sorted()
        centers = np.array([camera_center(world_map.keyframes[i].pose_cw) for i in kf_ids])

        # An Atlas segment that never found a merge candidate (see build_map.py's orphan-
        # segment-reconciliation pass) keeps its own arbitrary, independent coordinate
        # frame forever - kept in the map rather than discarded (preserving that stretch
        # of exploration was the whole point of Atlas re-init in the first place, same
        # reasoning tinynav_slam's own "keep tracking on a degraded estimate rather than
        # stopping" philosophy follows), but it must NOT be treated as if it shares the
        # main map's frame: mixing its coordinates into xy_range/z_range/path_length (or
        # drawing one continuous line through it) silently corrupts those numbers with an
        # arbitrary rotation/offset - this is exactly what inflated one dataset's Z range
        # from 1.36m to 6m. Excluded from every "trusted trajectory" computation below and
        # rendered as its own disconnected island instead.
        main_segment_id = world_map.keyframes[kf_ids[0]].segment_id
        segment_ids = np.array([world_map.keyframes[kf_id].segment_id for kf_id in kf_ids])
        orphan_mask = segment_ids != main_segment_id
        trusted_idx = np.flatnonzero(~orphan_mask)
        trusted_centers = centers[trusted_idx]

        xy_range = trusted_centers[:, :2].max(axis=0) - trusted_centers[:, :2].min(axis=0)
        z_range = float(trusted_centers[:, 2].max() - trusted_centers[:, 2].min())
        path_len = float(np.linalg.norm(np.diff(trusted_centers, axis=0), axis=1).sum())

        loop_kf_ids = set()
        for kf_a, kf_b, _rel, _n in world_map.loop_edges:
            loop_kf_ids.add(kf_a)
            loop_kf_ids.add(kf_b)
        # Atlas segment merges (world_map.segment_merges) are a distinct event kind from a
        # same-segment loop-closure edge - a merge rigidly re-places a whole segment with
        # no residual drift left for a pose-graph edge to correct, so it's never added to
        # loop_edges (see WorldMap.segment_merges' docstring) except for the experimental
        # rigid-excursion path, which adds both; excluded from loop_kf_ids here either way
        # so a merge keyframe renders only as a merge point, not double-counted as gold too.
        merge_kf_ids = set()
        for new_kf_id, cand_kf_id in getattr(world_map, "segment_merges", []):
            if new_kf_id in world_map.keyframes:
                merge_kf_ids.add(new_kf_id)
            if cand_kf_id in world_map.keyframes:
                merge_kf_ids.add(cand_kf_id)
        loop_kf_ids -= merge_kf_ids
        # Fallback-stitched keyframes (world_map.fallback_stitches, see build_map.py): an
        # Atlas segment welded on with a zero-displacement guess after exhaustive
        # reconciliation still found no genuine visual match - NOT verified the way a
        # real segment_merges weld is. Its segment_id is already relabeled to the main
        # segment by the time this loads (transform_segment did that), so segment_id
        # alone can no longer tell it apart from originally-main-segment keyframes -
        # that's exactly why build_map.py records the full stitched kf_id list.
        fallback_kf_ids = set()
        for _anchor_kf_id, stitched_kf_ids in getattr(world_map, "fallback_stitches", []):
            fallback_kf_ids.update(k for k in stitched_kf_ids if k in world_map.keyframes)
        loop_kf_ids -= fallback_kf_ids
        merge_kf_ids -= fallback_kf_ids
        loop_mask = np.array([kf_id in loop_kf_ids for kf_id in kf_ids]) & ~orphan_mask
        merge_mask = np.array([kf_id in merge_kf_ids for kf_id in kf_ids]) & ~orphan_mask
        fallback_mask = np.array([kf_id in fallback_kf_ids for kf_id in kf_ids]) & ~orphan_mask
        # tracking.continuous_imu_tracking (see tracker.py's _insert_imu_only_keyframe) -
        # a pose from pure IMU prediction, no coordinate-frame change and not part of any
        # of the categories above.
        imu_only_mask = np.array([getattr(world_map.keyframes[k], "imu_only", False) for k in kf_ids]) & ~orphan_mask

        # Recomputed against each anchor keyframe's *current* pose (not a frozen snapshot -
        # see WorldMap.relocalization_event_pose) so a relocalization recorded early in the
        # run still reflects whatever pose-graph/BA corrections have since been applied to
        # its anchor. Falls back to the raw stored pose for an older map.pkl saved before
        # this event format existed (2-tuple instead of 3-tuple).
        reloc_events = getattr(world_map, "relocalization_events", [])
        reloc_poses = []
        for event in reloc_events:
            if len(event) == 3:
                pose = world_map.relocalization_event_pose(event)
            else:
                _fid, pose = event
            if pose is not None:
                reloc_poses.append(pose)
        reloc_centers = np.array([camera_center(pose) for pose in reloc_poses]) if reloc_poses else np.zeros((0, 3))
        if len(reloc_centers):
            # A degenerate EPnP/RANSAC solve can occasionally report a wildly wrong pose
            # despite a healthy inlier count (see tracker.py's _try_relocalize sanity check,
            # added after this was first seen - filtered here too so a map.pkl saved before
            # that fix doesn't corrupt the plot). A genuine relocalization solves PnP
            # against an existing keyframe's already-triangulated map points, so its result
            # must land within the map's own already-explored volume - a fixed margin
            # around the keyframe trajectory's bounding box, not a distance ratio (which a
            # long there-and-back path like this one would size far too loosely).
            margin = 20.0
            bbox_min = trusted_centers.min(axis=0) - margin
            bbox_max = trusted_centers.max(axis=0) + margin
            plausible = np.all((reloc_centers >= bbox_min) & (reloc_centers <= bbox_max), axis=1)
            n_dropped = int((~plausible).sum())
            if n_dropped:
                print(f"  {label}: dropped {n_dropped} implausible relocalization pose(s) from the plot")
            reloc_centers = reloc_centers[plausible]

        plain_mask = ~loop_mask & ~merge_mask & ~orphan_mask & ~fallback_mask & ~imu_only_mask

        # Elapsed time (s) since this map's own first keyframe - the natural x-axis for
        # the height-over-time view. Relocalization events have no analogous time here
        # (their own frame isn't a keyframe, so no timestamp is stored for it - only
        # kf_ids' worth of timestamps exist), so that view omits them (empty reloc_xy).
        t0 = world_map.keyframes[kf_ids[0]].timestamp_ns
        times_s = np.array([(world_map.keyframes[k].timestamp_ns - t0) / 1e9 for k in kf_ids])

        _plot_map_projection(ax_xy, label, line_color, centers[:, 0], centers[:, 1], trusted_idx,
                              plain_mask, loop_mask, merge_mask, fallback_mask, imu_only_mask,
                              orphan_mask, segment_ids, reloc_centers[:, :2], path_len, z_range,
                              show_labels=True)
        n_orphan_segments = _plot_map_projection(
            ax_xz, label, line_color, centers[:, 0], centers[:, 2], trusted_idx,
            plain_mask, loop_mask, merge_mask, fallback_mask, imu_only_mask,
            orphan_mask, segment_ids, reloc_centers[:, [0, 2]], path_len, z_range,
            show_labels=False,
        )
        _plot_map_projection(
            ax_zt, label, line_color, times_s, centers[:, 2], trusted_idx,
            plain_mask, loop_mask, merge_mask, fallback_mask, imu_only_mask,
            orphan_mask, segment_ids, np.zeros((0, 2)), path_len, z_range,
            show_labels=False, connect_start=False,
        )

        print(f"{label}: keyframes={len(trusted_idx)} (+{int(orphan_mask.sum())} orphaned in "
              f"{n_orphan_segments} unmerged segment(s), excluded from stats below) "
              f"path_length={path_len:.2f}m xy_range=({xy_range[0]:.2f}, {xy_range[1]:.2f})m "
              f"z_range={z_range:.3f}m loop_kf={int(loop_mask.sum())} merge_kf={int(merge_mask.sum())} "
              f"fallback_stitch_kf={int(fallback_mask.sum())} imu_only_kf={int(imu_only_mask.sum())} "
              f"relocalizations={len(reloc_centers)}")

    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("XY (top-down)")
    ax_xy.legend(fontsize=7)
    ax_xy.grid(alpha=0.3)
    ax_xy.set_aspect("equal")

    ax_xz.set_xlabel("x (m)")
    ax_xz.set_ylabel("z (m)")
    ax_xz.set_title("XZ (side view) - gravity-alignment sanity check")
    ax_xz.grid(alpha=0.3)
    ax_xz.set_aspect("equal")

    ax_zt.set_xlabel("time (s)")
    ax_zt.set_ylabel("z (m)")
    ax_zt.set_title("height over time")
    ax_zt.grid(alpha=0.3)

    fig.suptitle(args.title or "Estimated trajectory (no ground truth)")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=125)
    print(f"saved {args.out_png}")


if __name__ == "__main__":
    main()
