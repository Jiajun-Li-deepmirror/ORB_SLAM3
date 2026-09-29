# SPLG-SLAM

Stereo (optionally stereo-inertial) visual SLAM built on learned features
instead of hand-crafted ones: [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)
keypoints matched with [LightGlue](https://github.com/cvg/LightGlue), fused through a
[GTSAM](https://gtsam.org/) factor graph (stereo/mono reprojection, `CombinedImuFactor`
preintegration, pose-graph loop closure). It's an offline, dataset-driven mapping
pipeline: point it at a recording, get back a keyframe/map-point map and a trajectory.

Supported input formats, selected per run via a config file:

- **EuRoC** (`mav0/` layout — MH_xx, V1_xx, V2_xx sequences)
- **KITTI** odometry (`image_0`/`image_1` + `calib.txt`)
- **rosbag2** (`.db3`/`.mcap`), read directly at runtime — no conversion step needed.
  Works with any rig that publishes stereo infra frames + `/tf_static` (e.g. Intel
  RealSense D4xx), including inertial data if an IMU topic is present.

## Features

- Stereo (and mono) tracking with SuperPoint/LightGlue, PnP-based pose estimation,
  content-driven keyframe insertion.
- Local and global bundle adjustment (GTSAM), stereo reprojection factors.
- Optional tight IMU fusion: preintegrated `CombinedImuFactor`, ORB-SLAM3-style static/
  dynamic initialization, periodic re-initialization with a Pareto-frontier accept
  ratchet (gravity accuracy vs. motion-diversity), auto-detected accelerometer scale
  correction.
- Loop closure via global descriptor retrieval + geometric verification + pose-graph
  optimization.
- Pure-stereo (no-IMU) vertical-axis recovery: a fixed axis-convention bootstrap plus an
  optional, default-on PCA-based post-processing step that aligns the map's vertical axis
  using a near-planar-motion assumption (skipped automatically whenever IMU fusion is
  enabled, since IMU gravity alignment already provides a real reference).
- Dynamic-object masking (drops keypoints on detected people).
- Downstream tools: dense/elevation/Octomap map building, path planning, relocalization,
  multi-session map merging.

## Requirements

- Linux, Python 3.10
- A CUDA-capable GPU is strongly recommended (SuperPoint/LightGlue run on `cuda` when
  available, falling back to `cpu` otherwise — much slower).
- [GTSAM](https://gtsam.org/) with its Python bindings.

## Installation

```bash
git clone <this-repo-url>
cd splg_slam_frontend

python3.10 -m venv .venv
source .venv/bin/activate

# torch/torchvision first, matched to your CUDA version:
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

pip install -r requirements.txt
```

Notes:

- `requirements.txt` pins `numpy<2` and the `opencv-python`/`opencv-contrib-python`
  versions — GTSAM's PyPI wheel is built against the NumPy 1.x ABI and segfaults under
  NumPy 2, so don't upgrade `numpy` independently.
- LightGlue is installed straight from its GitHub repo (see `requirements.txt`); no
  separate SuperPoint checkout is needed, it ships as part of that package.
- Scripts under `scripts/` add `src/` to `sys.path` themselves at runtime, so no
  `pip install -e .` step is required — just run them from within the activated venv.

## Quick start

1. Pick or write a config (see `configs/*.yaml` for real examples covering EuRoC, KITTI,
   and RealSense rosbag2 recordings — stereo-only and stereo+IMU variants of each).
2. Point `dataset:` at your data and set `output.map_dir` to where you want results
   written.
3. Build the map:

```bash
python scripts/build_map.py configs/euroc_mh01_stereo_imu.yaml
```

This runs the full offline pipeline (tracking → local BA → loop closure → final global
BA) and writes a pickled `WorldMap` (keyframes, map points, poses) under `output.map_dir`.

4. Evaluate against ground truth (EuRoC `state_groundtruth_estimate0/data.csv`, or a
   KITTI sequence directory):

```bash
python scripts/eval_trajectory.py /path/to/map_dir/map.pkl /path/to/gt --out traj.png
```

Reports Sim(3)-aligned ATE (RMSE/median/etc.) and optionally a robust, outlier-trimmed
alignment; `--out` saves a trajectory plot.

5. Plot one or more trajectories (with or without ground truth) directly:

```bash
python scripts/plot_xy.py out.png --map run1=/path/to/map_dir/map.pkl --map run2=/path/to/other/map.pkl
python scripts/plot_xy_vs_gt.py gt.csv out.png --map run1=/path/to/map_dir/map.pkl
```

### Running against a live RealSense recording (rosbag2)

```yaml
dataset:
  kind: rosbag2
  bag_dir: /path/to/rosbag2_YYYY_MM_DD-HH_MM_SS
  frame_stride: 1
  max_frames: null
  use_sensor_depth: false
  tf_static_from: null   # point at another bag's dir to reuse its /tf_static if this one is missing it

imu:
  enabled: true   # false = pure stereo; see vertical-axis PCA correction under mapping:
```

The rig extrinsics/intrinsics and (if present) IMU calibration are read directly from the
bag's own `/tf_static`/camera-info topics — no manual per-device config beyond the IMU
noise-model constants (see `configs/*_stereo_imu.yaml` for the `imu:` block, including
`gyro_bias_prior`/`accel_bias_prior` warm-start fields you can fill in from your own
Allan-variance calibration).

### Other entry points

| Script | Purpose |
| --- | --- |
| `scripts/build_map.py` | Main offline SLAM pipeline (tracking, BA, loop closure) |
| `scripts/build_map_multi.py` | Build and merge maps from multiple sessions |
| `scripts/eval_trajectory.py` | ATE evaluation against ground truth |
| `scripts/eval_localization.py` | Relocalization accuracy evaluation |
| `scripts/eval_multi_trajectory.py` / `eval_registration_merge.py` | Multi-session merge evaluation |
| `scripts/merge_maps_registration.py` | Register/merge independently built maps |
| `scripts/dense_map.py` / `build_elevation_map.py` / `build_octomap*.py` / `view_octomap.py` | Dense reconstruction downstream of a built map |
| `scripts/plan_path.py` / `plan_path_3d.py` / `*_plan_loop*.py` | Path planning on a built map |
| `scripts/localize.py` | Relocalize new frames against an existing map |
| `scripts/control_tracking_demo.py` | Interactive tracking demo |
| `scripts/rosbag_to_euroc.py` | Convert a rosbag2 recording to an EuRoC-layout directory (rarely needed — `dataset.kind: rosbag2` reads bags directly) |
| `scripts/plot_xy.py` / `plot_xy_vs_gt.py` | Quick 2D trajectory plotting |

All scripts take a config path (or a built map path) as their first argument; run with
`--help` for the full option list.

## Configuration

Every run is driven by a single YAML file (see `configs/` for full worked examples).
Top-level sections:

- `dataset` — input source (`kind: euroc|kitti|rosbag2` + its path).
- `features` — SuperPoint keypoint budget.
- `dynamic_objects` — optional person-detection keypoint masking.
- `stereo` — block matching + adaptive max-depth estimation.
- `imu` — enable/disable IMU fusion, noise model, static/dynamic init thresholds, periodic
  re-initialization ratchet tolerances, bias warm-start priors.
- `tracking` — keyframe insertion policy, PnP/RANSAC thresholds, relocalization,
  mono-bootstrap parameters (unused in stereo mode).
- `mapping` — local/global BA cadence and windows, keyframe culling, the pure-stereo
  vertical-axis PCA correction (`vertical_axis_pca_correction`, on by default).
- `retrieval` / `loop_closure` — global descriptor retrieval and loop-closure gating
  (similarity/inlier/consistency thresholds, arc-length gating against same-segment
  pseudo-loops).
- `output` — where the resulting map is written.

Every field in the shipped configs carries an inline comment explaining its default and
when you'd want to change it — start from the closest matching example (dataset kind +
stereo-only vs. stereo+IMU) rather than writing one from scratch.

## Repository layout

```
src/splg_slam/
  data/         dataset loaders (euroc, kitti, rosbag2) — common load_stereo_rig/
                load_stereo_frames/load_imu_* interface
  features/     SuperPoint+LightGlue extraction/matching, person detector
  geometry/     camera models, stereo rectification, pose utils, Sim(3) alignment
  mapping/      tracker (OfflineMapper), local/global BA, IMU init/preintegration,
                loop closure, pose graph
  localization/ global descriptor retrieval, relocalization
  map/          WorldMap data structure + pickle I/O
  planning/     path planning on a built map
  control/      tracking-control demo glue
  utils/        seeding, misc helpers
scripts/        CLI entry points (see table above)
configs/        example configs for EuRoC/KITTI/RealSense, stereo-only and stereo+IMU
```
