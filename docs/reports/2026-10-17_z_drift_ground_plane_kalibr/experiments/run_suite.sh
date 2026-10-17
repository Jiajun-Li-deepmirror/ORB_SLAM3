#!/usr/bin/env bash
# Baseline (tf_static identity IMU rotation) vs Kalibr IMU-camera extrinsic, ground plane off,
# same code / seed / time_offset_ms (8.0). Usage: run_suite.sh [variant ...]  (2 builds in
# parallel; default variants: kalibr baseline; each needs config_<variant>.yaml;
# DATASETS="a b" limits the sequences)
cd "$(dirname "$0")"
R=$(pwd)
job() {
  variant=$1; name=$2; out=$R/$variant/$name
  [ -f "$out/map.pkl" ] && return 0
  mkdir -p "$out"
  cd /home/dm/slam_ws/src/splg_slam && MPLBACKEND=Agg .venv/bin/python -u scripts/build_map.py "$R/config_$variant.yaml" \
    --dataset-dir /home/dm/slam_ws/datasets/realsense/$name --output-dir "$out" --seed 0 > "$out/build.log" 2>&1
  echo "$variant $name exit=$?"
}
export -f job; export R
VARIANTS=("$@"); [ ${#VARIANTS[@]} -gt 0 ] || VARIANTS=(kalibr baseline)
DATASETS=${DATASETS:-"line_for line_for_2 line_lat cali_line_for turn_for loop_for loop_for_2 loop_lat cali_loop_for"}
for n in $DATASETS; do
  for v in "${VARIANTS[@]}"; do echo "$v $n"; done
done | xargs -P 2 -n 2 bash -c 'job "$0" "$1"'
