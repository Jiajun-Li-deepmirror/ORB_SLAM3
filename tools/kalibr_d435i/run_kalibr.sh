#!/usr/bin/env bash
# Full D435i stereo + IMU calibration with Kalibr (docker image kalibr:noetic).
# Usage: run_kalibr.sh <rosbag2_dir> [work_dir]
#   0) check_target.py: target config + per-block detection/motion quality (FORCE=1 to ignore)
#   1) converts the rosbag2 recording to a ROS1 bag (rosbag2_to_kalibr.py)
#   2) kalibr_calibrate_cameras      -> kalibr-camchain.yaml, kalibr-report-cam.pdf
#   3) kalibr_calibrate_imu_camera   -> kalibr-camchain-imucam.yaml, kalibr-report-imucam.pdf
#   4) kalibr_to_splg.py             -> splg_imu_override.yaml (config snippet) + summary
# Set SKIP_CAM=1 to reuse the bag's factory intrinsics instead of step 2 (the D435i infra
# streams are rectified on-device with those intrinsics, which is also what SLAM uses).
# Set SKIP_CHECK=1 to skip step 0 when it already passed for this recording.
# Set BAG_FROM_TO="15 57" to use only that window (seconds from bag start) - long static
# stretches add no information but slow Kalibr down.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BAG2=$(realpath "$1")
WORK=$(realpath -m "${2:-$HERE/runs/$(basename "$BAG2")}")
REPO=$(cd "$HERE/../.." && pwd)
PY=${PY:-$REPO/.venv/bin/python}
mkdir -p "$WORK"
cp "$HERE/aprilgrid.yaml" "$HERE/imu.yaml" "$WORK/"
# Abort early if aprilgrid.yaml (rows/cols/spacing) does not match the recorded board.
if [ "${SKIP_CHECK:-0}" != 1 ]; then
  "$PY" "$HERE/check_target.py" "$BAG2" "$WORK/aprilgrid.yaml" --work "$WORK/check" | tee "$WORK/check_target.log" \
    || [ "${FORCE:-0}" = 1 ]
fi
[ -f "$WORK/kalibr.bag" ] || "$PY" "$HERE/rosbag2_to_kalibr.py" "$BAG2" "$WORK/kalibr.bag"

kalibr() {
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e ROS_HOME=/tmp/.ros -e MPLBACKEND=Agg \
    -v "$WORK":/data -w /data --entrypoint /bin/bash kalibr:noetic \
    -c "source /catkin_ws/devel/setup.bash && $*"
}

if [ "${SKIP_CAM:-0}" = 1 ]; then
  cp "$WORK/kalibr-camchain-factory.yaml" "$WORK/kalibr-camchain.yaml"
else
  kalibr rosrun kalibr kalibr_calibrate_cameras --bag /data/kalibr.bag --target /data/aprilgrid.yaml \
    --models pinhole-radtan pinhole-radtan --topics /cam0/image_raw /cam1/image_raw \
    --bag-freq 4 ${BAG_FROM_TO:+--bag-from-to $BAG_FROM_TO} --dont-show-report 2>&1 | tee "$WORK/calibrate_cameras.log"
fi
kalibr rosrun kalibr kalibr_calibrate_imu_camera --bag /data/kalibr.bag --cam /data/kalibr-camchain.yaml \
  --imu /data/imu.yaml --target /data/aprilgrid.yaml ${BAG_FROM_TO:+--bag-from-to $BAG_FROM_TO} --dont-show-report 2>&1 | tee "$WORK/calibrate_imu_camera.log"
"$PY" "$HERE/kalibr_to_splg.py" "$WORK" | tee "$WORK/summary.txt"
