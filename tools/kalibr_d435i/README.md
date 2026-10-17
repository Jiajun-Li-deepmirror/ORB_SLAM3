# D435i 双目 + IMU Kalibr 标定

标定板：AprilGrid（tag36h11），tag 黑色外框边长 55 mm，相邻 tag 间隙 16.5 mm，
即 `tagSize: 0.055`、`tagSpacing: 0.3`，见 `aprilgrid.yaml`。行列数默认 6×6，和实物不一致时请修改 `tagRows`/`tagCols`。
`run_kalibr.sh` 会先用 `check_target.py` 从录制数据里核对 tag ID 范围、行列布局和间隙比例，不一致就停止。
tagSize 的绝对尺寸无法从图像本身检验：标定完成后 `summary.txt` 会把 Kalibr 估出的双目基线和出厂基线（约 50.3 mm）对比，两者明显不一致时说明 tagSize 有误。

## 1. 录制（在原来录数据的设备上）

驱动启动：关闭红外投射器（散斑会干扰 AprilTag 检测），不开 depth/color，IMU 不合并（和之前的数据保持一致）。

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=false enable_depth:=false \
  enable_infra1:=true enable_infra2:=true \
  depth_module.infra_profile:=848x480x30 \
  depth_module.emitter_enabled:=0 \
  enable_gyro:=true enable_accel:=true gyro_fps:=200 accel_fps:=100 \
  unite_imu_method:=0
```

- 旧版 realsense-ros（4.51 及更早）把 `depth_module.infra_profile` 换成 `depth_module.profile:=848x480x30`。
  可以用 `ros2 param list /camera/camera | grep -E "profile|emitter|exposure"` 确认参数名。
- 启动后用 `ros2 param get /camera/camera depth_module.emitter_enabled` 确认是 0，并用 rqt_image_view 看 infra1 画面里没有散斑点。
- 运动模糊会明显降低精度：保证光照充足。可以的话关闭自动曝光，把曝光设短一些，例如
  `depth_module.enable_auto_exposure:=false depth_module.exposure:=4000`（单位 µs，画面太暗就调大）。

录制：

```bash
ros2 bag record -o kalibr_d435i_$(date +%Y%m%d_%H%M%S) \
  /camera/camera/infra1/image_rect_raw /camera/camera/infra1/camera_info /camera/camera/infra1/metadata \
  /camera/camera/infra2/image_rect_raw /camera/camera/infra2/camera_info /camera/camera/infra2/metadata \
  /camera/camera/gyro/sample /camera/camera/accel/sample \
  /camera/camera/gyro/imu_info /camera/camera/accel/imu_info \
  /camera/camera/extrinsics/depth_to_infra1 /camera/camera/extrinsics/depth_to_infra2 \
  /camera/camera/extrinsics/depth_to_gyro /camera/camera/extrinsics/depth_to_accel \
  /tf_static
```

动作要领（总长 60–90 s）：

1. 标定板固定不动（贴墙或平放在桌上），相机距离板 0.4–0.8 m。tag 只有 55 mm，1 m 外在画面里只剩约 24 像素。
   开始时让全部 36 个 tag 都在画面内。
2. 开始先静止 3–5 s。
3. 依次做：绕三个轴各来回旋转（俯仰、偏航、横滚），再沿三个方向各来回平移（上下、左右、前后），最后做几段组合运动。每种 3–4 次。
4. **慢**：单次摆动 ±20–30°，用 1–2 s 完成，角速度约 20–40°/s。
   第一次录制（2026-10-15 22:39）运动段每轴 RMS 50–80°/s，15 s 里一帧 tag 都没检测到。
5. 全程保持大部分 tag 在画面里，不要遮挡，不要过曝。
6. 结束前再静止 3–5 s。

拖影取决于曝光时间：录之前在 rqt_image_view 里按上面的速度晃一晃，确认 tag 的黑白格边缘清晰。
不清晰就关闭自动曝光、缩短曝光并提高增益，或者补光，例如
`depth_module.enable_auto_exposure:=false depth_module.exposure:=3000 depth_module.gain:=80`
（曝光单位 µs；画面太暗先加光或加增益，不要加长曝光）。

录完可以先单独跑预检，几分钟内就知道能不能用：
`.venv/bin/python tools/kalibr_d435i/check_target.py <bag目录>`（在仓库根目录执行）
它按 3 s 分段列出 tag 检测率和角速度，要求至少 30 s 运动中能看清标定板、且没有超过 3 s 的检测空档。

metadata 里有每帧实际曝光时间。图像时间戳打在曝光结束时，IMU 与图像的时间偏移会随曝光变化约 −曝光/2，
所以 Kalibr 估出的时间偏移只对这次录制的曝光有效，`kalibr_to_splg.py` 不会把它写进 SLAM 配置。

必须用和 SLAM 数据同一台相机（IMU 内参修正是按这台机器标定的）。

录好后把整个 bag 目录拷到数据集目录下（本机为 `/home/dm/slam_ws/datasets/realsense/`）。

## 2. 标定

先构建 Kalibr 镜像（只需一次，约 8 GB）：

```bash
git clone --depth 1 https://github.com/ethz-asl/kalibr.git kalibr_src && cd kalibr_src
sed -i 's/catkin build -j$(nproc)/catkin build -j6/' Dockerfile_ros1_20_04   # 限制并行数，避免编译时内存不足
docker build -t kalibr:noetic -f Dockerfile_ros1_20_04 .
```

预检脚本还需要一个装有 python3、opencv 和 `aprilgrid` pip 包的镜像，用环境变量 `APRILGRID_IMAGE` 指定（默认 `calibr-looper:humble`）。

然后在仓库根目录运行：

```bash
SKIP_CAM=1 BAG_FROM_TO="15 57" tools/kalibr_d435i/run_kalibr.sh <数据集目录>/<kalibr_bag_dir>
```

依次执行：标定板参数和录制质量预检、rosbag2 → ROS1 bag 转换、双目内参标定、相机-IMU 标定，最后生成 SLAM 配置片段。

- `SKIP_CAM=1`：跳过双目内参标定，直接用出厂内参。D435i 的红外图像已经在相机芯片上按出厂内参校正过，SLAM 用的也是出厂内参。2026-10-15 那次 Kalibr 内参标定在 cam0 焦距初始化时失败（初值 4083，正常约 432），所以最终结果用的是出厂内参。
- `BAG_FROM_TO="开始 结束"`：只用这段时间（秒，从 bag 开头算），去掉首尾过长的静止段。
- `SKIP_CHECK=1`：跳过预检；`FORCE=1`：预检不通过也继续。

结果写在 `tools/kalibr_d435i/runs/<bag 名>/`（不纳入版本管理）：

- `kalibr-report-imucam.pdf`：检查重投影误差和 IMU 残差。
- `kalibr-camchain-imucam.yaml`：Kalibr 原始结果。
- `splg_imu_override.yaml`：`imu.t_cam0_body_override` 配置片段。Kalibr 的时间偏移只作为注释保留（见上文，它随曝光变化）。
- `summary.txt`：和 tf_static 出厂外参的对比，以及双目基线核对。

2026-10-15 这次标定的结果保存在 `results/kalibr_d435i_20261015_224910/`，已写入 `configs/realsense_current_stereo_imu.yaml`。
完整的分析和实验见 `docs/reports/2026-10-17_z_drift_ground_plane_kalibr/report.md`。
