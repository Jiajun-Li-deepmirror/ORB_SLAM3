import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from splg_slam.control.velocity_controller import (
    PoseVelocityController,
    desired_orientation_from_direction,
    integrate_kinematics,
)
from splg_slam.planning.dense_grid_3d import make_segment_free_check_3d, plan_3d_dense
from splg_slam.planning.trajectory import build_trajectory


def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop verification of the 6-DOF velocity controller: plans one "
        "route with plan_3d_dense, starts a SIMULATED body pose deliberately offset from it "
        "(both position and orientation) and drives it with PoseVelocityController + a pure "
        "kinematic integrator (no physical robot to command, so this stands in for 'whatever "
        "the real low-level controller achieves'), then checks that position AND orientation "
        "error converge toward zero while tracking the path to the goal."
    )
    parser.add_argument("grid_npz", type=str)
    parser.add_argument("--start", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--goal", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--pos_offset", type=float, nargs=3, default=[1.0, -1.0, 0.3], metavar=("X", "Y", "Z"))
    parser.add_argument("--yaw_offset_deg", type=float, default=90.0)
    parser.add_argument("--robot_radius_m", type=float, default=0.2)
    parser.add_argument("--inflate_radius_m", type=float, default=0.3)
    parser.add_argument("--close_radius_m", type=float, default=0.2)
    parser.add_argument("--max_vel", type=float, default=2.0)
    parser.add_argument("--lookahead_m", type=float, default=0.8)
    parser.add_argument("--kp_pos", type=float, default=1.5)
    parser.add_argument("--kp_rot", type=float, default=2.5)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--max_sim_s", type=float, default=60.0)
    parser.add_argument("--goal_tol_m", type=float, default=0.3)
    parser.add_argument("--out_dir", type=str, default="results/control_tracking_demo")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = np.array(args.start)
    goal = np.array(args.goal)
    grid_data = np.load(args.grid_npz)
    grid = {k: grid_data[k] for k in ["occupied", "unknown", "origin", "resolution"]}
    grid["resolution"] = float(grid["resolution"])

    print("Planning reference route...")
    path_world, path_idx, blocked, origin, err = plan_3d_dense(
        grid, start, goal, allow_unknown=False, close_radius_m=args.close_radius_m,
        robot_radius_m=args.robot_radius_m, inflate_radius_m=args.inflate_radius_m,
    )
    if path_world is None:
        print(f"No reference path found: {err}")
        return
    is_segment_free = make_segment_free_check_3d(blocked, grid["origin"], grid["resolution"])
    traj = build_trajectory(path_world, is_segment_free, resample_ds=0.15, v_max=args.max_vel)
    path_xyz, path_dist = traj["positions"], traj["distance_m"]
    print(f"Reference path: {path_dist[-1]:.1f}m, {len(path_xyz)} waypoints")

    # Deliberately offset the simulated body's starting pose from the path start, in both
    # position and heading, so there's something real for the controller to converge from.
    forward0 = path_xyz[1] - path_xyz[0]
    r0_ref = desired_orientation_from_direction(forward0)
    r0_sim = r0_ref @ Rotation.from_euler("z", args.yaw_offset_deg, degrees=True).as_matrix()
    pose_wc = np.eye(4)
    pose_wc[:3, :3] = r0_sim
    pose_wc[:3, 3] = path_xyz[0] + np.array(args.pos_offset)

    controller = PoseVelocityController(kp_pos=args.kp_pos, kp_rot=args.kp_rot, max_lin_vel=args.max_vel)

    t = 0.0
    log = {"t": [], "pos": [], "pos_err_m": [], "rot_err_deg": [], "vel_cmd": []}
    while t < args.max_sim_s:
        vel_cmd, idx1 = controller.track_path(pose_wc, path_xyz, path_dist, args.lookahead_m)

        d = np.linalg.norm(path_xyz - pose_wc[:3, 3][None, :], axis=1)
        pos_err_m = float(d.min())
        forward = path_xyz[idx1] - pose_wc[:3, 3]
        r_ref_now = desired_orientation_from_direction(forward if np.linalg.norm(forward) > 1e-6 else pose_wc[:3, 0])
        r_err = pose_wc[:3, :3].T @ r_ref_now
        rot_err_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(r_err).as_rotvec())))

        log["t"].append(t)
        log["pos"].append(pose_wc[:3, 3].copy())
        log["pos_err_m"].append(pos_err_m)
        log["rot_err_deg"].append(rot_err_deg)
        log["vel_cmd"].append(vel_cmd.copy())

        if np.linalg.norm(pose_wc[:3, 3] - goal) < args.goal_tol_m:
            print(f"Reached goal at t={t:.1f}s")
            break

        pose_wc = integrate_kinematics(pose_wc, vel_cmd, args.dt)
        t += args.dt
    else:
        print(f"Did not reach goal within {args.max_sim_s}s (final dist to goal "
              f"{np.linalg.norm(pose_wc[:3, 3] - goal):.2f}m)")

    log = {k: np.array(v) for k, v in log.items()}
    print(f"Final position error to path: {log['pos_err_m'][-1]:.3f}m")
    print(f"Final orientation error: {log['rot_err_deg'][-1]:.1f}deg")
    print(f"Peak position error: {log['pos_err_m'].max():.3f}m, peak orientation error: {log['rot_err_deg'].max():.1f}deg")

    plot_results(path_xyz, log, out_dir)


def plot_results(path_xyz: np.ndarray, log: dict, out_dir: Path) -> None:
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2], "b-", label="reference path")
    ax1.plot(log["pos"][:, 0], log["pos"][:, 1], log["pos"][:, 2], "r-", label="simulated tracked pose")
    ax1.scatter(*log["pos"][0], c="green", marker="o", s=60, label="sim start (offset)")
    ax1.scatter(*path_xyz[0], c="blue", marker="x", s=60, label="path start")
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)"); ax1.set_zlabel("z (m)")
    ax1.legend(fontsize=8)
    ax1.set_title("3D reference path vs closed-loop simulated tracking")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(log["t"], log["pos_err_m"])
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("position error (m)")
    ax2.set_title("Position tracking error over time")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(log["t"], log["rot_err_deg"])
    ax3.set_xlabel("time (s)"); ax3.set_ylabel("orientation error (deg)")
    ax3.set_title("Orientation tracking error over time")
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    labels = ["vx", "vy", "vz", "vroll", "vpitch", "vyaw"]
    for i, lab in enumerate(labels):
        ax4.plot(log["t"], log["vel_cmd"][:, i], label=lab)
    ax4.set_xlabel("time (s)"); ax4.set_ylabel("commanded velocity (m/s or rad/s)")
    ax4.set_title("6-DOF body-frame velocity commands")
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(alpha=0.3)

    plt.suptitle("PoseVelocityController closed-loop tracking verification", fontsize=13)
    plt.tight_layout()
    out_path = out_dir / "control_tracking_verification.png"
    plt.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
