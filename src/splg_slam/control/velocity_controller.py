import numpy as np
from scipy.spatial.transform import Rotation

# Body axis convention (REP-103 style): x=forward, y=left, z=up, right-handed.
# Angular velocity command axes follow the same convention: vroll about x, vpitch about y,
# vyaw about z.


def desired_orientation_from_direction(forward: np.ndarray, up_hint: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Builds a world-from-body rotation matrix whose x-axis points along `forward` (the
    direction of travel - can have a vertical component, e.g. climbing/diving) and whose
    roll about that axis is fixed by keeping z as close to `up_hint` (world up) as possible.
    This is "face where you're going, never bank" - a reasonable default orientation
    reference for a vehicle with no independent attitude goal beyond facing its own path,
    used when the planner only produces a position trajectory (no orientation reference)."""
    x = forward / np.linalg.norm(forward)
    y = np.cross(up_hint, x)
    if np.linalg.norm(y) < 1e-6:  # forward nearly parallel to up_hint - up_hint is degenerate here
        y = np.cross(x, np.array([1.0, 0.0, 0.0]))
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)  # columns are the body axes expressed in world frame


class PoseVelocityController:
    """Kinematic (velocity-level) pose-tracking controller: turns a position+orientation
    reference into a 6-DOF BODY-FRAME velocity command (vx, vy, vz, vroll, vpitch, vyaw) -
    the same "cmd_vel"-style interface a real low-level controller/motor-allocation stack
    would take as input. This is proportional pose tracking at the velocity level (command
    velocity directly from pose error), not a full-dynamics force/torque controller - the
    right layer to sit directly below the trajectory planner, feeding whatever actually
    drives the motors."""

    def __init__(
        self, kp_pos: float = 1.5, kp_rot: float = 2.5,
        max_lin_vel: float = 2.0, max_ang_vel: float = 1.5,
    ):
        self.kp_pos = kp_pos
        self.kp_rot = kp_rot
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel

    def compute(self, pose_wc_current: np.ndarray, pos_ref: np.ndarray, r_ref: np.ndarray) -> np.ndarray:
        """pose_wc_current: 4x4 current world-from-body pose (R_wc rotation, t_wc =
        position). pos_ref: (3,) reference world position. r_ref: (3,3) reference
        world-from-body orientation. Returns [vx, vy, vz, vroll, vpitch, vyaw] in the body
        frame."""
        r_wc = pose_wc_current[:3, :3]
        pos_current = pose_wc_current[:3, 3]

        pos_err_world = pos_ref - pos_current
        pos_err_body = r_wc.T @ pos_err_world
        v_lin = np.clip(self.kp_pos * pos_err_body, -self.max_lin_vel, self.max_lin_vel)

        r_err_body = r_wc.T @ r_ref  # rotation from current orientation to reference, in body axes
        rotvec_body = Rotation.from_matrix(r_err_body).as_rotvec()
        v_ang = np.clip(self.kp_rot * rotvec_body, -self.max_ang_vel, self.max_ang_vel)

        return np.concatenate([v_lin, v_ang])

    def track_path(
        self, pose_wc_current: np.ndarray, path_xyz: np.ndarray, path_dist: np.ndarray,
        lookahead_m: float, up_hint: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ) -> tuple[np.ndarray, int]:
        """Pure-pursuit-style convenience: picks a lookahead point `lookahead_m` ahead (by
        arc length) of the closest point on `path_xyz` to the current position, builds a
        position+orientation reference from it (bearing to the lookahead point doubles as
        the desired heading), and returns (velocity_command, lookahead_index)."""
        pos_current = pose_wc_current[:3, 3]
        d = np.linalg.norm(path_xyz - pos_current[None, :], axis=1)
        idx0 = int(np.argmin(d))
        target_dist = path_dist[idx0] + lookahead_m
        idx1 = int(np.searchsorted(path_dist, target_dist))
        idx1 = min(idx1, len(path_xyz) - 1)

        pos_ref = path_xyz[idx1]
        forward = pos_ref - pos_current
        if np.linalg.norm(forward) < 1e-6:
            forward = pose_wc_current[:3, 0]  # already there - hold current heading
        r_ref = desired_orientation_from_direction(forward, up_hint)

        return self.compute(pose_wc_current, pos_ref, r_ref), idx1


def integrate_kinematics(pose_wc: np.ndarray, velocity_cmd: np.ndarray, dt: float) -> np.ndarray:
    """Advances a rigid-body world-from-body pose by one step under a body-frame velocity
    command [vx,vy,vz,vroll,vpitch,vyaw] - the simple kinematic (no mass/inertia) model a
    closed-loop control simulation integrates against, standing in for "whatever the real
    robot's low-level motor controller achieves" when there's no physical actuator to
    command."""
    r_wc = pose_wc[:3, :3]
    pos = pose_wc[:3, 3]
    v_lin, v_ang = velocity_cmd[:3], velocity_cmd[3:]

    new_pos = pos + r_wc @ v_lin * dt
    new_r = r_wc @ Rotation.from_rotvec(v_ang * dt).as_matrix()

    new_pose = np.eye(4)
    new_pose[:3, :3] = new_r
    new_pose[:3, 3] = new_pos
    return new_pose
