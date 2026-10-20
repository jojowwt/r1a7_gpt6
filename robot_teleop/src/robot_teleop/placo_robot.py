"""Whole-body inverse kinematics via Placo.

Both wrists are driven as soft frame tasks. Body joints (waist, lift) can either
join the whole-body solution or be masked out and driven by higher-level logic
(head and waist posture). The floating base is always masked: inference
commands the upper body only, and base motion is a separate velocity command.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import placo
from scipy.spatial.transform import Rotation


@dataclass
class IKTarget:
    frame_name: str
    pos: np.ndarray
    quat_xyzw: np.ndarray


def _matrix_from_pos_quat_xyzw(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    transform[:3, 3] = np.asarray(pos, dtype=float)
    return transform


def _pos_quat_xyzw_from_matrix(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return transform[:3, 3].copy(), Rotation.from_matrix(transform[:3, :3]).as_quat()


def _resolve_mesh_path(urdf_path: Path, filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    resolved = (urdf_path.parent / path).resolve()
    if resolved.exists():
        return resolved
    fallback = (urdf_path.parent / "meshes" / path.name).resolve()
    if fallback.exists():
        return fallback
    return resolved


def _make_placo_loadable_urdf(urdf_path: Path) -> Path:
    """Rewrite relative mesh paths to absolute ones; Placo cannot resolve them itself."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    changed = False
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename and not Path(filename).is_absolute():
            mesh.set("filename", str(_resolve_mesh_path(urdf_path, filename)))
            changed = True
    # A MuJoCo meshdir would be prepended to filenames that are already relative
    # to the URDF; drop it now that every path is absolute.
    for mujoco_tag in root.findall("mujoco"):
        for compiler in mujoco_tag.findall("compiler"):
            if compiler.attrib.pop("meshdir", None) is not None:
                changed = True
    if not changed:
        return urdf_path
    out_dir = Path(tempfile.gettempdir()) / "robot_teleop_placo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{urdf_path.stem}_placo.urdf"
    tree.write(out_path, encoding="unicode")
    return out_path


class PlacoIKSolver:
    """Placo whole-body IK wrapper.

    The floating base is masked. Arm and body joints participate in IK, while
    joints listed in `masked_joints` (for example a lift column, or the waist yaw
    in joystick mode) are kept out of the solver and driven externally.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        controlled_joints: list[str],
        left_ee_frame: str,
        right_ee_frame: str,
        dt: float,
        regularization_weight: float = 1e-2,
        body_joints: list[str] | None = None,
        body_regularization_weight: float = 0.2,
        body_joint_regularization_weights: dict[str, float] | None = None,
        masked_joints: list[str] | None = None,
        max_joint_step: float = 0.12,
        reset_joint_positions: dict[str, float] | None = None,
    ) -> None:
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.controlled_joints = list(controlled_joints)
        self.body_joints = list(body_joints or [])
        self.body_joint_regularization_weights = dict(body_joint_regularization_weights or {})
        self.masked_joints = list(masked_joints or [])
        self.max_joint_step = float(max_joint_step)
        self.name_to_index = {name: i for i, name in enumerate(self.controlled_joints)}
        self.placo_urdf_path = _make_placo_loadable_urdf(self.urdf_path)
        self.robot = placo.RobotWrapper(str(self.placo_urdf_path))
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.dt = dt
        self.solver.mask_fbase(True)
        self.solver.enable_joint_limits(True)
        self.solver.enable_velocity_limits(True)

        missing = [name for name in self.controlled_joints if name not in set(self.robot.joint_names())]
        missing += [name for name in self.masked_joints if name not in set(self.robot.joint_names())]
        if missing:
            raise ValueError(f"URDF missing placo joints: {missing}")
        for name in self.masked_joints:
            self.solver.mask_dof(name)

        self.left_task = self.solver.add_frame_task(left_ee_frame, self.robot.get_T_world_frame(left_ee_frame))
        self.left_task.configure("left_ee", "soft", 1.0)
        self.right_task = self.solver.add_frame_task(right_ee_frame, self.robot.get_T_world_frame(right_ee_frame))
        self.right_task.configure("right_ee", "soft", 1.0)
        self.left_manipulability_task = self.solver.add_manipulability_task(left_ee_frame, "both", 1.0)
        self.left_manipulability_task.configure("left_manipulability", "soft", 1e-2)
        self.right_manipulability_task = self.solver.add_manipulability_task(right_ee_frame, "both", 1.0)
        self.right_manipulability_task.configure("right_manipulability", "soft", 1e-2)

        # Soft regularization toward the reset posture keeps the redundant DoF
        # from drifting into awkward configurations.
        reset_joint_positions = dict(reset_joint_positions or {})
        self.reset_joints = {name: float(reset_joint_positions.get(name, 0.0)) for name in self.controlled_joints}
        self.joints_task = self.solver.add_joints_task()
        self.joints_task.set_joints(self.reset_joints)
        self.joints_task.configure("reset_posture_regularization", "soft", regularization_weight)

        # Per-body-joint weights: a larger weight makes that joint contribute
        # less to the end-effector solution.
        self.body_joints_tasks = {}
        for name in self.body_joints:
            if name in self.masked_joints:
                continue
            weight = float(self.body_joint_regularization_weights.get(name, body_regularization_weight))
            task = self.solver.add_joints_task()
            task.set_joints({name: float(reset_joint_positions.get(name, 0.0))})
            task.configure(f"body_reset_regularization_{name}", "soft", weight)
            self.body_joints_tasks[name] = task
        self.robot.update_kinematics()

    def set_joint_positions(self, joints: dict[str, float]) -> None:
        for name, value in joints.items():
            self.robot.state.q[self.robot.get_joint_offset(name)] = float(value)
        self.robot.update_kinematics()

    def reset(self) -> None:
        for name, value in self.reset_joints.items():
            self.robot.state.q[self.robot.get_joint_offset(name)] = value
        self.robot.update_kinematics()

    def frame_pose(self, frame_name: str) -> tuple[np.ndarray, np.ndarray]:
        self.robot.update_kinematics()
        return _pos_quat_xyzw_from_matrix(self.robot.get_T_world_frame(frame_name))

    def joint_qpos(self, joint_names: list[str]) -> np.ndarray:
        values: list[float] = []
        for name in joint_names:
            values.append(float(self.robot.state.q[self.robot.get_joint_offset(name)]))
        return np.asarray(values, dtype=float)

    def solve_ik(
        self,
        targets: list[IKTarget],
        active_left: bool,
        active_right: bool,
        left_ee_frame: str,
        right_ee_frame: str,
        iters: int,
    ) -> np.ndarray:
        """Solve for the commanded joint angles, with safety fallbacks.

        An inactive wrist is pinned to its current pose so it holds still. On
        solver failure, non-finite output, or a joint step beyond
        `max_joint_step`, the result is clamped or rolled back rather than
        letting a bad solution reach the robot.
        """
        current_left = self.robot.get_T_world_frame(left_ee_frame)
        current_right = self.robot.get_T_world_frame(right_ee_frame)
        self.left_task.T_world_frame = current_left
        self.right_task.T_world_frame = current_right

        for target in targets:
            transform = _matrix_from_pos_quat_xyzw(target.pos, target.quat_xyzw)
            if target.frame_name == left_ee_frame and active_left:
                self.left_task.T_world_frame = transform
            elif target.frame_name == right_ee_frame and active_right:
                self.right_task.T_world_frame = transform

        q_before = self.joint_qpos(self.controlled_joints)
        try:
            for _ in range(max(1, iters)):
                self.solver.solve(True)
                self.robot.update_kinematics()
        except RuntimeError as exc:
            self.set_joint_positions(dict(zip(self.controlled_joints, q_before)))
            print(f"[ik] solver failed, keeping previous q: {exc}")
            return q_before

        q_after = self.joint_qpos(self.controlled_joints)
        if not np.all(np.isfinite(q_after)):
            self.set_joint_positions(dict(zip(self.controlled_joints, q_before)))
            print("[ik] produced non-finite q, keeping previous q")
            return q_before
        dq = np.clip(q_after - q_before, -self.max_joint_step, self.max_joint_step)
        if np.any(np.abs(q_after - q_before) > self.max_joint_step):
            q_after = q_before + dq
            self.set_joint_positions(dict(zip(self.controlled_joints, q_after)))
        return q_after

    def arm_command_14(self, left_arm_joints: list[str], right_arm_joints: list[str]) -> np.ndarray:
        return self.joint_qpos(left_arm_joints + right_arm_joints)
