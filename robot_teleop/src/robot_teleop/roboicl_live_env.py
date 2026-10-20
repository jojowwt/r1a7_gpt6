"""Real-robot environment for the RoboICL rollout loop.

RoboICL drives a RoboDojo/Isaac Sim environment, but its execution loop
(`roboicl/rollout.py`) only touches seven members of that environment. This
module implements exactly those against the real R1-A7, so the in-context
policy, prompting and memory code runs unchanged while the simulator is
replaced by hardware:

    get_obs()        three RGB streams plus proprioception
    take_action(a)   Cartesian target -> IK -> the R1-A7 DDS publisher
    is_episode_end()
    step_lim         step budget
    save_dir         where the policy writes its logs
    ik_continuity_guard / _policy_failure

Actions arrive already validated by `validate_actions`, which enforces the
protocol's per-step translation and rotation bounds and resolves `ee_delta`
into absolute poses. What remains here is solving IK for those poses and
sending the result through the R1-A7 publisher, so PD gains,
the command ramp and the motion-mode release all behave identically.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .camera_convention import opencv_to_usd

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
# image_server slot -> RoboICL camera name. The head is stereo; RoboICL
# consumes a single head view, so the left eye is used.
CAMERA_SLOTS = {"cam_head": 0, "cam_left_wrist": 2, "cam_right_wrist": 3}
POLICY_HEAD_SIZE = (640, 480)
ARMS = ("left", "right")


def _policy_camera_frame(camera: str, frame: np.ndarray) -> np.ndarray:
    """Match live head RGB to the 640x480 head images in the TRAIN bundle."""
    pixels = np.asarray(frame)
    if camera != "cam_head" or pixels.shape[:2] == (POLICY_HEAD_SIZE[1], POLICY_HEAD_SIZE[0]):
        return pixels
    import cv2

    interpolation = cv2.INTER_AREA if (
        pixels.shape[1] > POLICY_HEAD_SIZE[0] or pixels.shape[0] > POLICY_HEAD_SIZE[1]
    ) else cv2.INTER_LINEAR
    return cv2.resize(pixels, POLICY_HEAD_SIZE, interpolation=interpolation)


class _CalibratedCamera:
    """One camera, shaped like the Isaac Sim camera `rollout.py` reaches for.

    `capture_head_calibration` reads the head camera's world pose and
    intrinsics straight off the simulator's camera object. On hardware the same
    quantities come from the hand-eye calibration composed with forward
    kinematics, so this exposes them through the same three accessors instead
    of duplicating that function.
    """

    def __init__(self, intrinsic: np.ndarray, camera_to_world: np.ndarray, size: tuple[int, int]):
        self._intrinsic = np.asarray(intrinsic, dtype=float).reshape(3, 3)
        self._transform = np.asarray(camera_to_world, dtype=float).reshape(4, 4)
        self._size = (int(size[0]), int(size[1]))

    def get_world_pose(self, camera_axes: str = "usd"):
        """Position and quaternion in w, x, y, z order, as USD reports it."""
        if camera_axes != "usd":
            raise ValueError(f"Only the usd camera convention is provided, got {camera_axes!r}")
        transform = opencv_to_usd(self._transform)
        quat_xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
        return transform[:3, 3].copy(), _quat_xyzw_to_wxyz(quat_xyzw)

    def get_intrinsics_matrix(self, device: str = "cpu"):
        del device
        return self._intrinsic.copy()

    def get_resolution(self):
        return [self._size[0], self._size[1]]


class _CameraManager:
    """Minimal stand-in for the simulator's camera manager."""

    def __init__(self, cameras: dict[str, _CalibratedCamera]):
        self.camera_names = [list(cameras)]
        self.cameras = [list(cameras.values())]


class _SceneShim:
    """Provides the environment origin the head calibration subtracts.

    Recorded poses are already relative to the robot's fixed base, so the
    origin is zero: subtracting it is a no-op that keeps the shared code path.
    """

    env_origins = np.zeros((1, 3), dtype=float)


class _RobotManagerShim:
    scene = _SceneShim()


def _quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).reshape(4)
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)


def _quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).reshape(4)
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=float)


class R1A7LiveEnv:
    """Expose the real R1-A7 through the interface `rollout.py` expects."""

    num_envs = 1

    def __init__(
        self,
        cfg,
        publisher,
        ik_solver,
        geometry,
        images,
        instruction: str,
        save_dir: str | Path,
        step_lim: int = 240,
        control_hz: float = 25.0,
        settle_s: float = 0.0,
        monitor=None,
    ) -> None:
        self.cfg = cfg
        self.publisher = publisher
        self.ik = ik_solver
        self.geometry = geometry
        self.images = images
        self.instruction = str(instruction)
        self.save_dir = str(Path(save_dir).expanduser().resolve())
        self.step_lim = int(step_lim)
        self.control_hz = float(control_hz)
        self.settle_s = float(settle_s)
        self.monitor = monitor
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.monitor is not None:
            self.monitor.step_lim = self.step_lim

        # Set by rollout.py; present so attribute access never fails.
        self.ik_continuity_guard = None
        self._policy_failure: dict | None = None

        self._steps = 0
        self._arm_joints = list(cfg.left_arm_joints) + list(cfg.right_arm_joints)
        self._gripper_joints = {
            "left": list(cfg.left_gripper_joints),
            "right": list(cfg.right_gripper_joints),
        }
        self._ee_frames = {"left": cfg.left_ee_body, "right": cfg.right_ee_body}
        self._last_action: dict | None = None
        self._action_in_chunk = 0
        self._arm_slices = {"left": slice(0, 7), "right": slice(7, 14)}
        self._joint_limits = self._read_joint_limits()
        self._quit_requested = False
        self._frame_thread: threading.Thread | None = None
        self._frames_running = False

    def start_frame_feed(self, fps: float = 2.5) -> None:
        """Keep pushing camera frames to the monitor on a worker thread.

        A model request can take minutes while `get_obs` is only called between
        actions, so a UI fed solely by the control loop would freeze during
        exactly the wait the operator most wants to watch.
        """
        if self.monitor is None or self._frame_thread is not None:
            return
        period = 1.0 / max(float(fps), 0.1)
        self._frames_running = True

        def pump() -> None:
            while self._frames_running:
                start = time.monotonic()
                try:
                    render = getattr(self.monitor, "render", None)
                    if callable(render) and render(self.monitor_frames()):
                        self._quit_requested = True
                except Exception:
                    pass
                elapsed = time.monotonic() - start
                if elapsed < period:
                    time.sleep(period - elapsed)

        self._frame_thread = threading.Thread(target=pump, name="roboicl-frames", daemon=True)
        self._frame_thread.start()

    def stop_frame_feed(self) -> None:
        self._frames_running = False
        thread, self._frame_thread = self._frame_thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _read_joint_limits(self) -> dict[str, np.ndarray]:
        """Per-arm joint limits from the URDF, as the continuity guard expects.

        The guard needs limits to map a periodic joint target onto its nearest
        legal equivalent before measuring continuity. These arms are revolute
        with finite limits, so none are reported periodic.
        """
        import xml.etree.ElementTree as ET

        bounds: dict[str, tuple[float, float]] = {}
        root = ET.parse(self.cfg.urdf_path).getroot()
        for joint in root.findall("joint"):
            limit = joint.find("limit")
            if limit is None:
                continue
            lower, upper = limit.get("lower"), limit.get("upper")
            if lower is not None and upper is not None:
                bounds[joint.attrib["name"]] = (float(lower), float(upper))
        out = {}
        for arm in ARMS:
            names = self._arm_joints[self._arm_slices[arm]]
            out[arm] = np.asarray([bounds.get(name, (-np.pi, np.pi)) for name in names], dtype=float)
        return out

    # -- observation -------------------------------------------------------
    def _measured_arm_angles(self) -> np.ndarray | None:
        snapshot = self.publisher.state_snapshot()
        measured = snapshot.get("arm_q_meas")
        if measured:
            return np.asarray(measured, dtype=float)
        motor_q = snapshot.get("motor_q")
        if motor_q and len(motor_q) > 28:
            return np.asarray([float(motor_q[i]) for i in range(15, 29)], dtype=float)
        return None

    def _measured_grippers(self) -> list[float]:
        """Measured gripper opening per arm, normalised to 0=closed, 1=open."""
        snapshot = self.publisher.state_snapshot()
        measured = snapshot.get("gripper_meas") or [None, None]
        span = self.cfg.r1a7_gripper_max - self.cfg.r1a7_gripper_min
        out = []
        for value in measured[:2]:
            if value is None or abs(span) < 1e-9:
                out.append(0.0)
            else:
                out.append(
                    float(np.clip((float(value) - self.cfg.r1a7_gripper_min) / span, 0.0, 1.0))
                )
        return out

    def _sync_model_to_measured(self) -> None:
        """Push measured joint angles into the IK model.

        Observations and the next IK solve must both start from where the robot
        actually is, not from the last commanded pose, or the two drift apart.
        """
        measured = self._measured_arm_angles()
        if measured is not None and measured.size == len(self._arm_joints):
            self.ik.set_joint_positions(dict(zip(self._arm_joints, measured)))

    def state(self) -> dict:
        """Proprioception in the exact keys `astra_policy.STATE_KEYS` reads."""
        self._sync_model_to_measured()
        measured = self._measured_arm_angles()
        if measured is None:
            measured = np.zeros(len(self._arm_joints))
        grippers = self._measured_grippers()
        poses = self.geometry.ee_poses()
        out: dict[str, list[float]] = {}
        for index, arm in enumerate(ARMS):
            out[f"{arm}_arm_joint_state"] = measured[index * 7 : (index + 1) * 7].tolist()
            out[f"{arm}_ee_joint_state"] = [grippers[index]]
            pose = poses.get(arm)
            if pose is None:
                raise RuntimeError(f"No forward kinematics for the {arm} end effector")
            out[f"{arm}_ee_pose"] = list(pose)
        return out

    def get_obs(self) -> dict:
        """One observation: three RGB views, proprioception and the instruction."""
        frames, _frame_id = self.images.get()
        vision: dict[str, dict] = {}
        for camera, slot in CAMERA_SLOTS.items():
            # rollout.py retries on blank frames, so hand back an all-zero
            # image rather than raising when the stream has not arrived.
            pixels = (
                _policy_camera_frame(camera, frames[slot])
                if frames is not None and slot < len(frames)
                else np.zeros((480, 640, 3), dtype=np.uint8)
            )
            vision[camera] = {"color": pixels}
        return {"vision": vision, "state": self.state(), "instruction": self.instruction}

    def monitor_frames(self) -> dict[str, np.ndarray]:
        """Latest camera frames for the monitor, independent of the control loop."""
        frames, _frame_id = self.images.get()
        out: dict[str, np.ndarray] = {}
        for camera, slot in CAMERA_SLOTS.items():
            if frames is not None and slot < len(frames):
                out[camera] = _policy_camera_frame(camera, frames[slot])
        return out

    def head_calibration(self, observation: dict) -> dict:
        """Deprecated: `rollout.py` calls `capture_head_calibration(env, obs)`.

        Kept as a direct entry point for tests; the live path goes through
        `camera_manager` and `robot_manager` so the upstream function runs
        unmodified.
        """
        from roboicl.policy.head_calibration_capture import capture_head_calibration

        return capture_head_calibration(self, observation)

    @property
    def camera_manager(self) -> "_CameraManager":
        """Cameras shaped as the head-calibration capture expects.

        Built per access because the extrinsics follow the head joints: a pinned
        head makes them constant, but nothing here assumes that.
        """
        frames, _frame_id = self.images.get()
        image_sizes = {}
        if frames is not None and len(frames) > CAMERA_SLOTS["cam_head"]:
            head = _policy_camera_frame("cam_head", frames[CAMERA_SLOTS["cam_head"]])
            image_sizes["cam_head"] = (int(head.shape[1]), int(head.shape[0]))
        geometry = self.geometry.camera_geometry(image_sizes)
        record = geometry.get("cam_head")
        if record is None:
            raise RuntimeError(
                "No head camera calibration. Prepare the device-specific "
                "robot_teleop/config/camera_calibration.yaml before rollout."
            )
        width, height = record.get("image_size", [640, 480])
        cameras = {
            "cam_head": _CalibratedCamera(
                record["intrinsic_matrix"], record["camera_to_world"], (int(width), int(height))
            )
        }
        return _CameraManager(cameras)

    @property
    def robot_manager(self) -> "_RobotManagerShim":
        return _RobotManagerShim()

    # -- action ------------------------------------------------------------
    def take_action(self, action: dict) -> None:
        """Execute one validated Cartesian target through the R1-A7 publisher."""
        if self._steps >= self.step_lim:
            return
        targets = []
        gripper_urdf: list[float] = []
        for arm in ARMS:
            pose = action.get(f"{arm}_ee_pose")
            if pose is None:
                raise ValueError(
                    f"R1-A7 execution needs {arm}_ee_pose; joint-mode actions are not wired up"
                )
            pose = np.asarray(pose, dtype=float).reshape(7)
            targets.append((self._ee_frames[arm], pose[:3], _quat_wxyz_to_xyzw(pose[3:])))
            grip = float(np.asarray(action[f"{arm}_ee_joint_state"], dtype=float).reshape(-1)[0])
            gripper_urdf.extend(self._gripper_command(arm, grip))

        self._solve_and_publish(targets, gripper_urdf)
        self._steps += 1
        self._last_action = {k: np.asarray(v).tolist() for k, v in action.items()}
        if self.monitor is not None:
            self.monitor.action_executed(self._steps, self._action_in_chunk)
            self._action_in_chunk += 1
        if self.settle_s > 0.0:
            time.sleep(self.settle_s)
        else:
            time.sleep(1.0 / max(self.control_hz, 1e-3))

    def _gripper_command(self, arm: str, normalised: float) -> list[float]:
        """Map a 0=closed..1=open command onto this arm's URDF joint values."""
        joints = self._gripper_joints[arm]
        if not joints:
            return []
        opening = float(np.clip(normalised, 0.0, 1.0))
        # gripper_open_position/closed_position describe the URDF range; the
        # publisher converts from there into motor units.
        value = self.cfg.gripper_closed_position + opening * (
            self.cfg.gripper_open_position - self.cfg.gripper_closed_position
        )
        return [value] * len(joints)

    def move_to_initial_pose(self) -> None:
        """Interpolate arms to configured reset joints and open both grippers."""
        snapshot = self.publisher.state_snapshot()
        measured = snapshot.get("arm_q_meas")
        if measured is None or len(measured) != len(self._arm_joints):
            raise RuntimeError("Cannot initialize pose: R1-A7 arm LowState is unavailable")

        start = np.asarray(measured, dtype=float)
        target = np.asarray(
            [float(self.cfg.reset_joint_positions.get(name, value))
             for name, value in zip(self._arm_joints, start)],
            dtype=float,
        )
        arm_speed = max(float(self.cfg.r1a7_arm_velocity_limit), 0.0)
        arm_duration = (
            float(np.max(np.abs(target - start))) / arm_speed
            if arm_speed > 1e-6 else float(self.cfg.reset_duration)
        )
        gripper_duration = (
            abs(float(self.cfg.r1a7_gripper_max) - float(self.cfg.r1a7_gripper_min))
            / max(float(self.cfg.r1a7_gripper_speed), 1e-6)
        )
        duration = max(float(self.cfg.reset_duration), arm_duration, gripper_duration)
        steps = max(1, int(np.ceil(duration * max(self.control_hz, 1.0))))
        period = 1.0 / max(self.control_hz, 1.0)
        gripper_q = np.full(
            len(self.cfg.left_gripper_joints) + len(self.cfg.right_gripper_joints),
            float(self.cfg.gripper_open_position),
            dtype=float,
        )
        base_cmd = [0.0] * (3 + len(self.cfg.base_cmd_joints))
        waist_yaw = snapshot.get("waist_yaw_meas")
        if waist_yaw is not None:
            for index, name in enumerate(self.cfg.base_cmd_joints):
                if name == self.cfg.r1a7_waist_yaw_joint:
                    base_cmd[3 + index] = float(waist_yaw)

        print(
            f"[live] init_pose=true: moving both arms to configured reset pose and "
            f"opening grippers ({duration:.2f}s)"
        )
        for index in range(1, steps + 1):
            alpha = index / steps
            arm_q = start + (target - start) * alpha
            self.publisher.publish(
                arm_q=arm_q,
                arm_tau=np.zeros(arm_q.size),
                base_cmd=base_cmd,
                gripper_q=gripper_q,
                head_q=None,
                lift_position=None,
            )
            time.sleep(period)
        self.ik.set_joint_positions(dict(zip(self._arm_joints, target)))

    def _solve_and_publish(self, targets, gripper_urdf: list[float]) -> None:
        from robot_teleop.placo_robot import IKTarget

        self._sync_model_to_measured()
        before = self.ik.joint_qpos(self._arm_joints)
        ik_targets = [IKTarget(frame, pos, quat) for frame, pos, quat in targets]
        self.ik.solve_ik(
            ik_targets,
            active_left=True,
            active_right=True,
            left_ee_frame=self._ee_frames["left"],
            right_ee_frame=self._ee_frames["right"],
            iters=self.cfg.ik_iters,
        )
        arm_q = self.ik.joint_qpos(self._arm_joints)

        if self.ik_continuity_guard is not None:
            # The upstream guard takes per-arm candidates and keeps its own
            # accepted-target history, so check and commit are separate calls.
            # Rejection must leave nothing dispatched and the IK model rolled
            # back, because the solve already moved it to a pose that will
            # never execute and the policy replans from the observed pose.
            candidates = {
                arm: {
                    "status": "Success",
                    "current": before[self._arm_slices[arm]].tolist(),
                    "target": arm_q[self._arm_slices[arm]].tolist(),
                    "joint_limits": self._joint_limits[arm].tolist(),
                    "periodic": [False] * 7,
                }
                for arm in ARMS
            }
            try:
                checks = self.ik_continuity_guard.check(candidates)
            except Exception as exc:
                self.ik.set_joint_positions(dict(zip(self._arm_joints, before)))
                record = getattr(self.ik_continuity_guard, "record", None)
                if callable(record):
                    record(self._steps, [], False)
                if self.monitor is not None:
                    event = getattr(exc, "event", {}) or {}
                    violations = event.get("violations") or []
                    detail = violations[0] if violations else {}
                    self.monitor.action_rejected(
                        f"{detail.get('error', type(exc).__name__)}"
                        + (
                            f" on {detail['arm']} arm, "
                            f"{detail.get('max_current_delta_rad', 0):.3f} rad"
                            if detail.get("arm")
                            else ""
                        )
                    )
                raise
            self.ik_continuity_guard.commit(checks)
            record = getattr(self.ik_continuity_guard, "record", None)
            if callable(record):
                record(self._steps, checks, True)

        head_q = (
            self.ik.joint_qpos(self.cfg.head_joints) if self.cfg.head_joints else None
        )
        base_cmd = [0.0] * (3 + len(self.cfg.base_cmd_joints))
        if self.cfg.base_cmd_joints:
            base_cmd[3:] = self.ik.joint_qpos(self.cfg.base_cmd_joints).tolist()
        self.publisher.publish(
            arm_q=arm_q,
            arm_tau=np.zeros(arm_q.size),
            base_cmd=base_cmd,
            gripper_q=np.asarray(gripper_urdf, dtype=float) if gripper_urdf else None,
            head_q=head_q,
            lift_position=None,
        )

    # -- episode -----------------------------------------------------------
    def is_episode_end(self) -> bool:
        # Closing the UI ends the episode: nobody is watching an arm that is
        # still being driven.
        return self._quit_requested or self._steps >= self.step_lim

    @property
    def steps(self) -> int:
        return self._steps

    def close(self) -> None:
        self.stop_frame_feed()
        close = getattr(self.publisher, "close", None)
        if callable(close):
            close()
        close_images = getattr(self.images, "close", None)
        if callable(close_images):
            close_images()
