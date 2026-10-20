from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import copy
import json

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@dataclass
class TeleopConfig:
    """Configuration for the R1-A7 GPT-6 live inference stack."""

    # ---- Inference control loop ----
    urdf_path: str = "../../assets/g1_d/g1_d.urdf"
    fps: int = 50
    ik_dt: float = 0.02
    ik_iters: int = 1
    activation_grip_threshold: float = 0.9
    max_hand_delta: float = 0.45
    max_joint_step: float = 0.12
    reset_duration: float = 1.0
    reset_joint_positions: dict[str, float] = field(default_factory=dict)
    placo_regularization_weight: float = 1e-2
    body_regularization_weight: float = 0.2
    body_joint_regularization_weights: dict[str, float] = field(default_factory=dict)
    ik_masked_joints: list[str] = field(default_factory=list)

    lz_height_joints: list[str] = field(default_factory=list)
    lz_height_scale: float = 1.0
    lz_height_offset: float = 0.0
    lz_joint_min: float = 0.0
    lz_joint_max: float = 0.21
    manual_lz_speed: float = 0.10
    # ---- Visualization / publishing ----
    floating_base: bool = True
    floating_base_child_link: str = ""
    publish_topic: str = "rt/whole_body_sdk"
    hardware_interface: str = "json"
    rnea_feedforward_joints: list[str] = field(default_factory=list)

    # ---- G1-D hardware interface (hardware_interface: g1_d_dds) ----
    g1d_motion_mode: bool = True
    g1d_lowcmd_topic: str = "rt/arm_sdk"
    g1d_lowstate_topic: str = "rt/lowstate"
    g1d_lowstate_timeout: float = 3.0
    g1d_height_cmd_topic: str = "rt/cmd_hispeed"
    g1d_move_cmd_topic: str = "rt/cmd_vel_no_limit"
    g1d_left_gripper_cmd_topic: str = "rt/dex1/left/cmd"
    g1d_right_gripper_cmd_topic: str = "rt/dex1/right/cmd"
    g1d_lift_joints: list[str] = field(default_factory=list)
    g1d_waist_yaw_joint: str = ""
    g1d_waist_pitch_joint: str = ""
    # Waist pitch channel:
    #   "topic"  -> dedicated rt/waistpitchcmd (unitree_hg MotorCmd_), state rt/waistpitchstate
    #   "lowcmd" -> rt/lowcmd motor index 13
    g1d_waist_pitch_channel: str = "topic"
    g1d_waist_pitch_cmd_topic: str = "rt/waistpitchcmd"
    g1d_waist_pitch_state_topic: str = "rt/waistpitchstate"
    # Pitch carries the whole upper body: it needs high kp plus a gravity
    # feedforward torque, otherwise it sags or will not lift at all.
    g1d_kp_waist_pitch: float = 500.0
    g1d_kd_waist_pitch: float = 3.0
    g1d_waist_pitch_tau_ff: float = -8.0
    g1d_kp_arm: float = 80.0
    g1d_kd_arm: float = 3.0
    g1d_kp_wrist: float = 40.0
    g1d_kd_wrist: float = 1.5
    g1d_kp_waist: float = 100.0
    g1d_kd_waist: float = 2.0
    g1d_kp_lock: float = 300.0
    g1d_kd_lock: float = 3.0
    g1d_gripper_min: float = 0.0
    g1d_gripper_max: float = 5.0
    g1d_gripper_kp: float = 5.0
    g1d_gripper_kd: float = 0.05
    # Arms and waist stream from a dedicated high-rate thread, with the target
    # velocity-clipped so the first frame never jumps.
    g1d_control_dt: float = 1.0 / 250.0
    g1d_arm_velocity_limit: float = 20.0
    # Seconds to ramp the arm_sdk authority weight from 0 to 1. 0 means jump
    # straight to 1. Only used in motion_mode.
    g1d_weight_ramp_time: float = 1.0
    # Built-in Dex1 grippers driven by motor index inside the same LowCmd.
    # When False, external rt/dex1/*/cmd MotorCmds_ topics are used instead.
    g1d_internal_gripper: bool = True
    g1d_left_gripper_motor_index: int = 31
    g1d_right_gripper_motor_index: int = 33
    g1d_gripper_delta: float = 0.18
    # Lift position loop: rt/cmd_hispeed Point32.z is a velocity command
    # (-1..1) and the hardware exposes no position interface, so the publisher
    # subscribes to the measured height and closes the loop with a PD term.
    g1d_height_state_topic: str = "rt/hispeed_state"
    g1d_lift_position_kp: float = 20.0
    g1d_lift_position_kd: float = 0.3
    g1d_lift_max_velocity: float = 0.8
    g1d_lift_deadband: float = 0.001
    g1d_lift_velocity_sign: float = 1.0
    g1d_lift_state_timeout: float = 0.5
    # Linear calibration between measured height (m) and the lift joint target:
    # meas = offset + scale * joint. Two lift joints in series are both driven
    # to lz_height, so the physical travel is about 2x, hence scale 2.0.
    # The offset is the measured height at the bottom and must be read once on
    # the real robot.
    g1d_lift_meas_scale: float = 2.0
    g1d_lift_meas_offset: float = 0.0

    # ---- R1-A7 hardware interface (hardware_interface: r1_a7_dds) ----
    # Fixed-base upper body: arms (motors 15..28), head (pitch 29, yaw 30) and
    # waist yaw (13) over hg LowCmd on rt/lowcmd. No lift, no mobile base.
    # Grippers use the external rt/dex1/*/cmd MotorCmds_ topics.
    r1a7_lowcmd_topic: str = "rt/lowcmd"
    r1a7_lowstate_topic: str = "rt/lowstate"
    r1a7_lowstate_timeout: float = 3.0
    r1a7_waist_yaw_joint: str = ""
    r1a7_control_dt: float = 1.0 / 250.0
    r1a7_arm_velocity_limit: float = 2.0
    r1a7_kp_arm: float = 100.0
    r1a7_kd_arm: float = 2.0
    r1a7_kp_medium: float = 100.0
    r1a7_kd_medium: float = 2.0
    r1a7_kp_wrist: float = 50.0
    r1a7_kd_wrist: float = 2.0
    r1a7_kp_head: float = 50.0
    r1a7_kd_head: float = 2.0
    # Head yaw carries little load and uses a different gear ratio, so the
    # reference implementation gives it a much softer gain.
    r1a7_kp_head_yaw: float = 10.0
    r1a7_kd_head_yaw: float = 0.1
    r1a7_kp_waist: float = 50.0
    r1a7_kd_waist: float = 3.0
    r1a7_kp_lock: float = 200.0
    r1a7_kd_lock: float = 3.0
    r1a7_left_gripper_cmd_topic: str = "rt/dex1/left/cmd"
    r1a7_right_gripper_cmd_topic: str = "rt/dex1/right/cmd"
    r1a7_left_gripper_state_topic: str = "rt/dex1/left/state"
    r1a7_right_gripper_state_topic: str = "rt/dex1/right/state"
    # New R1-A7/Dex1 path: include motors 31/33 in the same whole-robot
    # LowCmd frame as the arms, instead of the serial-to-DDS gripper service.
    r1a7_internal_gripper_lowcmd: bool = False
    r1a7_left_gripper_motor_index: int = 31
    r1a7_right_gripper_motor_index: int = 33
    r1a7_gripper_min: float = 0.0
    r1a7_gripper_max: float = 5.0
    # Gripper polarity: alpha is the closing ratio (full trigger -> 1).
    # invert=False maps closed to min and open to max. Flip this if the real
    # gripper opens when it should close.
    r1a7_gripper_invert: bool = False
    r1a7_gripper_kp: float = 5.0
    r1a7_gripper_kd: float = 0.05
    # Max gripper travel in motor units per second. Applied as a command-side
    # ramp against wall-clock time, so the open/close speed does not depend on
    # independent of how often publish() happens to be called.
    r1a7_gripper_speed: float = 9.0
    # Release the robot's built-in controller before taking over rt/lowcmd,
    # otherwise it fights us for the same joints.
    r1a7_release_motion_mode: bool = True

    robot_name: str = ""
    # Four-camera image_server ZMQ source (JPEG frames).
    image_server_host: str = "127.0.0.1"
    image_server_port: int = 5555
    camera_start_timeout: float = 10.0

    # ---- Joint name lists ----
    controlled_joints: list[str] = field(default_factory=list)
    body_joints: list[str] = field(default_factory=list)
    left_arm_joints: list[str] = field(default_factory=list)
    right_arm_joints: list[str] = field(default_factory=list)
    left_gripper_joints: list[str] = field(default_factory=list)
    right_gripper_joints: list[str] = field(default_factory=list)
    head_joints: list[str] = field(default_factory=list)
    head_yaw_joint: str = ""
    head_pitch_joint: str = ""
    head_yaw_scale: float = 1.0
    head_pitch_scale: float = 1.0
    head_joint_limits: dict[str, list[float]] = field(default_factory=dict)
    # Head tracking and fixed-head parameters used by the live IK model.
    head_tracking: bool = True
    head_fixed_pitch: float | None = None
    head_fixed_yaw: float | None = None
    gripper_open_position: float = 0.0
    gripper_closed_position: float = 0.0245
    base_cmd_joints: list[str] = field(default_factory=list)
    left_ee_body: str = "left_wrist_yaw_link"
    right_ee_body: str = "right_wrist_yaw_link"


def _read_config_file(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    return yaml.safe_load(path.read_text()) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def config_path_for_robot(robot: str, config_dir: str | Path = CONFIG_DIR) -> Path:
    name = robot.strip().replace("-", "_")
    return Path(config_dir).expanduser().resolve() / f"{name}.yaml"


def load_config(path: str | Path, common_path: str | Path | None = None) -> TeleopConfig:
    """Load config/<robot>.yaml on top of config/common.yaml."""
    path = Path(path).expanduser().resolve()
    data: dict = {}
    if common_path is not None:
        common = Path(common_path).expanduser().resolve()
        data = _read_config_file(common)
    data = _deep_merge(data, _read_config_file(path))
    cfg = TeleopConfig(**data)
    urdf = Path(cfg.urdf_path)
    if not urdf.is_absolute():
        cfg.urdf_path = str((path.parent / urdf).resolve())
    return cfg
