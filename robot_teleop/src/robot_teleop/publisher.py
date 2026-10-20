from __future__ import annotations

import json
import threading
import time

import numpy as np

from .config import TeleopConfig


class CommandPublisher:
    def publish(
        self,
        arm_q: np.ndarray,
        arm_tau: np.ndarray,
        base_cmd: list[float],
        gripper_q: np.ndarray | None = None,
        head_q: np.ndarray | None = None,
        lift_position: float | None = None,
    ) -> None:
        raise NotImplementedError

    def state_snapshot(self) -> dict:
        """Measured-state snapshot used to synchronize the live inference loop."""
        return {
            "motor_q": None,
            "motor_dq": None,
            "motor_tau": None,
            "measured_height_m": None,
            "waist_pitch_meas": None,
        }

    def close(self) -> None:
        pass


class PrintPublisher(CommandPublisher):
    def __init__(self, every: int = 25) -> None:
        self.every = max(every, 1)
        self.count = 0

    def publish(
        self,
        arm_q: np.ndarray,
        arm_tau: np.ndarray,
        base_cmd: list[float],
        gripper_q: np.ndarray | None = None,
        head_q: np.ndarray | None = None,
        lift_position: float | None = None,
    ) -> None:
        self.count += 1
        if self.count % self.every == 0:
            payload = {
                "arm_q": np.round(arm_q, 4).tolist(),
                "arm_tau": np.round(arm_tau, 4).tolist(),
                "base_cmd": base_cmd,
            }
            if gripper_q is not None:
                payload["gripper_q"] = np.round(gripper_q, 4).tolist()
            if head_q is not None:
                payload["head_q"] = np.round(head_q, 4).tolist()
            if lift_position is not None:
                payload["lift_position"] = round(float(lift_position), 4)
            print(json.dumps(payload, ensure_ascii=False))


class UnitreeSDKPublisher(CommandPublisher):
    """Fallback JSON publisher for older whole_body_sdk style receivers."""

    def __init__(self, topic: str, net: str) -> None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

        ChannelFactoryInitialize(0, net)
        self._pub = ChannelPublisher(topic, String_)
        self._pub.Init()
        self._msg = String_(data="")
        self._lock = threading.Lock()

    def publish(
        self,
        arm_q: np.ndarray,
        arm_tau: np.ndarray,
        base_cmd: list[float],
        gripper_q: np.ndarray | None = None,
        head_q: np.ndarray | None = None,
        lift_position: float | None = None,
    ) -> None:
        data = {
            "arm_q": np.asarray(arm_q, dtype=float).tolist(),
            "arm_tau": np.asarray(arm_tau, dtype=float).tolist(),
            "base_cmd": [float(x) for x in base_cmd],
        }
        if gripper_q is not None:
            data["gripper_q"] = np.asarray(gripper_q, dtype=float).tolist()
        if head_q is not None:
            data["head_q"] = np.asarray(head_q, dtype=float).tolist()
        if lift_position is not None:
            data["lift_position"] = float(lift_position)
        payload = json.dumps(data)
        with self._lock:
            self._msg.data = payload
            self._pub.Write(self._msg)


class G1DDSPublisher(CommandPublisher):
    """G1-D real robot publisher aligned with the proven test_g1_dex1_internal.py.

    Default path (``g1d_motion_mode: false`` + ``rt/lowcmd``) mirrors that working
    script exactly:
      - arms/waist/grippers: rt/lowcmd, unitree_hg LowCmd_ (direct PD control)
      - every ctrl cycle first copies the *measured* joint state into every
        motor_cmd (hold in place), then overrides the arm (15..28), waist (12/13)
        and the internal Dex1 grippers (31/33) with the desired targets.
      - motors index >= 29 that are not grippers are disabled (mode=0, kp=kd=0).
      - lift: rt/cmd_hispeed, geometry_msgs Point32_
      - mobile base: rt/cmd_vel_no_limit, geometry_msgs Twist_

    Optional path (``g1d_motion_mode: true`` + ``rt/arm_sdk``) keeps the arm_sdk
    weight (motor_cmd[29].q) and external rt/dex1 gripper topics, for robots that
    are driven through the loco controller in Regular mode.

    The arm/waist target is velocity-clipped against the measured joint state
    (``clip_arm_q_target`` equivalent) so the first frame never jumps.
    """

    ARM_MOTOR_INDICES = (15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)
    WAIST_YAW_INDEX = 12
    # This robot has a 2-DoF waist (yaw=12, pitch=13) and no separate roll.
    # Pitch is driven through rt/lowcmd index 13 and needs high kp plus a
    # gravity feedforward torque.
    WAIST_PITCH_INDEX = 13
    WEIGHT_MOTOR_INDEX = 29  # kNotUsedJoint0: arm_sdk authority weight [0, 1]
    NUM_MOTORS = 35

    def __init__(self, cfg: TeleopConfig, net: str) -> None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import (
            geometry_msgs_msg_dds__Point32_,
            geometry_msgs_msg_dds__Twist_,
            unitree_go_msg_dds__MotorCmd_,
            unitree_hg_msg_dds__LowCmd_,
            unitree_hg_msg_dds__MotorCmd_,
        )
        from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point32_, Twist_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
            LowCmd_ as HgLowCmd,
            LowState_ as HgLowState,
            MotorCmd_ as HgMotorCmd,
            MotorState_ as HgMotorState,
        )
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(0, net)
        self.cfg = cfg
        self._crc = CRC()
        self._last_lowstate = None
        self._state_lock = threading.Lock()
        self._motion_mode = bool(cfg.g1d_motion_mode)
        self._internal_gripper = bool(cfg.g1d_internal_gripper)
        self._left_gripper_index = int(cfg.g1d_left_gripper_motor_index)
        self._right_gripper_index = int(cfg.g1d_right_gripper_motor_index)
        self._gripper_indices = (self._left_gripper_index, self._right_gripper_index)

        # Arms (15..28) and waist yaw (12) go through the lowcmd array. Waist
        # pitch is always handled separately (see _apply_waist_pitch); its
        # channel is selected by g1d_waist_pitch_channel.
        self._arm_waist_indices = tuple(self.ARM_MOTOR_INDICES) + (self.WAIST_YAW_INDEX,)
        self._pitch_via_topic = str(cfg.g1d_waist_pitch_channel).lower() == "topic"

        self._lowcmd_pub = ChannelPublisher(cfg.g1d_lowcmd_topic, HgLowCmd)
        self._lowcmd_pub.Init()
        self._lowstate_sub = ChannelSubscriber(cfg.g1d_lowstate_topic, HgLowState)
        self._lowstate_sub.Init()
        self._height_pub = ChannelPublisher(cfg.g1d_height_cmd_topic, Point32_)
        self._height_pub.Init()
        # Measured lift height in metres, used to close the position loop.
        self._height_state_sub = ChannelSubscriber(cfg.g1d_height_state_topic, Point32_)
        self._height_state_sub.Init()
        self._measured_height = None
        self._measured_height_time = 0.0
        self._lift_velocity = 0.0
        self._lift_target = None
        self._lift_meas_target = None
        self._lift_prev_error = None
        self._lift_prev_time = None
        self._move_pub = ChannelPublisher(cfg.g1d_move_cmd_topic, Twist_)
        self._move_pub.Init()

        # Waist pitch is handled on its own. In "topic" mode it uses
        # rt/waistpitchcmd (MotorCmd_) plus rt/waistpitchstate.
        self._waist_pitch_target = None      # target angle (the IK Yaw_Joint)
        self._waist_pitch_tau = 0.0          # RNEA feedforward (only when tau_ff is 0)
        self._waist_pitch_meas = None        # measured angle
        self._waist_pitch_meas_time = 0.0
        self._waist_pitch_pub = None
        self._waist_pitch_state_sub = None
        self._waist_pitch_msg = None
        if self._pitch_via_topic:
            self._waist_pitch_pub = ChannelPublisher(cfg.g1d_waist_pitch_cmd_topic, HgMotorCmd)
            self._waist_pitch_pub.Init()
            self._waist_pitch_state_sub = ChannelSubscriber(cfg.g1d_waist_pitch_state_topic, HgMotorState)
            self._waist_pitch_state_sub.Init()
            self._waist_pitch_msg = unitree_hg_msg_dds__MotorCmd_()
            self._waist_pitch_msg.mode = 1
            self._waist_pitch_msg.dq = 0.0
            self._waist_pitch_msg.kp = cfg.g1d_kp_waist_pitch
            self._waist_pitch_msg.kd = cfg.g1d_kd_waist_pitch
        # External rt/dex1 gripper topics are only used when internal grippers are off.
        self._left_gripper_pub = None
        self._right_gripper_pub = None
        self._left_gripper_msg = None
        self._right_gripper_msg = None
        if not self._internal_gripper:
            self._left_gripper_pub = ChannelPublisher(cfg.g1d_left_gripper_cmd_topic, MotorCmds_)
            self._left_gripper_pub.Init()
            self._right_gripper_pub = ChannelPublisher(cfg.g1d_right_gripper_cmd_topic, MotorCmds_)
            self._right_gripper_pub.Init()
            self._left_gripper_msg = MotorCmds_()
            self._right_gripper_msg = MotorCmds_()
            self._left_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
            self._right_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
            for msg in (self._left_gripper_msg, self._right_gripper_msg):
                msg.cmds[0].dq = 0.0
                msg.cmds[0].tau = 0.0
                msg.cmds[0].kp = cfg.g1d_gripper_kp
                msg.cmds[0].kd = cfg.g1d_gripper_kd

        self._lowcmd_msg = unitree_hg_msg_dds__LowCmd_()
        self._height_msg = geometry_msgs_msg_dds__Point32_()
        self._move_msg = geometry_msgs_msg_dds__Twist_()
        self._publish_count = 0

        self._lowcmd_msg.mode_pr = 0

        # Shared targets updated by publish(); consumed by the ctrl thread.
        self._ctrl_lock = threading.Lock()
        self._q_target = np.zeros(len(self._arm_waist_indices), dtype=float)
        self._tau_target = np.zeros(len(self._arm_waist_indices), dtype=float)
        self._gripper_target = np.zeros(2, dtype=float)  # [left, right] in gripper motor units
        self._weight = 0.0
        self._start_time = time.monotonic()

        self._running = True
        self._sub_thread = threading.Thread(target=self._subscribe_lowstate, daemon=True)
        self._sub_thread.start()
        self._height_state_thread = threading.Thread(target=self._subscribe_height_state, daemon=True)
        self._height_state_thread.start()
        if self._pitch_via_topic:
            self._waist_pitch_state_thread = threading.Thread(target=self._subscribe_waist_pitch_state, daemon=True)
            self._waist_pitch_state_thread.start()
        self._wait_for_lowstate()
        self._init_lowcmd_defaults()
        self._apply_mode_machine()
        # Start holding the measured pose so the first frame never jumps.
        self._q_target[:] = self._current_arm_waist_q()
        if self._internal_gripper:
            self._gripper_target[0] = self._latest_motor_q(self._left_gripper_index)
            self._gripper_target[1] = self._latest_motor_q(self._right_gripper_index)
        self._start_time = time.monotonic()
        self._ctrl_thread = threading.Thread(target=self._ctrl_loop, daemon=True)
        self._ctrl_thread.start()
        print(
            f"[teleop] G1-D DDS publisher ready: lowcmd={cfg.g1d_lowcmd_topic}, "
            f"motion_mode={self._motion_mode}, "
            f"gripper={'internal@%d/%d' % self._gripper_indices if self._internal_gripper else 'external rt/dex1'}, "
            f"lift={cfg.g1d_height_cmd_topic}, base={cfg.g1d_move_cmd_topic}, "
            f"vel_limit={cfg.g1d_arm_velocity_limit} rad/s, ctrl_dt={cfg.g1d_control_dt:.4f}s"
        )

    def _subscribe_lowstate(self) -> None:
        while self._running:
            msg = self._lowstate_sub.Read()
            if msg is not None:
                with self._state_lock:
                    self._last_lowstate = msg
            time.sleep(0.002)

    def _subscribe_height_state(self) -> None:
        # rt/hispeed_state: Point32, .y is the measured lift height in metres.
        while self._running:
            msg = self._height_state_sub.Read()
            if msg is not None:
                with self._state_lock:
                    self._measured_height = float(msg.y)
                    self._measured_height_time = time.monotonic()
            time.sleep(0.005)

    def _subscribe_waist_pitch_state(self) -> None:
        # rt/waistpitchstate: MotorState_, .q is the measured waist pitch angle.
        while self._running:
            msg = self._waist_pitch_state_sub.Read()
            if msg is not None:
                with self._state_lock:
                    self._waist_pitch_meas = float(msg.q)
                    self._waist_pitch_meas_time = time.monotonic()
            time.sleep(0.005)

    def _wait_for_lowstate(self) -> None:
        deadline = time.monotonic() + max(self.cfg.g1d_lowstate_timeout, 0.0)
        while time.monotonic() < deadline:
            with self._state_lock:
                if self._last_lowstate is not None:
                    return
            time.sleep(0.02)
        print(
            "[teleop] warning: no rt/lowstate received before startup; "
            "mode_machine=0 and velocity clipping disabled (arm_sdk may reject commands)"
        )

    def _latest_motor_q(self, index: int) -> float:
        with self._state_lock:
            state = self._last_lowstate
        if state is None:
            return 0.0
        return float(state.motor_state[index].q)

    def _current_arm_waist_q(self) -> np.ndarray:
        with self._state_lock:
            state = self._last_lowstate
        if state is None:
            return self._q_target.copy()
        return np.asarray([float(state.motor_state[i].q) for i in self._arm_waist_indices], dtype=float)

    def _init_lowcmd_defaults(self) -> None:
        arm_indices = set(self.ARM_MOTOR_INDICES)
        gripper_indices = set(self._gripper_indices) if self._internal_gripper else set()
        for i in range(self.NUM_MOTORS):
            cmd = self._lowcmd_msg.motor_cmd[i]
            cmd.q = self._latest_motor_q(i)
            cmd.dq = 0.0
            cmd.tau = 0.0
            if i in gripper_indices:
                # Internal Dex1 grippers driven inside this LowCmd (test_g1_dex1_internal.py).
                cmd.mode = 1
                cmd.kp = self.cfg.g1d_gripper_kp
                cmd.kd = self.cfg.g1d_gripper_kd
            elif i >= self.WEIGHT_MOTOR_INDEX:
                # Motors 29..34 (kNotUsedJoint / non-gripper): disabled in direct mode.
                # In arm_sdk motion mode motor 29 carries the authority weight.
                if self._motion_mode and i == self.WEIGHT_MOTOR_INDEX:
                    cmd.mode = 1
                    cmd.kp = 0.0
                    cmd.kd = 0.0
                else:
                    cmd.mode = 0
                    cmd.kp = 0.0
                    cmd.kd = 0.0
            elif i in arm_indices:
                cmd.mode = 1
                is_wrist = i in {19, 20, 21, 26, 27, 28}
                cmd.kp = self.cfg.g1d_kp_wrist if is_wrist else self.cfg.g1d_kp_arm
                cmd.kd = self.cfg.g1d_kd_wrist if is_wrist else self.cfg.g1d_kd_arm
            elif i == self.WAIST_YAW_INDEX:
                cmd.mode = 1
                cmd.kp = self.cfg.g1d_kp_waist
                cmd.kd = self.cfg.g1d_kd_waist
            elif i == self.WAIST_PITCH_INDEX:
                # Pitch is handled separately (see _apply_waist_pitch).
                #   topic mode:  dedicated rt/waistpitchcmd; index 13 here is
                #                disabled (mode=0) so the two do not fight.
                #   lowcmd mode: high kp plus gravity feedforward, written to
                #                motor_cmd[13] every cycle.
                if self._pitch_via_topic:
                    cmd.mode = 0
                    cmd.kp = 0.0
                    cmd.kd = 0.0
                else:
                    cmd.mode = 1
                    cmd.kp = self.cfg.g1d_kp_waist_pitch
                    cmd.kd = self.cfg.g1d_kd_waist_pitch
            else:
                cmd.mode = 1
                cmd.kp = self.cfg.g1d_kp_lock
                cmd.kd = self.cfg.g1d_kd_lock

    def _apply_mode_machine(self) -> None:
        with self._state_lock:
            state = self._last_lowstate
        if state is not None and hasattr(state, "mode_machine"):
            self._lowcmd_msg.mode_machine = state.mode_machine

    def _current_weight(self) -> float:
        if not self.cfg.g1d_motion_mode:
            return 0.0
        ramp = max(self.cfg.g1d_weight_ramp_time, 0.0)
        if ramp <= 0.0:
            return 1.0
        return float(np.clip((time.monotonic() - self._start_time) / ramp, 0.0, 1.0))

    def _clip_arm_waist_target(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        limit = self.cfg.g1d_arm_velocity_limit
        if limit <= 0.0:
            return target
        delta = target - current
        max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
        allowed = limit * self.cfg.g1d_control_dt
        if allowed <= 0.0:
            return target
        scale = max_delta / allowed
        return current + delta / max(scale, 1.0)

    def _ctrl_loop(self) -> None:
        dt = max(self.cfg.g1d_control_dt, 1e-3)
        # Skip the hold step for arms, waist yaw, grippers and waist pitch.
        hold_skip = set(self._arm_waist_indices)
        hold_skip |= set(self._gripper_indices) if self._internal_gripper else set()
        hold_skip.add(self.WAIST_PITCH_INDEX)
        while self._running:
            start = time.monotonic()
            with self._state_lock:
                state = self._last_lowstate

            with self._ctrl_lock:
                q_target = self._q_target.copy()
                tau_target = self._tau_target.copy()
                gripper_target = self._gripper_target.copy()

            # 1) Hold every joint at its measured position (like the working script),
            #    so nothing but the arms/waist/grippers we command below drifts.
            if state is not None:
                for i in range(self.NUM_MOTORS):
                    if i in hold_skip:
                        continue
                    cmd = self._lowcmd_msg.motor_cmd[i]
                    cmd.q = float(state.motor_state[i].q)
                    cmd.dq = 0.0
                    cmd.tau = 0.0

            # 2) Override arm (15..28) + waist yaw(12) with velocity-clipped targets.
            current = self._current_arm_waist_q()
            clipped = self._clip_arm_waist_target(q_target, current)
            for idx, motor_index in enumerate(self._arm_waist_indices):
                cmd = self._lowcmd_msg.motor_cmd[motor_index]
                cmd.q = float(clipped[idx])
                cmd.dq = 0.0
                cmd.tau = float(tau_target[idx])

            # 2b) Waist pitch closes its own loop (dedicated topic or index 13).
            self._apply_waist_pitch(dt, state)

            # 3) Override internal Dex1 grippers (31/33), rate-limited per cycle.
            if self._internal_gripper and state is not None:
                delta = max(self.cfg.g1d_gripper_delta, 0.0)
                for slot, motor_index in enumerate(self._gripper_indices):
                    cur = float(state.motor_state[motor_index].q)
                    tgt = float(gripper_target[slot])
                    if delta > 0.0:
                        tgt = cur + float(np.clip(tgt - cur, -delta, delta))
                    cmd = self._lowcmd_msg.motor_cmd[motor_index]
                    cmd.mode = 1
                    cmd.q = tgt
                    cmd.dq = 0.0
                    cmd.tau = 0.0
                    cmd.kp = self.cfg.g1d_gripper_kp
                    cmd.kd = self.cfg.g1d_gripper_kd

            self._apply_mode_machine()
            self._weight = self._current_weight()
            if self._motion_mode:
                self._lowcmd_msg.motor_cmd[self.WEIGHT_MOTOR_INDEX].q = self._weight

            self._lowcmd_msg.crc = self._crc.Crc(self._lowcmd_msg)
            self._lowcmd_pub.Write(self._lowcmd_msg)

            elapsed = time.monotonic() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def _apply_waist_pitch(self, dt: float, state) -> None:
        """Close the waist pitch loop.

        Pitch carries the whole upper body, so it needs high kp plus a gravity
        feedforward torque, otherwise it will not move or it sags.

        - topic mode:  target and torque go out on rt/waistpitchcmd (MotorCmd_);
                       the measured angle comes from rt/waistpitchstate.
        - lowcmd mode: motor_cmd[13] is written directly and ships with this
                       cycle's LowCmd; the measured angle comes from lowstate.
        """
        with self._ctrl_lock:
            target = self._waist_pitch_target
            rnea_tau = float(self._waist_pitch_tau)
        if target is None:
            return

        # Measured angle: dedicated state topic, or lowstate motor 13.
        if self._pitch_via_topic:
            with self._state_lock:
                measured = self._waist_pitch_meas
        else:
            measured = float(state.motor_state[self.WAIST_PITCH_INDEX].q) if state is not None else None

        # Velocity clamp (rad/s) so the first frame never jumps.
        cmd_q = float(target)
        limit = self.cfg.g1d_arm_velocity_limit
        if measured is not None and limit > 0.0:
            allowed = limit * dt
            delta = cmd_q - float(measured)
            if allowed > 0.0 and abs(delta) > allowed:
                cmd_q = float(measured) + np.sign(delta) * allowed

        # Feedforward torque: the constant gravity term wins; fall back to RNEA when it is 0.
        ff = float(self.cfg.g1d_waist_pitch_tau_ff)
        tau = ff if ff != 0.0 else rnea_tau

        if self._pitch_via_topic:
            self._waist_pitch_msg.mode = 1
            self._waist_pitch_msg.q = cmd_q
            self._waist_pitch_msg.dq = 0.0
            self._waist_pitch_msg.tau = tau
            self._waist_pitch_msg.kp = self.cfg.g1d_kp_waist_pitch
            self._waist_pitch_msg.kd = self.cfg.g1d_kd_waist_pitch
            self._waist_pitch_pub.Write(self._waist_pitch_msg)
        else:
            cmd = self._lowcmd_msg.motor_cmd[self.WAIST_PITCH_INDEX]
            cmd.mode = 1
            cmd.q = cmd_q
            cmd.dq = 0.0
            cmd.tau = tau
            cmd.kp = self.cfg.g1d_kp_waist_pitch
            cmd.kd = self.cfg.g1d_kd_waist_pitch

    def _clip_gripper_q(self, q: float) -> float:
        return float(np.clip(q, self.cfg.g1d_gripper_min, self.cfg.g1d_gripper_max))

    def _update_arm_waist_target(self, arm_q: np.ndarray, arm_tau: np.ndarray, base_cmd: list[float]) -> None:
        q = np.asarray(arm_q, dtype=float).reshape(-1)
        tau = np.asarray(arm_tau, dtype=float).reshape(-1)
        if q.size != len(self.ARM_MOTOR_INDICES):
            raise ValueError(f"G1-D arm_q must have 14 values, got {q.size}")

        target = np.empty(len(self._arm_waist_indices), dtype=float)
        tau_out = np.zeros(len(self._arm_waist_indices), dtype=float)
        target[: len(self.ARM_MOTOR_INDICES)] = q
        tau_out[: len(self.ARM_MOTOR_INDICES)] = tau[: len(self.ARM_MOTOR_INDICES)]

        # Waist yaw uses the lowcmd array (index 12); pitch is stored separately.
        body = {name: float(value) for name, value in zip(self.cfg.base_cmd_joints, base_cmd[3:])}
        yaw_slot = len(self.ARM_MOTOR_INDICES)
        target[yaw_slot] = body.get(self.cfg.g1d_waist_yaw_joint, self._current_arm_waist_q()[yaw_slot]) \
            if self.cfg.g1d_waist_yaw_joint else self._q_target[yaw_slot]

        # Waist pitch target plus its RNEA feedforward (15th entry), stored for _apply_waist_pitch.
        pitch_target = None
        pitch_tau = 0.0
        if self.cfg.g1d_waist_pitch_joint and self.cfg.g1d_waist_pitch_joint in body:
            pitch_target = body[self.cfg.g1d_waist_pitch_joint]
            if tau.size > len(self.ARM_MOTOR_INDICES):
                pitch_tau = float(tau[len(self.ARM_MOTOR_INDICES)])

        with self._ctrl_lock:
            self._q_target[:] = target
            self._tau_target[:] = tau_out
            if pitch_target is not None:
                self._waist_pitch_target = pitch_target
                self._waist_pitch_tau = pitch_tau

    def _lift_velocity_for_target(self, target: float | None) -> float:
        """Convert a lift joint target into a velocity command in -1..1.

        rt/cmd_hispeed Point32.z is a velocity and the hardware exposes no
        position interface, so this runs a PD loop on the measured height from
        rt/hispeed_state. Returns 0 when the measurement is missing or stale,
        which is the safe choice: never move blind.
        """
        if target is None:
            self._lift_prev_error = None
            self._lift_prev_time = None
            return 0.0
        with self._state_lock:
            measured = self._measured_height
            meas_time = self._measured_height_time
        if measured is None or (time.monotonic() - meas_time) > self.cfg.g1d_lift_state_timeout:
            self._lift_prev_error = None
            self._lift_prev_time = None
            return 0.0
        # The target is a lift joint value while the feedback is in metres, with
        # a different zero. Map the target into metres first
        # (meas_target = offset + scale * joint) and run the PD loop there, so
        # the physical height it settles at matches MuJoCo.
        meas_target = self.cfg.g1d_lift_meas_offset + self.cfg.g1d_lift_meas_scale * float(target)
        self._lift_meas_target = meas_target
        error = meas_target - measured
        # Derivative term: d_error = (error - prev_error) / dt
        now = time.monotonic()
        d_error = 0.0
        if self._lift_prev_error is not None and self._lift_prev_time is not None:
            dt = now - self._lift_prev_time
            if dt > 0.0:
                d_error = (error - self._lift_prev_error) / dt
        self._lift_prev_error = error
        self._lift_prev_time = now
        if abs(error) < self.cfg.g1d_lift_deadband:
            return 0.0
        vel = (error * self.cfg.g1d_lift_position_kp + d_error * self.cfg.g1d_lift_position_kd)
        vel *= self.cfg.g1d_lift_velocity_sign
        max_v = abs(self.cfg.g1d_lift_max_velocity)
        return float(np.clip(vel, -max_v, max_v))

    def _write_mobile_lift(self, base_cmd: list[float], lift_position: float | None) -> None:
        vx = float(base_cmd[0]) if len(base_cmd) > 0 else 0.0
        vyaw = float(base_cmd[2]) if len(base_cmd) > 2 else 0.0
        self._lift_velocity = self._lift_velocity_for_target(lift_position)
        self._lift_target = lift_position
        self._height_msg.z = self._lift_velocity
        self._move_msg.linear.x = vx
        self._move_msg.linear.y = 0.0
        self._move_msg.linear.z = 0.0
        self._move_msg.angular.x = 0.0
        self._move_msg.angular.y = 0.0
        self._move_msg.angular.z = vyaw
        self._height_pub.Write(self._height_msg)
        self._move_pub.Write(self._move_msg)

    def _write_grippers(self, gripper_q: np.ndarray | None) -> None:
        if gripper_q is None:
            return
        q = np.asarray(gripper_q, dtype=float).reshape(-1)
        if q.size == 0:
            return
        left_urdf = float(np.mean(q[: max(1, len(self.cfg.left_gripper_joints))]))
        right_start = len(self.cfg.left_gripper_joints)
        right_values = q[right_start : right_start + max(1, len(self.cfg.right_gripper_joints))]
        right_urdf = float(np.mean(right_values)) if right_values.size else left_urdf
        denom = max(self.cfg.gripper_closed_position - self.cfg.gripper_open_position, 1e-6)
        left_alpha = np.clip((left_urdf - self.cfg.gripper_open_position) / denom, 0.0, 1.0)
        right_alpha = np.clip((right_urdf - self.cfg.gripper_open_position) / denom, 0.0, 1.0)
        span = self.cfg.g1d_gripper_max - self.cfg.g1d_gripper_min
        # On hardware 0 is closed and max is open, which is the opposite of the
        # trigger (squeeze to close). alpha is the closing ratio, so the motor
        # command uses (1 - alpha): released maps to open, full squeeze to closed.
        left_cmd = self._clip_gripper_q(self.cfg.g1d_gripper_min + (1.0 - left_alpha) * span)
        right_cmd = self._clip_gripper_q(self.cfg.g1d_gripper_min + (1.0 - right_alpha) * span)

        if self._internal_gripper:
            # Consumed by the ctrl thread which writes motor_cmd[31/33] in the LowCmd.
            with self._ctrl_lock:
                self._gripper_target[0] = left_cmd
                self._gripper_target[1] = right_cmd
            return

        self._left_gripper_msg.cmds[0].q = left_cmd
        self._right_gripper_msg.cmds[0].q = right_cmd
        self._left_gripper_pub.Write(self._left_gripper_msg)
        self._right_gripper_pub.Write(self._right_gripper_msg)

    def publish(
        self,
        arm_q: np.ndarray,
        arm_tau: np.ndarray,
        base_cmd: list[float],
        gripper_q: np.ndarray | None = None,
        head_q: np.ndarray | None = None,
        lift_position: float | None = None,
    ) -> None:
        # Arm/waist LowCmd is streamed by the ctrl thread; here we only update the target.
        self._update_arm_waist_target(arm_q, arm_tau, base_cmd)
        self._write_mobile_lift(base_cmd, lift_position)
        self._write_grippers(gripper_q)
        self._publish_count += 1
        if self._publish_count % 50 == 0:
            current = self._current_arm_waist_q()
            err = float(np.linalg.norm(self._q_target[: len(self.ARM_MOTOR_INDICES)] - current[: len(self.ARM_MOTOR_INDICES)]))
            print(
                "[g1d_dds] "
                f"topic={self.cfg.g1d_lowcmd_topic} "
                f"lowstate={'ok' if self._last_lowstate is not None else 'MISSING'} "
                f"mode_machine={getattr(self._lowcmd_msg, 'mode_machine', None)} "
                f"weight={self._weight:.2f} "
                f"arm_target_norm={float(np.linalg.norm(self._q_target[: len(self.ARM_MOTOR_INDICES)])):.4f} "
                f"arm_track_err={err:.4f} "
                f"tau_norm={float(np.linalg.norm(arm_tau)):.4f} "
                f"lift_meas={'%.4f' % self._measured_height if self._measured_height is not None else 'MISSING'} "
                f"lift_target={'%.4f' % self._lift_target if self._lift_target is not None else 'None'} "
                f"lift_meas_target={'%.4f' % self._lift_meas_target if self._lift_meas_target is not None else 'None'} "
                f"lift_vel={self._lift_velocity:.3f} "
                f"wpitch_ch={'topic' if self._pitch_via_topic else 'lowcmd'} "
                f"wpitch_target={'%.4f' % self._waist_pitch_target if self._waist_pitch_target is not None else 'None'} "
                f"wpitch_meas={'%.4f' % self._waist_pitch_meas if self._waist_pitch_meas is not None else 'MISSING'}"
            )

    def state_snapshot(self) -> dict:
        """Snapshot: 35 motor q/dq/tau_est, measured lift height (m), waist pitch angle."""
        with self._state_lock:
            st = self._last_lowstate
            mh = self._measured_height
            wp = self._waist_pitch_meas
        out = {
            "motor_q": None,
            "motor_dq": None,
            "motor_tau": None,
            "measured_height_m": float(mh) if mh is not None else None,
            "waist_pitch_meas": float(wp) if wp is not None else None,
        }
        if st is not None:
            out["motor_q"] = [float(st.motor_state[i].q) for i in range(self.NUM_MOTORS)]
            out["motor_dq"] = [float(st.motor_state[i].dq) for i in range(self.NUM_MOTORS)]
            out["motor_tau"] = [float(st.motor_state[i].tau_est) for i in range(self.NUM_MOTORS)]
        return out

    def close(self) -> None:
        # Both the lift (rt/cmd_hispeed.z) and the base (rt/cmd_vel_no_limit)
        # take velocity commands, so they must be explicitly zeroed and repeated
        # a few times on exit. Otherwise the robot keeps drifting on the last
        # command it received.
        try:
            self._height_msg.z = 0.0
            self._move_msg.linear.x = 0.0
            self._move_msg.linear.y = 0.0
            self._move_msg.linear.z = 0.0
            self._move_msg.angular.x = 0.0
            self._move_msg.angular.y = 0.0
            self._move_msg.angular.z = 0.0
            for _ in range(5):
                self._height_pub.Write(self._height_msg)
                self._move_pub.Write(self._move_msg)
                time.sleep(0.01)
            self._lift_velocity = 0.0
        except Exception as exc:
            print(f"[teleop] warning: failed to zero lift/base velocity on close: {exc}")
        self._running = False



class R1A7DDSPublisher(CommandPublisher):
    """R1-A7 fixed-base upper-body publisher.

    Drives hg LowCmd on rt/lowcmd (mode_pr=0, mode_machine taken from lowstate):
      - arms:      motors 15..28 (14 joints)
      - head:      pitch=29, yaw=30
      - waist yaw: motor 13 (waist roll 12 and pitch 14 are unused here)

    Every control cycle first writes the measured angle into every motor command
    (hold in place), then overwrites the arms, head and waist yaw with the
    ramped targets; everything else stays locked. Grippers use the external
    rt/dex1/{left,right}/cmd topics (MotorCmds_), rate-limited against their
    measured state. No lift and no mobile base.
    """

    ARM_MOTOR_INDICES = (15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)
    WAIST_YAW_INDEX = 13
    HEAD_PITCH_INDEX = 29
    HEAD_YAW_INDEX = 30
    WRIST_INDICES = frozenset({19, 20, 21, 26, 27, 28})
    MEDIUM_ARM_INDICES = frozenset({17, 18, 24, 25})  # shoulder yaw, elbow
    NUM_MOTORS = 35

    def __init__(self, cfg: TeleopConfig, net: str) -> None:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import (
            unitree_go_msg_dds__MotorCmd_,
            unitree_hg_msg_dds__LowCmd_,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
            LowCmd_ as HgLowCmd,
            LowState_ as HgLowState,
        )
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(0, net)
        self.cfg = cfg
        self._internal_gripper = bool(cfg.r1a7_internal_gripper_lowcmd)
        self._gripper_indices = (
            cfg.r1a7_left_gripper_motor_index,
            cfg.r1a7_right_gripper_motor_index,
        )
        if self._internal_gripper and set(self._gripper_indices) != {31, 33}:
            raise ValueError("R1-A7 internal Dex1 indices must be 31 and 33")
        self._crc = CRC()
        self._last_lowstate = None
        self._state_lock = threading.Lock()
        if cfg.r1a7_release_motion_mode:
            self._release_motion_mode()

        # Head joint name to motor index, matched against the configured names.
        self._head_index_by_joint: dict[str, int] = {}
        if cfg.head_yaw_joint:
            self._head_index_by_joint[cfg.head_yaw_joint] = self.HEAD_YAW_INDEX
        if cfg.head_pitch_joint:
            self._head_index_by_joint[cfg.head_pitch_joint] = self.HEAD_PITCH_INDEX
        # Target vector layout: 14 arm joints, optional waist yaw, then head joints.
        self._head_indices = [
            self._head_index_by_joint[name]
            for name in cfg.head_joints
            if name in self._head_index_by_joint
        ]
        self._waist_yaw_enabled = bool(cfg.r1a7_waist_yaw_joint)
        self._controlled_indices = list(self.ARM_MOTOR_INDICES)
        if self._waist_yaw_enabled:
            self._controlled_indices.append(self.WAIST_YAW_INDEX)
        self._controlled_indices += self._head_indices
        self._n_ctrl = len(self._controlled_indices)

        self._lowcmd_pub = ChannelPublisher(cfg.r1a7_lowcmd_topic, HgLowCmd)
        self._lowcmd_pub.Init()
        self._lowstate_sub = ChannelSubscriber(cfg.r1a7_lowstate_topic, HgLowState)
        self._lowstate_sub.Init()

        self._left_gripper_pub = None
        self._right_gripper_pub = None
        self._left_gripper_state_sub = None
        self._right_gripper_state_sub = None
        self._left_gripper_msg = None
        self._right_gripper_msg = None
        if not self._internal_gripper:
            # Legacy R1-A7 path: external serial-to-DDS gripper service.
            self._left_gripper_pub = ChannelPublisher(cfg.r1a7_left_gripper_cmd_topic, MotorCmds_)
            self._left_gripper_pub.Init()
            self._right_gripper_pub = ChannelPublisher(cfg.r1a7_right_gripper_cmd_topic, MotorCmds_)
            self._right_gripper_pub.Init()
            self._left_gripper_state_sub = ChannelSubscriber(cfg.r1a7_left_gripper_state_topic, MotorStates_)
            self._left_gripper_state_sub.Init()
            self._right_gripper_state_sub = ChannelSubscriber(cfg.r1a7_right_gripper_state_topic, MotorStates_)
            self._right_gripper_state_sub.Init()
            self._left_gripper_msg = MotorCmds_()
            self._right_gripper_msg = MotorCmds_()
            self._left_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
            self._right_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
            for msg in (self._left_gripper_msg, self._right_gripper_msg):
                msg.cmds[0].dq = 0.0
                msg.cmds[0].tau = 0.0
                msg.cmds[0].kp = cfg.r1a7_gripper_kp
                msg.cmds[0].kd = cfg.r1a7_gripper_kd
        self._gripper_meas = [None, None]        # [left, right] measured motor angle
        self._gripper_meas_time = 0.0
        self._gripper_cmd_sent = [None, None]    # [left, right] last commanded motor angle
        self._gripper_cmd_time = None            # wall-clock stamp of that command

        self._lowcmd_msg = unitree_hg_msg_dds__LowCmd_()
        self._lowcmd_msg.mode_pr = 0
        self._publish_count = 0

        # Shared targets written by publish() and consumed by the ctrl thread.
        self._ctrl_lock = threading.Lock()
        self._q_target = np.zeros(self._n_ctrl, dtype=float)
        self._tau_target = np.zeros(self._n_ctrl, dtype=float)
        # Command-side ramp state (the last q actually sent), advanced each cycle.
        self._q_command = np.zeros(self._n_ctrl, dtype=float)

        self._running = True
        self._sub_thread = threading.Thread(target=self._subscribe_lowstate, daemon=True)
        self._sub_thread.start()
        self._gripper_state_thread = None
        if not self._internal_gripper:
            self._gripper_state_thread = threading.Thread(target=self._subscribe_gripper_state, daemon=True)
            self._gripper_state_thread.start()
        self._wait_for_lowstate()
        if self._internal_gripper:
            with self._state_lock:
                state = self._last_lowstate
                if state is not None:
                    self._gripper_meas = [float(state.motor_state[i].q) for i in self._gripper_indices]
                    self._gripper_cmd_sent = list(self._gripper_meas)
        self._init_lowcmd_defaults()
        self._apply_mode_machine()
        # Start by holding the measured pose so the first frame never jumps. The
        # ramp must start from the measurement too, otherwise the first cycle
        # climbs from 0 and yanks the joints toward zero.
        self._q_target[:] = self._current_ctrl_q()
        self._q_command[:] = self._q_target
        self._ctrl_thread = threading.Thread(target=self._ctrl_loop, daemon=True)
        self._ctrl_thread.start()
        print(
            f"[publisher] R1-A7 DDS ready: lowcmd={cfg.r1a7_lowcmd_topic}, "
            f"arms=15..28, waist_yaw={'13' if self._waist_yaw_enabled else 'locked'}, "
            f"head=pitch29/yaw30, "
            f"grippers={'LowCmd@31/33' if self._internal_gripper else 'external DDS topics'}, "
            f"vel_limit={cfg.r1a7_arm_velocity_limit} rad/s, ctrl_dt={cfg.r1a7_control_dt:.4f}s"
        )

    # -- Subscriber threads ------------------------------------------------
    def _release_motion_mode(self) -> None:
        """Release the robot's built-in controller before taking over rt/lowcmd.

        Without this the internal controller fights us for the same joints,
        which shows up as shaking, springing back, or simply not following.
        Retries up to three times; an empty name from CheckMode means released.
        Degrades safely when the SDK is missing so startup is never blocked.
        """
        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

            msc = MotionSwitcherClient()
            msc.SetTimeout(5.0)
            msc.Init()
            for _ in range(3):
                _, result = msc.CheckMode()
                name = result.get("name", "") if result else ""
                if not name:
                    print("[teleop] R1-A7 motion mode released, taking over rt/lowcmd")
                    return
                print(f"[teleop] R1-A7 motion mode={name}, requesting ReleaseMode()")
                msc.ReleaseMode()
                time.sleep(1.0)
            print(
                "[teleop] warning: R1-A7 motion mode still set after release; "
                "the built-in controller may still be fighting for the joints"
            )
        except Exception as exc:
            print(f"[teleop] warning: R1-A7 motion switcher release skipped/failed: {exc}")

    def _subscribe_lowstate(self) -> None:
        while self._running:
            msg = self._lowstate_sub.Read()
            if msg is not None:
                with self._state_lock:
                    self._last_lowstate = msg
                    if self._internal_gripper:
                        self._gripper_meas = [
                            float(msg.motor_state[i].q) for i in self._gripper_indices
                        ]
            time.sleep(0.002)

    def _subscribe_gripper_state(self) -> None:
        while self._running:
            left = self._left_gripper_state_sub.Read()
            right = self._right_gripper_state_sub.Read()
            if left is not None and right is not None and left.states and right.states:
                with self._state_lock:
                    self._gripper_meas[0] = float(left.states[0].q)
                    self._gripper_meas[1] = float(right.states[0].q)
                    self._gripper_meas_time = time.monotonic()
            time.sleep(0.002)

    def _wait_for_lowstate(self) -> None:
        deadline = time.monotonic() + max(self.cfg.r1a7_lowstate_timeout, 0.0)
        while time.monotonic() < deadline:
            with self._state_lock:
                if self._last_lowstate is not None:
                    return
            time.sleep(0.02)
        print(
            "[teleop] warning: no rt/lowstate from R1-A7 before timeout; "
            "mode_machine=0 and the robot may reject commands"
        )

    # -- State access ------------------------------------------------------
    def _latest_motor_q(self, index: int) -> float:
        with self._state_lock:
            state = self._last_lowstate
        if state is None:
            return 0.0
        return float(state.motor_state[index].q)

    def _current_ctrl_q(self) -> np.ndarray:
        with self._state_lock:
            state = self._last_lowstate
        if state is None:
            return self._q_target.copy()
        return np.asarray(
            [float(state.motor_state[i].q) for i in self._controlled_indices], dtype=float
        )

    def _kp_kd_for(self, index: int) -> tuple[float, float]:
        c = self.cfg
        if self._internal_gripper and index in self._gripper_indices:
            return c.r1a7_gripper_kp, c.r1a7_gripper_kd
        if index in self.WRIST_INDICES:
            return c.r1a7_kp_wrist, c.r1a7_kd_wrist
        if index in self.MEDIUM_ARM_INDICES:
            return c.r1a7_kp_medium, c.r1a7_kd_medium
        if index in self.ARM_MOTOR_INDICES:
            return c.r1a7_kp_arm, c.r1a7_kd_arm
        if index == self.HEAD_PITCH_INDEX:
            return c.r1a7_kp_head, c.r1a7_kd_head
        if index == self.HEAD_YAW_INDEX:
            return c.r1a7_kp_head_yaw, c.r1a7_kd_head_yaw
        if index == self.WAIST_YAW_INDEX:
            return c.r1a7_kp_waist, c.r1a7_kd_waist
        return c.r1a7_kp_lock, c.r1a7_kd_lock

    def _init_lowcmd_defaults(self) -> None:
        for i in range(self.NUM_MOTORS):
            cmd = self._lowcmd_msg.motor_cmd[i]
            cmd.mode = 1
            cmd.q = self._latest_motor_q(i)
            cmd.dq = 0.0
            cmd.tau = 0.0
            kp, kd = self._kp_kd_for(i)
            cmd.kp = kp
            cmd.kd = kd

    def _apply_mode_machine(self) -> None:
        with self._state_lock:
            state = self._last_lowstate
        if state is not None and hasattr(state, "mode_machine"):
            self._lowcmd_msg.mode_machine = state.mode_machine

    def _clip_ctrl_target(self, target: np.ndarray) -> np.ndarray:
        """Advance the command toward the target by at most `allowed` per cycle.

        The ramp must start from self._q_command, the last value actually sent,
        and never from the measured angle. Starting from the measurement pins
        the command `allowed` ahead of wherever the joint currently is: as soon
        as gravity or load keeps it from covering that step, the next cycle
        recomputes from the new measurement and the arm stops moving entirely
        (a constant arm_track_err with motionless joints).
        """
        limit = self.cfg.r1a7_arm_velocity_limit
        if limit <= 0.0:
            self._q_command[:] = target
            return self._q_command.copy()
        allowed = abs(limit * self.cfg.r1a7_control_dt)
        if allowed <= 0.0:
            self._q_command[:] = target
            return self._q_command.copy()
        delta = np.clip(target - self._q_command, -allowed, allowed)
        self._q_command += delta
        return self._q_command.copy()

    # -- Control thread ----------------------------------------------------
    def _ctrl_loop(self) -> None:
        dt = max(self.cfg.r1a7_control_dt, 1e-3)
        hold_skip = set(self._controlled_indices)
        while self._running:
            start = time.monotonic()
            with self._state_lock:
                state = self._last_lowstate
            with self._ctrl_lock:
                q_target = self._q_target.copy()
                tau_target = self._tau_target.copy()

            # 1) Hold every non-controlled motor at its measured angle.
            if state is not None:
                for i in range(self.NUM_MOTORS):
                    if i in hold_skip:
                        continue
                    cmd = self._lowcmd_msg.motor_cmd[i]
                    cmd.q = float(state.motor_state[i].q)
                    cmd.dq = 0.0
                    cmd.tau = 0.0

            if self._internal_gripper and state is not None:
                with self._state_lock:
                    gripper_targets = list(self._gripper_cmd_sent)
                for index, target in zip(self._gripper_indices, gripper_targets):
                    if target is None:
                        target = float(state.motor_state[index].q)
                    cmd = self._lowcmd_msg.motor_cmd[index]
                    cmd.mode = 1
                    cmd.q = float(target)
                    cmd.dq = 0.0
                    cmd.tau = 0.0
                    cmd.kp = self.cfg.r1a7_gripper_kp
                    cmd.kd = self.cfg.r1a7_gripper_kd

            # 2) Overwrite the controlled motors with the ramped target.
            clipped = self._clip_ctrl_target(q_target)
            for idx, motor_index in enumerate(self._controlled_indices):
                cmd = self._lowcmd_msg.motor_cmd[motor_index]
                cmd.q = float(clipped[idx])
                cmd.dq = 0.0
                cmd.tau = float(tau_target[idx])

            self._apply_mode_machine()
            self._lowcmd_msg.crc = self._crc.Crc(self._lowcmd_msg)
            self._lowcmd_pub.Write(self._lowcmd_msg)

            elapsed = time.monotonic() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    # -- publish ----------------------------------------------------------
    def _update_targets(self, arm_q, arm_tau, base_cmd, head_q) -> None:
        q = np.asarray(arm_q, dtype=float).reshape(-1)
        if q.size != len(self.ARM_MOTOR_INDICES):
            raise ValueError(f"R1-A7 arm_q must have 14 values, got {q.size}")
        tau = np.asarray(arm_tau, dtype=float).reshape(-1) if arm_tau is not None else np.zeros(q.size)

        target = self._q_target.copy()
        tau_out = np.zeros(self._n_ctrl, dtype=float)
        target[: q.size] = q
        tau_out[: min(tau.size, q.size)] = tau[: q.size]
        slot = len(self.ARM_MOTOR_INDICES)

        # Optional waist yaw, taken from the body joints in base_cmd. When
        # disabled the torso is held locked by the ctrl thread.
        if self._waist_yaw_enabled:
            body = {name: float(v) for name, v in zip(self.cfg.base_cmd_joints, base_cmd[3:])}
            if self.cfg.r1a7_waist_yaw_joint in body:
                target[slot] = body[self.cfg.r1a7_waist_yaw_joint]
            slot += 1

        # Head joints, in cfg.head_joints order, skipping unmapped names.
        if head_q is not None and self._head_indices:
            hq = np.asarray(head_q, dtype=float).reshape(-1)
            for name, value in zip(self.cfg.head_joints, hq):
                if name in self._head_index_by_joint:
                    target[slot] = float(value)
                    slot += 1

        with self._ctrl_lock:
            self._q_target[:] = target
            self._tau_target[:] = tau_out

    def _write_grippers(self, gripper_q: np.ndarray | None) -> None:
        if gripper_q is None:
            return
        q = np.asarray(gripper_q, dtype=float).reshape(-1)
        if q.size == 0:
            return
        n_left = max(1, len(self.cfg.left_gripper_joints))
        left_urdf = float(np.mean(q[:n_left]))
        right_vals = q[n_left : n_left + max(1, len(self.cfg.right_gripper_joints))]
        right_urdf = float(np.mean(right_vals)) if right_vals.size else left_urdf
        # denom may be negative (this model has closed < open), so it must not
        # be clamped positive with max(.., 1e-6).
        denom = self.cfg.gripper_closed_position - self.cfg.gripper_open_position
        if abs(denom) < 1e-6:
            denom = 1e-6
        left_alpha = np.clip((left_urdf - self.cfg.gripper_open_position) / denom, 0.0, 1.0)
        right_alpha = np.clip((right_urdf - self.cfg.gripper_open_position) / denom, 0.0, 1.0)
        span = self.cfg.r1a7_gripper_max - self.cfg.r1a7_gripper_min
        # alpha is the closing ratio (full trigger gives 1). By default closed
        # maps to min and open to max; invert swaps them.
        left_frac = left_alpha if self.cfg.r1a7_gripper_invert else (1.0 - left_alpha)
        right_frac = right_alpha if self.cfg.r1a7_gripper_invert else (1.0 - right_alpha)
        left_cmd = self.cfg.r1a7_gripper_min + left_frac * span
        right_cmd = self.cfg.r1a7_gripper_min + right_frac * span

        # Command-side ramp against wall-clock time. The start point is the last
        # command, never the measured angle: measuring-side clamping pins the
        # command within +/-delta of wherever the jaw currently is, so once it
        # stalls on an object it can never be driven further. Using real time
        # also keeps the speed independent of the publish() call rate, which is
        # independent of the inference loop frame rate.
        now = time.monotonic()
        dt = 0.0 if self._gripper_cmd_time is None else max(now - self._gripper_cmd_time, 0.0)
        self._gripper_cmd_time = now
        speed = max(self.cfg.r1a7_gripper_speed, 0.0)
        if speed > 0.0 and dt > 0.0:
            allowed = speed * dt
            for slot, target in ((0, left_cmd), (1, right_cmd)):
                previous = self._gripper_cmd_sent[slot]
                if previous is None:
                    continue
                stepped = previous + float(np.clip(target - previous, -allowed, allowed))
                if slot == 0:
                    left_cmd = stepped
                else:
                    right_cmd = stepped

        left_cmd = float(np.clip(left_cmd, self.cfg.r1a7_gripper_min, self.cfg.r1a7_gripper_max))
        right_cmd = float(np.clip(right_cmd, self.cfg.r1a7_gripper_min, self.cfg.r1a7_gripper_max))

        if self._internal_gripper:
            with self._state_lock:
                self._gripper_cmd_sent[:] = [left_cmd, right_cmd]
        else:
            self._left_gripper_msg.cmds[0].q = left_cmd
            self._right_gripper_msg.cmds[0].q = right_cmd
            self._left_gripper_pub.Write(self._left_gripper_msg)
            self._right_gripper_pub.Write(self._right_gripper_msg)
        # Store commanded motor units in the final two entries of robot state.
        if not self._internal_gripper:
            with self._state_lock:
                self._gripper_cmd_sent[0] = left_cmd
                self._gripper_cmd_sent[1] = right_cmd

    def publish(
        self,
        arm_q: np.ndarray,
        arm_tau: np.ndarray,
        base_cmd: list[float],
        gripper_q: np.ndarray | None = None,
        head_q: np.ndarray | None = None,
        lift_position: float | None = None,
    ) -> None:
        # Arm, head and waist targets are streamed by the ctrl thread; here we
        # only update the target and write the grippers directly.
        self._update_targets(arm_q, arm_tau, base_cmd, head_q)
        self._write_grippers(gripper_q)
        self._publish_count += 1
        if self._publish_count % 50 == 0:
            current = self._current_ctrl_q()
            n_arm = len(self.ARM_MOTOR_INDICES)
            err = float(np.linalg.norm(self._q_target[:n_arm] - current[:n_arm]))
            # ramp_left is how far the command ramp still is from the target.
            # Above 0 means it is still climbing, which is normal. If it sits at
            # 0 while track_err stays large, the command arrived but the joint
            # did not follow (low kp, load, or not enabled).
            ramp = float(np.linalg.norm(self._q_target[:n_arm] - self._q_command[:n_arm]))
            with self._state_lock:
                meas = list(self._gripper_meas)
            print(
                "[r1a7_dds] "
                f"lowstate={'ok' if self._last_lowstate is not None else 'MISSING'} "
                f"mode_machine={getattr(self._lowcmd_msg, 'mode_machine', None)} "
                f"arm_track_err={err:.4f} ramp_left={ramp:.4f} "
                f"grip_cmd={self._gripper_cmd_sent[0]}/{self._gripper_cmd_sent[1]} "
                f"gripL_meas={('%.3f' % meas[0]) if meas[0] is not None else 'MISSING'} "
                f"gripR_meas={('%.3f' % meas[1]) if meas[1] is not None else 'MISSING'}"
            )

    def state_snapshot(self) -> dict:
        """Snapshot: 35 motor q/dq/tau_est, gripper measured/commanded angles,
        and the 14 measured arm angles."""
        with self._state_lock:
            st = self._last_lowstate
            grip = list(self._gripper_meas)
            grip_cmd = list(self._gripper_cmd_sent)
        out = {
            "motor_q": None,
            "motor_dq": None,
            "motor_tau": None,
            "measured_height_m": None,
            "waist_pitch_meas": None,
            "gripper_meas": [grip[0], grip[1]],
            # Commanded gripper angles in motor units; the last 2 action entries.
            "gripper_cmd": [grip_cmd[0], grip_cmd[1]],
            # Measured arm angles (motors 15..28); the first 14 state entries.
            "arm_q_meas": None,
            "waist_yaw_meas": None,
        }
        if st is not None:
            out["motor_q"] = [float(st.motor_state[i].q) for i in range(self.NUM_MOTORS)]
            out["motor_dq"] = [float(st.motor_state[i].dq) for i in range(self.NUM_MOTORS)]
            # unitree_hg MotorState exposes tau_est; fall back to 0 if absent.
            out["motor_tau"] = [
                float(getattr(st.motor_state[i], "tau_est", 0.0)) for i in range(self.NUM_MOTORS)
            ]
            out["arm_q_meas"] = [float(st.motor_state[i].q) for i in self.ARM_MOTOR_INDICES]
            out["waist_yaw_meas"] = float(st.motor_state[self.WAIST_YAW_INDEX].q)
        return out

    def close(self) -> None:
        # Arms and grippers take position commands, so there is no velocity to
        # zero out: hold the last gripper command and stop the ctrl thread.
        self._running = False


def make_publisher(mode: str, cfg: TeleopConfig, net: str) -> CommandPublisher:
    if mode != "robot":
        return PrintPublisher()
    if cfg.hardware_interface == "g1_d_dds":
        return G1DDSPublisher(cfg=cfg, net=net)
    if cfg.hardware_interface == "r1_a7_dds":
        return R1A7DDSPublisher(cfg=cfg, net=net)
    return UnitreeSDKPublisher(topic=cfg.publish_topic, net=net)
