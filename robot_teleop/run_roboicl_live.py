#!/usr/bin/env python3
"""Run a RoboICL in-context rollout on the real R1-A7.

RoboICL ships as a RoboDojo/Isaac Sim harness, but its policy half (prompting,
anchored interaction memory, action validation, transport) only needs an
environment exposing seven members. This script wires that policy to the real
robot through `R1A7LiveEnv`, so no simulator is involved and actions go through
the R1-A7 DDS publisher.

Prerequisites:

  1. a device-specific head-camera calibration at
     robot_teleop/config/camera_calibration.yaml (prepare it with your camera
     calibration workflow; calibration tooling is not included here)
  2. image_server publishing the required camera streams
  3. optionally, a RoboICL train_reference_bundle.json built from your own
     demonstrations using the RoboICL data-preparation workflow
  4. ASTRA_API_KEY in the environment for live model requests

Example::

    export ASTRA_API_KEY="$GPT_KEY"
    python robot_teleop/run_roboicl_live.py \\
        --task bread_to_plate --net enp5s0 \\
        --reference /tmp/roboicl_data/runtime-data/bread_to_plate/reference_1shot_train12_endpoint_h12

Add --dry-run to exercise the whole loop with a scripted policy and no API call.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Annotated, Literal

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
ROBOICL = ROOT.parent / "third_party/RoboICL"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROBOICL) not in sys.path:
    sys.path.insert(0, str(ROBOICL))

# RoboICL's own path helper puts its pinned submodules on sys.path. This is not
# optional: the policy reaches into XPolicyLab (image codecs) and RoboDojo (the
# IK continuity guard) lazily, deep inside calls like update_obs, so a missing
# entry only surfaces once a rollout is already running against the robot.
from roboicl.paths import configure_imports  # noqa: E402

configure_imports("policy")
ROBODOJO = ROBOICL / "third_party/RoboDojo"
XPOLICYLAB = ROBOICL / "third_party/XPolicyLab"

import numpy as np
import tyro

from robot_teleop.config import CONFIG_DIR, config_path_for_robot, load_config


class LocalPolicyClient:
    """Adapt a policy object to the `client.call(func_name=..., **kwargs)` shape.

    Upstream runs the policy in a separate process behind a WebSocket, because
    the simulator and policy environments have conflicting dependencies. Here
    both live in one interpreter, so the RPC layer is replaced by a direct call
    while keeping the interface `rollout.py` expects.

    This is also the only place that sees a turn begin and end, so it is where
    the monitor learns what the model said and how long it took.
    """

    def __init__(self, model, monitor=None, log_dir: Path | None = None) -> None:
        self.model = model
        self.monitor = monitor
        self.log_dir = log_dir
        self.turn = 0

    def call(self, func_name: str, **kwargs):
        method = getattr(self.model, func_name, None)
        if method is None:
            raise AttributeError(f"Policy has no method {func_name}")
        if func_name != "get_action" or self.monitor is None:
            return method(**kwargs)
        self.turn += 1
        self.monitor.request_started(self.turn)
        try:
            actions = method(**kwargs)
        except Exception:
            self.monitor.request_finished(None)
            raise
        self.monitor.request_finished(self._summarize(actions))
        return actions

    def _summarize(self, actions) -> dict:
        """Pull the reasoning summary and raw chunk out of the latest exchange.

        The validated actions handed back to the rollout are absolute poses, so
        the human-readable plan has to come from the logged exchange instead.
        """
        summary = {"actions": [], "reasoning": [], "note": "", "tokens": 0}
        try:
            summary["actions"] = [
                list(np.asarray(a.get("left_ee_pose", []), dtype=float))
                for a in (actions or [])
            ]
        except Exception:
            pass
        exchange = self._latest_exchange()
        if not exchange:
            return summary
        response = exchange.get("response") or {}
        reasoning: list[str] = []
        for item in response.get("output") or []:
            if item.get("type") == "reasoning":
                for part in item.get("summary") or []:
                    text = part.get("text", "").strip().strip("*")
                    if text:
                        reasoning.append(text)
            elif item.get("type") == "function_call":
                arguments = item.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except Exception:
                        arguments = {}
                if isinstance(arguments, dict):
                    summary["note"] = str(arguments.get("execution_note", ""))
                    rows = arguments.get("actions")
                    if isinstance(rows, list):
                        summary["actions"] = rows
        summary["reasoning"] = reasoning
        summary["tokens"] = int((response.get("usage") or {}).get("total_tokens", 0))
        return summary

    def _latest_exchange(self) -> dict | None:
        directory = getattr(self.model, "episode_dir", None) or self.log_dir
        if directory is None:
            return None
        path = Path(directory) / "exchanges.jsonl"
        if not path.is_file():
            return None
        try:
            last = None
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        last = line
            return json.loads(last) if last else None
        except Exception:
            return None


class ScriptedPolicy:
    """Replay recorded actions instead of calling the model.

    Used by --dry-run to prove the observation, IK and publishing path works
    before any API credit is spent: a failure here is ours, not the model's.
    """

    def __init__(self, bundle_json: Path | None, horizon: int) -> None:
        self.horizon = int(horizon)
        self.chunks: list[list[dict]] = []
        self.index = 0
        self.state: dict | None = None
        if bundle_json is not None:
            bundle = json.loads(Path(bundle_json).read_text())
            for example in bundle["examples"]:
                rows = example["action_tensor"]
                self.chunks.append([
                    {
                        "mode": "ee_delta",
                        "left": row[:6],
                        "left_gripper": row[6],
                        "right": row[7:13],
                        "right_gripper": row[13],
                    }
                    for row in rows
                ])
            print(f"[dry-run] replaying {len(self.chunks)} recorded action chunks")

    def reset(self):
        self.index = 0

    def update_obs(self, obs):
        self.state = obs.get("state")

    def get_action(self):
        from roboicl.policy.astra_policy import validate_actions

        if not self.chunks:
            # No bundle: hold the measured pose so the loop still exercises
            # observation, IK and publishing.
            raw = [{
                "mode": "ee_delta",
                "left": [0.0] * 6,
                "left_gripper": self.state["left_ee_joint_state"][0],
                "right": [0.0] * 6,
                "right_gripper": self.state["right_ee_joint_state"][0],
            }]
        else:
            raw = self.chunks[self.index % len(self.chunks)]
            self.index += 1
        return validate_actions(raw, len(raw), self.state)

    def notify_action_rejection(self, obs):
        print(f"[dry-run] action rejected at step {obs.get('step')}: {obs.get('event')}")

    def finalize(self):
        print(f"[dry-run] finished after {self.index} chunks")


def _check_instruction_matches_demo(reference_json: Path, instruction: str) -> None:
    """Fail early when the demonstration and live instructions differ.

    `dialogue_policy` requires them to be identical and rejects a mismatch only
    once it builds the first request, which is after the robot has been taken
    over. The bundle instruction was fixed when the reference bundle was built,
    so a task config written later can easily differ by a word or even a full
    stop.
    """
    demo = json.loads(reference_json.read_text()).get("instruction", "")
    if demo == instruction:
        return
    detail = ""
    if demo and instruction.startswith(demo):
        detail = f"\n  the live text adds {instruction[len(demo):]!r}"
    elif demo.startswith(instruction):
        detail = f"\n  the demonstration adds {demo[len(instruction):]!r}"
    raise SystemExit(
        "The demonstration and live instructions must match exactly.\n"
        f"  demonstration: {demo!r}\n"
        f"  live:          {instruction!r}{detail}\n"
        "Fix the task config description, or pass --instruction with the "
        "demonstration's exact text."
    )


def _preflight_imports(need_guard: bool) -> None:
    """Import everything the rollout reaches for lazily, before touching the robot.

    The policy imports XPolicyLab's image codec inside `update_obs` and the
    rollout imports RoboDojo's guard inside its loop. A missing submodule
    therefore surfaces only after the robot has been taken over and, with a
    real model, after a request has already been paid for. Fail here instead.
    """
    checks = [("XPolicyLab.utils.process_data", XPOLICYLAB, "third_party/XPolicyLab")]
    if need_guard:
        checks.append(("src.eval_client.ik_continuity", ROBODOJO, "third_party/RoboDojo"))
    for module, location, submodule in checks:
        try:
            __import__(module)
        except ModuleNotFoundError as exc:
            raise SystemExit(
                f"Cannot import {module} (missing {exc.name}); expected it under {location}.\n"
                "Fetch the pinned submodule:\n"
                f"  cd third_party/RoboICL && git submodule update --init {submodule}"
            ) from None
    # The demonstration bundle is re-verified against its source HDF5 on the
    # first observation, so a missing or moved source also belongs here.
    print(f"[live] preflight: {', '.join(name for name, _, _ in checks)} importable")


class _QtAdapter:
    """Bridge the env and policy callbacks onto the Qt console's shared state.

    `R1A7LiveEnv` and `LocalPolicyClient` call the same small set of methods on
    whatever monitor they are given, so the Qt console only has to answer to
    that interface rather than the env knowing which UI is in use.
    """

    def __init__(self, state) -> None:
        self.state = state
        self.step_lim = state.step_lim

    def render(self, frames: dict) -> bool:
        self.state.set_frames(frames)
        return self.state.should_quit

    def request_started(self, turn: int) -> None:
        self.state.request_started_now(turn)

    def request_finished(self, payload: dict | None) -> None:
        self.state.request_finished(payload)

    def action_executed(self, step: int, index: int) -> None:
        self.state.action_executed(step, index)

    def action_rejected(self, reason: str) -> None:
        self.state.action_rejected(reason)

    def close(self) -> None:
        pass


def _compose_harness(task: str, task_config: dict, profile_path: Path | None,
                     horizon: int | None, save_dir: Path,
                     overrides: dict | None = None) -> Path:
    """Build the harness config the policy reads from ASTRA_HARNESS_CONFIG.

    Without this the policy falls back to DialogueConfig's defaults, which
    differ from the published protocol in ways that matter: a 24-step horizon
    instead of 15, a 2 cm translation bound instead of 3 cm, no action
    corrections and no transport retries. Compose it in the same order
    `roboicl/run.py` does, so a live rollout uses the same contract as the
    paper harness: shared defaults, then protocol, then task, then an explicit
    horizon override.
    """
    from roboicl.config import load_adaptive_horizons, load_profile, render_official_task_prompt

    config = json.loads((ROBOICL / "configs/shared_harness.json").read_text())
    if profile_path is not None:
        profile = load_profile(profile_path)
        config.update(profile.get("harness_overrides", {}))
        task_horizons = profile["task_horizons"]
    else:
        task_horizons = load_adaptive_horizons()
    # Task contracts refine shared and protocol defaults, so they go last.
    config.update(task_config.get("harness_overrides", {}))
    chosen = horizon if horizon is not None else task_horizons[task]
    config.update(predict_horizon=chosen, execute_horizon=chosen)
    if not 1 <= int(config["predict_horizon"]) <= 64:
        raise SystemExit("action horizon must be in [1, 64]")

    version, source, prompt = render_official_task_prompt(task_config)
    config["official_task_documentation_version"] = version
    config["official_task_documentation_source"] = source
    config["official_task_prompt"] = prompt
    config["reference_instruction_aliases"] = []
    config["localized_task_prompt_zh"] = task_config.get("localized_task_prompt_zh", "")
    config["reasoning_language"] = "zh"
    # Gateway capabilities come last: a relay may reject options the official
    # endpoint accepts, and that is a property of the deployment rather than of
    # the protocol or the task.
    if overrides:
        config.update(overrides)

    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "harness.json"
    path.write_text(json.dumps(config, indent=2))
    return path


def _build_policy(task: str, model_name: str, seed: int, reasoning_effort: str):
    """Construct the dialogue policy.

    The demonstration is not passed through this config: `bounded_policy` reads
    it from the ASTRA_DEMO_MANIFEST environment variable, which the caller sets
    to the bundle JSON before this runs.
    """
    from roboicl.policy.dialogue_policy import Model

    return Model({
        "policy_name": "RoboICL",
        "model": model_name,
        "bench_name": "RoboDojo",
        "task_name": task,
        "seed": seed,
        "action_type": "ee",
        "max_chunk": 64,
        "max_tool_rounds": 16,
        "reasoning_effort": reasoning_effort,
    })


def run(
    task: str,
    net: str,
    robot: str = "r1_a7_new",
    config: str | None = None,
    common_config: str = str(CONFIG_DIR / "common.yaml"),
    reference: str | None = None,
    data_root: str | None = None,
    profile: str | None = None,
    horizon: int | None = None,
    cache_mode: str = "implicit",
    model: str = "gpt-6-astra",
    seed: int = 0,
    reasoning_effort: str = "xhigh",
    step_lim: int = 1200,
    control_hz: float = 25.0,
    results_dir: str = "./results/roboicl_live",
    instruction: str | None = None,
    dry_run: bool = False,
    ui: str = "qt",
    monitor_scale: float = 0.6,
    release_motion_mode: bool = True,
    init_pose: Annotated[
        Literal["true", "false"],
        tyro.conf.arg(
            name="init-pose",
            aliases=["--init_pose"],
            help="Move both arms to the configured initial pose and open both grippers before inference.",
        ),
    ] = "false",
) -> None:
    """Drive the real R1-A7 from a RoboICL in-context policy.

    task:
      Task name registered under third_party/RoboICL/configs/tasks/.
    net:
      DDS interface, e.g. enp5s0.
    reference:
      One-shot reference bundle: either its directory or the
      train_reference_bundle.json inside it. Omit for a zero-shot rollout.
    data_root:
      ROBOICL_DATA_ROOT used to resolve the bundle's source HDF5 for its
      integrity re-check. Defaults to the existing environment value.
    profile:
      Protocol profile JSON, for example
      third_party/RoboICL/configs/protocols/one_shot_j12_b12.json. Its
      harness_overrides are layered onto the shared defaults.
    horizon:
      Override the per-task action horizon. Must match the horizon the
      reference bundle was built with.
    cache_mode:
      implicit - send explicit prompt cache breakpoints (the published default).
      omit     - leave them out, for a gateway that rejects
                 prompt_cache_breakpoint. Costs cache hits, nothing else.
    model:
      Model identifier passed to the policy transport.
    step_lim:
      Maximum controller steps before the episode is cut off.
    control_hz:
      Rate at which actions are issued, matching the 25 Hz demonstrations.
    instruction:
      Override the task instruction. Defaults to the task documentation.
    dry_run:
      Replay the reference actions instead of calling the model. No API key
      needed; use it to validate the hardware path first.
    ui:
      qt   - Qt console with the three cameras, reasoning and action chunk.
      cv   - minimal OpenCV preview window.
      none - no window; terminal output only.
    monitor_scale:
      Scale factor for the OpenCV window.
    release_motion_mode:
      Release the robot's built-in controller before taking over rt/lowcmd.
    """
    from roboicl.config import load_task

    task_config = load_task(task)
    if instruction is None:
        instruction = task_config["official_task_documentation"]["description"]
    reference_path = Path(reference).expanduser().resolve() if reference else None
    if reference_path is not None:
        # A directory is the usual form; accept the bundle JSON directly too.
        if reference_path.is_dir():
            reference_path = reference_path / "train_reference_bundle.json"
        if not reference_path.is_file():
            raise SystemExit(f"Reference bundle not found: {reference_path}")
        # bounded_policy reads the demonstration from the environment, and
        # resolves each episode's source_file against ROBOICL_DATA_ROOT.
        os.environ["ASTRA_DEMO_MANIFEST"] = str(reference_path)
        if data_root:
            os.environ["ROBOICL_DATA_ROOT"] = str(Path(data_root).expanduser().resolve())
        elif not os.environ.get("ROBOICL_DATA_ROOT"):
            print(
                "[live] warning: ROBOICL_DATA_ROOT is unset, so the bundle's source "
                "HDF5 may not be found for its integrity re-check"
            )
    if not dry_run and not os.environ.get("ASTRA_API_KEY"):
        raise SystemExit(
            "ASTRA_API_KEY is not set. Export it first:\n"
            '  export ASTRA_API_KEY="$GPT_KEY"\n'
            "Or pass --dry-run to validate the hardware path without the model."
        )

    config_path = config if config is not None else str(config_path_for_robot(robot))
    cfg = load_config(config_path, common_config)
    cfg.robot_name = robot
    cfg.r1a7_release_motion_mode = release_motion_mode
    if init_pose == "true" and robot != "r1_a7_new":
        raise SystemExit("--init_pose true is supported only with --robot r1_a7_new")

    # Resolve the harness contract and prove every lazy import works before the
    # robot is taken over or a request is paid for.
    save_dir = Path(results_dir).expanduser().resolve() / task
    if cache_mode not in ("implicit", "omit"):
        raise SystemExit("--cache-mode must be implicit or omit")
    if ui not in ("qt", "cv", "none"):
        raise SystemExit("--ui must be qt, cv or none")
    harness_path = _compose_harness(
        task,
        task_config,
        Path(profile).expanduser().resolve() if profile else None,
        horizon,
        save_dir,
        overrides={"cache_mode": cache_mode},
    )
    os.environ["ASTRA_HARNESS_CONFIG"] = str(harness_path)
    os.environ.setdefault("ASTRA_LOG_DIR", str(save_dir))

    from roboicl.policy.dialogue_policy import load_config as load_harness

    harness = load_harness(str(harness_path))
    print(
        f"[live] harness: horizon={harness.predict_horizon} "
        f"max_translation={harness.max_translation_m}m "
        f"max_rotation={harness.max_rotation_rad}rad "
        f"corrections={harness.max_action_corrections} "
        f"transport_attempts={harness.max_transport_attempts} "
        f"cache_mode={harness.cache_mode}"
    )
    _preflight_imports(need_guard=bool(harness.ik_continuity_guard))
    if reference_path is not None:
        _check_instruction_matches_demo(reference_path, instruction)
        print("[live] preflight: demonstration instruction matches")
    if harness.ik_continuity_guard:
        print(f"[live] IK continuity guard on at {harness.max_joint_increment_rad} rad/joint")

    from robot_teleop.placo_robot import PlacoIKSolver
    from robot_teleop.publisher import make_publisher
    from robot_teleop.live_cameras import FourCameraZMQReceiver, FrameGeometry
    from robot_teleop.roboicl_live_env import R1A7LiveEnv

    # Load the selected UI before constructing a hardware publisher. Qt plugin
    # or shared-library problems must fail before motion mode is released.
    view = None
    qt_state = None
    if ui == "qt":
        from robot_teleop.qt_console import RolloutState

        qt_state = RolloutState(step_lim=step_lim)
        view = _QtAdapter(qt_state)
    elif ui == "cv":
        from robot_teleop.live_monitor import LiveMonitor

        view = LiveMonitor(scale=monitor_scale, enabled=True)

    ik = PlacoIKSolver(
        cfg.urdf_path,
        cfg.controlled_joints,
        cfg.left_ee_body,
        cfg.right_ee_body,
        dt=cfg.ik_dt,
        regularization_weight=cfg.placo_regularization_weight,
        body_joints=cfg.body_joints,
        body_regularization_weight=cfg.body_regularization_weight,
        body_joint_regularization_weights=cfg.body_joint_regularization_weights,
        masked_joints=cfg.ik_masked_joints,
        max_joint_step=cfg.max_joint_step,
        reset_joint_positions=cfg.reset_joint_positions,
    )
    geometry = FrameGeometry(ik, cfg)
    if "cam_head" not in geometry.camera_geometry():
        raise SystemExit(
            "No head camera calibration found at "
            f"{CONFIG_DIR / 'camera_calibration.yaml'}. Prepare a device-specific "
            "calibration with your camera calibration workflow before connecting "
            "to the robot. Calibration tooling is not included in this repository."
        )

    images = FourCameraZMQReceiver(cfg.image_server_host, cfg.image_server_port)
    images.start()
    if not images.wait_first_frame(cfg.camera_start_timeout):
        raise SystemExit(
            f"No four-camera frames from {images.endpoint}. Start image_server first."
        )

    try:
        publisher = make_publisher("robot", cfg, net)
    except Exception:
        images.close()
        raise
    env = R1A7LiveEnv(
        cfg=cfg,
        publisher=publisher,
        ik_solver=ik,
        geometry=geometry,
        images=images,
        instruction=instruction,
        save_dir=save_dir,
        step_lim=step_lim,
        control_hz=control_hz,
        monitor=view,
    )
    if init_pose == "true":
        try:
            env.move_to_initial_pose()
        except BaseException:
            env.close()
            raise
    else:
        print("[live] init_pose=false: keeping current arm and gripper positions")
    if ui == "cv":
        view.start(env.monitor_frames, fps=10.0)
    elif ui == "qt":
        env.start_frame_feed(fps=2.5)

    policy = (
        ScriptedPolicy(reference_path, harness.predict_horizon)
        if dry_run
        else _build_policy(task, model, seed, reasoning_effort)
    )
    client = LocalPolicyClient(policy, monitor=view, log_dir=save_dir)

    print(f"[live] task={task} instruction={instruction!r}")
    print(f"[live] reference={'none (zero-shot)' if reference_path is None else reference_path}")
    print(f"[live] step_lim={step_lim} control_hz={control_hz} save_dir={save_dir}")
    print(f"[live] mode={'DRY RUN, no model call' if dry_run else model}")
    print(f"[live] ui={ui}")
    print("[live] clear the workspace; Ctrl+C to stop")

    from roboicl.rollout import eval_one_episode

    def rollout() -> None:
        try:
            eval_one_episode(env, client)
        except KeyboardInterrupt:
            print("\n[live] interrupted")
        except Exception as exc:
            print(f"\n[live] rollout failed: {type(exc).__name__}: {exc}")
            if qt_state is not None:
                qt_state.mark_finished(f"回合异常结束：{type(exc).__name__}")
            raise
        finally:
            if env._policy_failure:
                print(f"[live] policy failure: {env._policy_failure}")
                (save_dir / "policy_failure.json").write_text(
                    json.dumps(env._policy_failure, ensure_ascii=False, indent=2)
                )
            env.close()
            print(f"[live] stopped after {env.steps} steps; logs in {save_dir}")
            if qt_state is not None:
                qt_state.mark_finished(f"回合结束，共执行 {env.steps} 步")

    if qt_state is None:
        try:
            rollout()
        finally:
            if view is not None:
                view.close()
        return

    # Qt owns the main thread, so the rollout runs beside it. The worker is a
    # daemon: closing the window should not leave the robot being driven by a
    # loop nobody is watching.
    import threading

    worker = threading.Thread(target=rollout, name="roboicl-rollout", daemon=False)
    worker.start()
    from robot_teleop.qt_console import run_console

    run_console(qt_state, task, instruction)
    worker.join(timeout=5.0)
    if worker.is_alive():
        print("[live] window closed; waiting for the current policy request to finish")
        env.step_lim = 0
        worker.join()


def cli() -> None:
    tyro.cli(run)


if __name__ == "__main__":
    cli()
