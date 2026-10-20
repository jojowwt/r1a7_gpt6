"""Qt console for a real-robot in-context rollout.

Shows the prompt cameras next to what the policy is doing with them: its
reasoning summary, the action chunk being executed, joint targets, request
timings and any controller rejection. Without it the only feedback is a
terminal that sits silent for minutes during a request, which makes a slow
request indistinguishable from a stalled one.

The layout is three columns -- vision, decision, progress -- because those are
the three questions an operator asks while a rollout runs, in that order: what
does the robot see, what did the model decide, and how far along is it.

Nothing in the progress column is scripted. The policy does not emit a task
breakdown, so the timeline is built from what actually happened: one entry per
model turn, titled with that turn's own execution note. A fixed list of
plausible-looking stages would read as ground truth while being invented.

Threading follows the reference desktop app: the control loop only writes to a
lock-protected snapshot, and a QTimer pulls from it. No widget is ever touched
off the Qt thread.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

CAMERA_TILES = (
    ("cam_head", "HEAD"),
    ("cam_left_wrist", "WRIST · LEFT"),
    ("cam_right_wrist", "WRIST · RIGHT"),
)
# Per-robot tile layouts. A single-arm robot has no left wrist camera, so
# showing a permanently black tile for it would waste the space and read as a
# fault rather than as an absence.
LAYOUTS = {
    "dual": CAMERA_TILES,
    "single": (
        ("cam_head", "HEAD · 全局"),
        ("cam_right_wrist", "WRIST · 腕部"),
    ),
}
# Column offsets of each arm inside an ee_delta action row:
# [dxyz, drot, gripper] per side.
ARM_COLUMNS = {
    "dual": (("左臂", 0), ("右臂", 7)),
    "single": (("机械臂", 7),),
}
JOINT_NAMES = ("J1 肩旋", "J2 肩抬", "J3 肘", "J4 腕俯", "J5 腕转")

# One palette, named, so a colour is changed in one place rather than in the
# dozen stylesheets that use it.
BG = "#0a1018"
PANEL = "#111b29"
LINE = "#1f3045"
TEXT = "#dbe6f2"
MUTED = "#7e93ab"
ACCENT = "#2fe3b0"
WARN = "#ffc46b"
BAD = "#ff8f7a"

STYLE = f'''
QWidget {{ background:{BG}; color:{TEXT}; font:14px "Noto Sans CJK SC","Source Han Sans SC",sans-serif; }}
QLabel {{ background:transparent; }}
QFrame#card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px; }}
QFrame#topbar {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px; }}
QLabel#cardTitle {{ font-size:16px; font-weight:600; color:{TEXT}; }}
QLabel#sectionTitle {{ font-size:13px; color:{MUTED}; }}
QLabel#appTitle {{ font-size:16px; font-weight:600; }}
QLabel#instruction {{ font-size:21px; font-weight:600; line-height:150%; color:#f0f6ff; }}
QLabel#reasoning {{ font-size:14px; color:#c3d3e4; line-height:168%; }}
QLabel#muted {{ color:{MUTED}; }}
QLabel#metricValue {{ font-size:19px; font-weight:600; }}
QLabel#tileImage {{ background:#080e16; border:none; border-radius:10px; }}
QLabel#badge {{ background:rgba(8,14,22,0.78); color:{TEXT}; font-size:11px;
                padding:3px 8px; border-radius:6px; border:1px solid {LINE}; }}
QPushButton#stop {{ background:#2a1b22; color:#ff9f8f; border:1px solid #5d2f32;
                    border-radius:8px; padding:6px 14px; font-size:13px; }}
QPushButton#stop:hover {{ background:#3a242c; }}
QScrollArea {{ border:none; background:transparent; }}
QScrollBar:vertical {{ background:transparent; width:8px; margin:0; }}
QScrollBar::handle:vertical {{ background:#27384e; border-radius:4px; min-height:30px; }}
QScrollBar::add-line,QScrollBar::sub-line {{ height:0; }}
'''


class RolloutState:
    """What the control loop publishes and the UI reads.

    Every setter takes the lock and every read copies, so the Qt thread never
    sees a half-written update and the control loop never blocks on painting.
    """

    def __init__(self, step_lim: int = 0, frames_are_rgb: bool = False) -> None:
        self._lock = threading.Lock()
        # Whether incoming frames are RGB. The two robots differ: the R1-A7's
        # arrive from cv2.imdecode and are BGR, while this project's wrappers
        # convert to RGB because that is what the policy encodes. Only the
        # caller knows which, so it says so rather than this guessing.
        self.frames_are_rgb = bool(frames_are_rgb)
        self.frames: dict[str, bytes] = {}
        self.frame_ms: dict[str, float] = {}
        self._frame_at: dict[str, float] = {}
        self.started_at = time.monotonic()
        self.step = 0
        self.step_lim = int(step_lim)
        self.status = "等待第一帧观测"
        self.status_kind = "idle"
        self.reasoning: list[str] = []
        self.note = ""
        self.chunk: list[list[float]] = []
        self.chunk_index = 0
        self.turn = 0
        self.ok_requests = 0
        self.rejections = 0
        self.tokens = 0
        self.last_latency = 0.0
        self.last_rejection = ""
        self.joints_target: list[float] = []
        self.joints_measured: list[float] = []
        self.gripper_target = 0.0
        # One entry per model turn, in order. The progress column is rendered
        # straight from this; see the module docstring on why it is not a
        # fixed stage list.
        self.turns: list[dict] = []
        self.events: list[str] = []
        self.request_started: float | None = None
        self.finished = False
        self.quit_requested = False

    # -- written by the control loop ---------------------------------------
    def set_frames(self, frames: dict[str, np.ndarray]) -> None:
        """Store JPEG bytes; encoding here keeps cv2 off the Qt thread.

        `cv2.imencode` assumes BGR, so an RGB frame written straight through
        swaps red and blue and a yellow object renders cyan. Which order
        arrives depends on the robot, so `frames_are_rgb` decides rather than
        this converting unconditionally.
        """
        import cv2

        encoded = {}
        for name, pixels in frames.items():
            if pixels is None:
                continue
            image = pixels[:, :, :3]
            if self.frames_are_rgb:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok:
                encoded[name] = buffer.tobytes()
        now = time.monotonic()
        with self._lock:
            for name in encoded:
                previous = self._frame_at.get(name)
                if previous is not None:
                    self.frame_ms[name] = (now - previous) * 1000.0
                self._frame_at[name] = now
            self.frames = encoded

    def set_status(self, text: str, kind: str = "info") -> None:
        with self._lock:
            self.status, self.status_kind = text, kind

    def set_joints(self, target, measured=None, gripper: float | None = None) -> None:
        """Publish the joint command just dispatched, and what is measured.

        Optional: the R1-A7 path does not call it, and the panel simply shows
        nothing rather than inventing a readout.
        """
        with self._lock:
            self.joints_target = [float(v) for v in np.asarray(target).reshape(-1)]
            if measured is not None:
                self.joints_measured = [
                    float(v) for v in np.asarray(measured).reshape(-1)
                ]
            if gripper is not None:
                self.gripper_target = float(gripper)

    def request_started_now(self, turn: int) -> None:
        with self._lock:
            self._close_open_turn("done")
            self.turn = turn
            self.chunk = []
            self.chunk_index = 0
            self.request_started = time.monotonic()
            self.status, self.status_kind = f"第 {turn} 轮：等待模型响应", "wait"
            self.turns.append(
                {"index": turn, "title": "等待模型响应", "detail": "推理中", "state": "active"}
            )
            self.events.append(f"第 {turn} 轮：请求已发出")

    def _close_open_turn(self, state: str) -> None:
        """Mark the newest turn finished. Caller holds the lock."""
        if self.turns and self.turns[-1]["state"] == "active":
            self.turns[-1]["state"] = state
            if state == "done":
                self.turns[-1]["detail"] = "已完成"

    def request_finished(self, payload: dict | None) -> None:
        with self._lock:
            elapsed = 0.0 if self.request_started is None else time.monotonic() - self.request_started
            self.request_started = None
            self.last_latency = elapsed
            if payload is None:
                self.status, self.status_kind = f"请求失败，耗时 {elapsed:.0f}s", "error"
                if self.turns:
                    self.turns[-1].update(
                        title="请求失败", detail=f"{elapsed:.0f}s", state="error"
                    )
                self.events.append(f"第 {self.turn} 轮：请求失败（{elapsed:.0f}s）")
                return
            self.ok_requests += 1
            self.reasoning = list(payload.get("reasoning", []))
            self.note = str(payload.get("note", ""))
            self.chunk = list(payload.get("actions", []))
            self.chunk_index = 0
            self.tokens += int(payload.get("tokens", 0))
            self.status = f"第 {self.turn} 轮：收到 {len(self.chunk)} 步动作（{elapsed:.0f}s）"
            self.status_kind = "ok"
            if self.turns:
                # The model's own note is the honest title for this turn; the
                # reasoning summary is the fallback when it gave none.
                title = self.note.strip() or (
                    self.reasoning[0] if self.reasoning else f"第 {self.turn} 轮决策"
                )
                self.turns[-1].update(
                    title=title,
                    detail=f"{len(self.chunk)} 步 · {elapsed:.0f}s",
                    state="active",
                )
            self.events.append(f"第 {self.turn} 轮：{len(self.chunk)} 步动作（{elapsed:.0f}s）")

    def action_executed(self, step: int, index: int) -> None:
        with self._lock:
            self.step, self.chunk_index = step, index
            total = max(len(self.chunk), 1)
            self.status = f"执行动作 {min(index + 1, total)}/{total}"
            self.status_kind = "info"
            if self.turns and self.turns[-1]["state"] == "active":
                self.turns[-1]["detail"] = f"执行 {min(index + 1, total)}/{total}"

    def action_rejected(self, reason: str) -> None:
        with self._lock:
            self.rejections += 1
            self.last_rejection = reason
            self.status, self.status_kind = "动作被 IK 守卫拒绝，正在重新规划", "error"
            self.events.append(f"拒绝：{reason}")

    def decision_stopped(self, reason: str) -> None:
        """A valid model response stopped execution, rather than an API failure."""
        with self._lock:
            self.status, self.status_kind = "模型主动停止：" + reason, "wait"
            self.finished = True
            if self.turns:
                self.turns[-1].update(title="模型主动停止", detail=reason, state="error")
            self.events.append("模型主动停止：" + reason)

    def decision_rejected(self, payload: dict, reason: str) -> None:
        """Show the model's explanation without presenting its proposal as executed."""
        with self._lock:
            self.reasoning = list(payload.get("reasoning", []))
            self.note = str(payload.get("note", ""))
            self.chunk = []
            self.chunk_index = 0
            self.tokens += int(payload.get("tokens", 0))
            self.status, self.status_kind = "本地安全门拒绝动作：" + reason, "error"
            self.finished = True
            if self.turns:
                self.turns[-1].update(
                    title="动作被本地安全门拒绝", detail=reason, state="error"
                )
            self.events.append("本地安全门拒绝动作：" + reason)

    def mark_finished(self, text: str) -> None:
        with self._lock:
            self._close_open_turn("done")
            self.finished = True
            self.status, self.status_kind = text, "info"

    def mark_failed(self, text: str) -> None:
        with self._lock:
            self._close_open_turn("error")
            self.finished = True
            self.status, self.status_kind = text, "error"

    # -- read by the UI ----------------------------------------------------
    def take_frames(self) -> dict[str, bytes]:
        """Hand the UI the current JPEGs and stop sharing them.

        The frames must leave the lock as objects the writer will never touch
        again. A plain `dict(self.frames)` copies the mapping but shares every
        `bytes` inside it, so the Qt thread decodes buffers whose refcount the
        control thread is simultaneously dropping when `set_frames` replaces
        them. Two threads mutating one refcount is a data race, and it crashed
        the interpreter outright with `none_dealloc: deallocating None`.

        Detaching the dict under the lock means the writer's next `set_frames`
        rebinds `self.frames` to a fresh mapping and leaves this one's contents
        owned solely by the caller.
        """
        with self._lock:
            frames, self.frames = self.frames, {}
            return frames, dict(self.frame_ms)

    def snapshot(self) -> dict:
        """Everything except the frames, which `take_frames` owns.

        Deliberately excludes the JPEG buffers: they are expensive to copy and
        this is polled ten times a second.
        """
        with self._lock:
            waited = (
                None if self.request_started is None else time.monotonic() - self.request_started
            )
            return {
                "step": self.step, "step_lim": self.step_lim,
                "status": self.status, "status_kind": self.status_kind,
                "reasoning": list(self.reasoning), "note": self.note,
                "chunk": [list(row) for row in self.chunk], "chunk_index": self.chunk_index,
                "turn": self.turn, "ok_requests": self.ok_requests,
                "rejections": self.rejections, "tokens": self.tokens,
                "last_rejection": self.last_rejection, "last_latency": self.last_latency,
                "joints_target": list(self.joints_target),
                "joints_measured": list(self.joints_measured),
                "gripper_target": self.gripper_target,
                "turns": [dict(item) for item in self.turns],
                "elapsed": time.monotonic() - self.started_at,
                "events": self.events[-8:], "waited": waited, "finished": self.finished,
            }

    def request_quit(self) -> None:
        with self._lock:
            self.quit_requested = True

    @property
    def should_quit(self) -> bool:
        with self._lock:
            return self.quit_requested


# -- small building blocks -------------------------------------------------
def _card(title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    box = QVBoxLayout(frame)
    box.setContentsMargins(14, 12, 14, 12)
    box.setSpacing(8)
    if title:
        label = QLabel(title)
        label.setObjectName("cardTitle")
        box.addWidget(label)
    return frame, box


class _CameraTile(QWidget):
    """One camera view with its label and refresh interval floating on top.

    The badges overlay the image rather than sitting above it so the picture
    keeps the full width of its column; with three tiles stacked, a header row
    each would cost more vertical space than the wrist views can spare.
    """

    def __init__(self, caption: str) -> None:
        super().__init__()
        self.caption = caption
        self.image = QLabel("无画面", self)
        self.image.setObjectName("tileImage")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(200, 140)
        # Ignored in both directions: otherwise the scaled pixmap grows the
        # label, which grows the next scaled pixmap, and the tile creeps.
        self.image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.name_badge = QLabel(f"● {caption}", self)
        self.name_badge.setObjectName("badge")
        self.rate_badge = QLabel("--", self)
        self.rate_badge.setObjectName("badge")
        for badge in (self.name_badge, self.rate_badge):
            badge.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._pixmap: QPixmap | None = None

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        if pixmap is not None:
            self._pixmap = pixmap
        self._redraw()

    def set_rate(self, milliseconds: float | None) -> None:
        self.rate_badge.setText(
            "--" if not milliseconds else f"{milliseconds:.0f} ms · {1000.0 / milliseconds:.0f} FPS"
        )
        self.rate_badge.adjustSize()
        self._place_badges()

    def _redraw(self) -> None:
        if self._pixmap is None:
            self.image.setText("无画面")
            return
        # Preserve the complete camera view, including fingers and plate edges.
        self.image.setPixmap(self._pixmap.scaled(
            self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def _place_badges(self) -> None:
        self.name_badge.move(10, 10)
        self.rate_badge.move(max(self.width() - self.rate_badge.width() - 10, 10), 10)
        self.name_badge.raise_()
        self.rate_badge.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.image.setGeometry(0, 0, self.width(), self.height())
        self.name_badge.adjustSize()
        self._place_badges()
        self._redraw()


def _headline(text: str, limit: int = 26) -> str:
    """The first clause of a model note, for use as a one-line title.

    `execution_note` is a paragraph: it describes the targets dispatched, the
    tracking error and what the next observation is meant to settle. Rendering
    all of it in a narrow column pushes every later turn off screen, so the
    timeline shows the opening clause and keeps the rest in the tooltip.
    """
    text = " ".join(str(text).split())
    for mark in ("。", "，", "；", ". ", ", ", "; "):
        head, sep, _ = text.partition(mark)
        if sep and len(head) <= limit:
            return head
    return text if len(text) <= limit else text[: limit - 1] + "…"


class _TurnRow(QWidget):
    """One model turn in the progress column: dot, title, status."""

    DOT = {
        "done": (ACCENT, "✓"),
        "active": (ACCENT, "●"),
        "error": (BAD, "!"),
        "idle": ("#34465c", ""),
    }

    def __init__(self) -> None:
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        rail = QVBoxLayout()
        rail.setContentsMargins(0, 1, 0, 0)
        rail.setSpacing(0)
        self.dot = QLabel("")
        self.dot.setFixedSize(18, 18)
        self.dot.setAlignment(Qt.AlignCenter)
        rail.addWidget(self.dot, 0, Qt.AlignTop)
        self.line = QFrame()
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFixedWidth(2)
        self.line.setStyleSheet(f"background:{LINE};border:none;")
        rail.addWidget(self.line, 1)
        row.addLayout(rail)

        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(3)
        self.title = QLabel("")
        self.title.setWordWrap(True)
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setObjectName("muted")
        column.addWidget(self.title)
        column.addWidget(self.detail)
        column.addStretch(1)
        row.addLayout(column, 1)

    def update_row(self, index: int, title: str, detail: str, state: str,
                   is_last: bool) -> None:
        colour, glyph = self.DOT.get(state, self.DOT["idle"])
        self.dot.setText(glyph)
        fill = colour if state != "idle" else "transparent"
        self.dot.setStyleSheet(
            f"border:2px solid {colour};border-radius:9px;font-size:10px;"
            + (f"background:{fill};color:#06131b;" if state == "done" else f"color:{colour};")
        )
        weight = "600" if state == "active" else "400"
        shade = ACCENT if state == "active" else TEXT
        self.title.setStyleSheet(f"font-size:14px;font-weight:{weight};color:{shade};")
        self.title.setText(f"{index:02d}  {_headline(title)}")
        # The full text is worth keeping reachable: the headline is a first
        # clause, and the rest of the note is often where the caveat lives.
        self.title.setToolTip(title)
        self.detail.setText(detail)
        self.line.setVisible(not is_last)


class ConsoleWindow(QWidget):
    """Vision on the left, the model's decision in the middle, progress right."""

    def __init__(self, state: RolloutState, task: str, instruction: str,
                 layout_name: str = "dual") -> None:
        super().__init__()
        self.state = state
        self.layout_name = layout_name if layout_name in LAYOUTS else "dual"
        self.tiles: dict[str, _CameraTile] = {}
        self._pixmaps: dict[str, QPixmap] = {}
        self._turn_rows: list[_TurnRow] = []
        self.setWindowTitle("RoboICL 真机推理控制台")
        self.resize(1560, 920)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(12)
        root.addWidget(self._build_topbar(task))

        columns = QHBoxLayout()
        columns.setSpacing(12)
        root.addLayout(columns, 1)
        vision = self._build_vision()
        vision.setMinimumWidth(440)
        decision = self._build_decision(instruction)
        decision.setMinimumWidth(360)
        progress = self._build_progress()
        progress.setMinimumWidth(200)
        progress.setMaximumWidth(300)
        columns.addWidget(vision, 46)
        columns.addWidget(decision, 38)
        columns.addWidget(progress, 16)

        # Images refresh slower than status: decoding the JPEGs is the
        # expensive part and the scene barely changes between frames.
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(100)
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.poll_cameras)
        self.camera_timer.start(300)

    # -- construction ------------------------------------------------------
    def _build_topbar(self, task: str) -> QFrame:
        bar = QFrame()
        bar.setObjectName("topbar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(16, 10, 16, 10)
        dot = QLabel("◉")
        dot.setStyleSheet(f"color:{ACCENT};font-size:16px;")
        row.addWidget(dot)
        title = QLabel(f"具身智能机器人控制台 · {task}")
        title.setObjectName("appTitle")
        row.addWidget(title)
        row.addStretch(1)
        self.topbar_stats = QLabel("")
        self.topbar_stats.setObjectName("muted")
        row.addWidget(self.topbar_stats)
        self.clock = QLabel("")
        self.clock.setObjectName("muted")
        row.addWidget(self.clock)
        return bar

    def _build_vision(self) -> QFrame:
        card, box = _card()
        header = QHBoxLayout()
        label = QLabel("实时视觉")
        label.setObjectName("cardTitle")
        header.addWidget(label)
        header.addStretch(1)
        self.camera_count = QLabel("")
        self.camera_count.setObjectName("muted")
        header.addWidget(self.camera_count)
        box.addLayout(header)

        tiles = LAYOUTS[self.layout_name]
        head = _CameraTile(tiles[0][1])
        self.tiles[tiles[0][0]] = head
        box.addWidget(head, 1 if self.layout_name == "single" else 3)
        if len(tiles) > 1:
            row = QHBoxLayout()
            row.setSpacing(10)
            for name, caption in tiles[1:]:
                tile = _CameraTile(caption)
                self.tiles[name] = tile
                row.addWidget(tile, 1)
            box.addLayout(row, 1 if self.layout_name == "single" else 2)
        return card

    def _build_decision(self, instruction: str) -> QWidget:
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(12)

        card, box = _card()
        header = QHBoxLayout()
        title = QLabel("任务指令")
        title.setObjectName("cardTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.status_pill = QLabel("启动中")
        self.status_pill.setWordWrap(True)
        self.status_pill.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.stop_button = QPushButton("■ 停止任务")
        self.stop_button.setObjectName("stop")
        self.stop_button.clicked.connect(self._on_stop)
        header.addWidget(self.stop_button)
        box.addLayout(header)
        box.addWidget(self.status_pill)
        self.instruction = QLabel(instruction)
        self.instruction.setObjectName("instruction")
        self.instruction.setWordWrap(True)
        box.addWidget(self.instruction)
        column.addWidget(card)

        card, box = _card("模型推理")
        self.reasoning = QLabel("等待第一次推理")
        self.reasoning.setObjectName("reasoning")
        self.reasoning.setWordWrap(True)
        self.reasoning.setAlignment(Qt.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.reasoning)
        box.addWidget(scroll, 1)
        column.addWidget(card, 1)

        card, box = _card("执行动作")
        self.action_grid = QGridLayout()
        self.action_grid.setHorizontalSpacing(14)
        self.action_grid.setVerticalSpacing(4)
        # The joint readout needs only its digits; the chunk summary is prose
        # and takes what is left, so it does not wrap at six characters.
        for index in range(3):
            self.action_grid.setColumnStretch(index, 0)
        self.action_grid.setColumnMinimumWidth(3, 12)
        self.action_grid.setColumnStretch(4, 1)
        box.addLayout(self.action_grid)
        self._action_cells: dict[tuple[int, int], QLabel] = {}
        column.addWidget(card)

        card, box = _card("系统状态")
        metrics = QHBoxLayout()
        metrics.setSpacing(18)
        self.metrics: dict[str, QLabel] = {}
        for key, caption in (
            ("elapsed", "总用时"), ("progress", "执行进度"),
            ("latency", "推理延迟"), ("tokens", "累计 tokens"),
        ):
            cell = QVBoxLayout()
            cell.setSpacing(2)
            head = QLabel(caption)
            head.setObjectName("sectionTitle")
            value = QLabel("--")
            value.setObjectName("metricValue")
            cell.addWidget(head)
            cell.addWidget(value)
            metrics.addLayout(cell)
            self.metrics[key] = value
        metrics.addStretch(1)
        box.addLayout(metrics)
        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(f"color:{BAD};")
        self.warning.setVisible(False)
        box.addWidget(self.warning)
        column.addWidget(card)
        return holder

    def _build_progress(self) -> QFrame:
        card, box = _card("任务阶段")
        hint = QLabel("每一行是模型的一次决策")
        hint.setObjectName("muted")
        box.addWidget(hint)
        inner = QWidget()
        self.turn_column = QVBoxLayout(inner)
        self.turn_column.setContentsMargins(0, 6, 6, 0)
        self.turn_column.setSpacing(14)
        self.turn_column.addStretch(1)
        self.turn_scroll = QScrollArea()
        self.turn_scroll.setWidgetResizable(True)
        self.turn_scroll.setWidget(inner)
        box.addWidget(self.turn_scroll, 1)
        return card

    # -- polling -----------------------------------------------------------
    def _on_stop(self) -> None:
        self.stop_button.setText("正在停止…")
        self.stop_button.setEnabled(False)
        self.state.request_quit()

    def poll_cameras(self) -> None:
        # Decoded pixmaps are cached because `take_frames` hands over each JPEG
        # exactly once: a tick with no new capture must redraw the last one
        # rather than blank the tile.
        frames, rates = self.state.take_frames()
        for name, jpeg in frames.items():
            if not jpeg:
                continue
            pixmap = QPixmap()
            if pixmap.loadFromData(jpeg):
                self._pixmaps[name] = pixmap
        live = 0
        for name, tile in self.tiles.items():
            pixmap = self._pixmaps.get(name)
            tile.set_pixmap(pixmap)
            tile.set_rate(rates.get(name))
            live += pixmap is not None
        self.camera_count.setText(f"{live}/{len(self.tiles)} 路在线")

    def poll(self) -> None:
        s = self.state.snapshot()
        colour = {"ok": ACCENT, "error": BAD, "wait": WARN}.get(s["status_kind"], ACCENT)
        text = s["status"]
        if s["waited"] is not None:
            text += f" · {s['waited']:.0f}s"
        # The pill's tint follows its text: an error outlined in the healthy
        # green reads as fine at a glance, which is the one moment it must not.
        tint = {"ok": "47,227,176", "error": "255,143,122", "wait": "255,196,107"}.get(
            s["status_kind"], "47,227,176"
        )
        self.status_pill.setStyleSheet(
            f"color:{colour};background:rgba({tint},0.10);border:1px solid {colour};"
            "border-radius:10px;padding:3px 10px;font-size:12px;"
        )
        if self.status_pill.text() != text:
            self.status_pill.setText(text)

        stats = (
            f"轮次 {s['turn']}　成功 {s['ok_requests']}　拒绝 {s['rejections']}"
        )
        if self.topbar_stats.text() != stats:
            self.topbar_stats.setText(stats)
        self.clock.setText(time.strftime("　%Y年%m月%d日 %H:%M"))

        body = "\n\n".join(s["reasoning"]) if s["reasoning"] else "等待第一次推理"
        if s["note"]:
            body = f"{s['note']}\n\n{body}"
        if self.reasoning.text() != body:
            self.reasoning.setText(body)

        self._render_actions(s)
        self._render_metrics(s)
        self._render_turns(s["turns"])

        warning = f"最近一次拒绝：{s['last_rejection']}" if s["last_rejection"] else ""
        if self.warning.text() != warning:
            self.warning.setText(warning)
            self.warning.setVisible(bool(warning))

    def _cell(self, row: int, col: int, text: str, kind: str = "value") -> None:
        label = self._action_cells.get((row, col))
        if label is None:
            label = QLabel()
            style = {
                "head": f"color:{MUTED};font-size:12px;",
                "name": f"color:{MUTED};",
                "value": "font-family:monospace;",
            }[kind]
            label.setStyleSheet(style)
            label.setAlignment(Qt.AlignRight if kind == "value" else Qt.AlignLeft)
            self.action_grid.addWidget(label, row, col)
            self._action_cells[(row, col)] = label
        if label.text() != text:
            label.setText(text)

    def _render_actions(self, s: dict) -> None:
        """Joint targets on the left, the Cartesian chunk on the right.

        Both are real: the joint column is what was last dispatched to the
        motors, the chunk column is the model's own ee_delta row. The joint
        table appears only on robots that publish one -- the R1-A7 path does
        not, and a column of dashes would be clutter claiming to be a readout.
        """
        target = s["joints_target"]
        measured = s["joints_measured"]
        rows_used = 0
        if target or measured:
            self._cell(0, 0, "关节", "head")
            self._cell(0, 1, "目标 rad", "head")
            self._cell(0, 2, "实测 rad", "head")
            for i, name in enumerate(JOINT_NAMES):
                row = i + 1
                self._cell(row, 0, name, "name")
                self._cell(row, 1, f"{target[i]:+.3f}" if i < len(target) else "--")
                self._cell(row, 2, f"{measured[i]:+.3f}" if i < len(measured) else "--")
            grip_row = len(JOINT_NAMES) + 1
            self._cell(grip_row, 0, "夹爪", "name")
            self._cell(grip_row, 1, f"{s['gripper_target'] * 1000:.1f} mm")
            self._cell(grip_row, 2, "")
            rows_used = grip_row + 1

        chunk = s["chunk"]
        self._cell(0, 4, "本轮动作块", "head")
        if not chunk:
            lines = ["尚未收到"]
        else:
            rows = np.asarray(chunk, dtype=float)
            index = min(s["chunk_index"], rows.shape[0] - 1)
            lines = [f"{rows.shape[0]} 步，执行第 {index + 1} 步"]
            # ee_delta rows are [left dxyz, left drot, left grip, right ...]. A
            # single-arm robot only occupies the right half; showing the
            # placeholder left side would be reporting on a limb that is not
            # there.
            for side, offset in ARM_COLUMNS[self.layout_name]:
                if rows.shape[1] < offset + 7:
                    continue
                total = rows[:, offset : offset + 3].sum(axis=0) * 1000.0
                step = rows[index, offset : offset + 3] * 1000.0
                lines.append(
                    f"{side} 整块 {total[0]:+.0f} {total[1]:+.0f} {total[2]:+.0f} mm"
                )
                lines.append(
                    f"{side} 本步 {step[0]:+.1f} {step[1]:+.1f} {step[2]:+.1f} mm"
                )
                lines.append(f"{side} 夹爪指令 {rows[index, offset + 6]:.2f}")
        # Sized to the text, not to the joint table: a two-arm chunk needs more
        # rows than a five-joint arm has, and truncating to the shorter of the
        # two silently dropped the second arm's gripper command.
        for row in range(1, max(len(lines) + 1, rows_used)):
            self._cell(row, 4, lines[row - 1] if row - 1 < len(lines) else "", "name")

    def _render_metrics(self, s: dict) -> None:
        seconds = int(s["elapsed"])
        self.metrics["elapsed"].setText(
            f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
        )
        self.metrics["progress"].setText(f"{s['step']}/{s['step_lim']}")
        latency = s["waited"] if s["waited"] is not None else s["last_latency"]
        self.metrics["latency"].setText(f"{latency:.0f}s" if latency else "--")
        self.metrics["tokens"].setText(f"{s['tokens']:,}")

    def _render_turns(self, turns: list[dict]) -> None:
        """Newest first, so the live turn is visible without scrolling."""
        ordered = list(reversed(turns))
        while len(self._turn_rows) < len(ordered):
            row = _TurnRow()
            self._turn_rows.append(row)
            self.turn_column.insertWidget(len(self._turn_rows) - 1, row)
        for position, row in enumerate(self._turn_rows):
            if position >= len(ordered):
                row.setVisible(False)
                continue
            item = ordered[position]
            row.setVisible(True)
            row.update_row(
                item["index"], item["title"], item["detail"], item["state"],
                position == len(ordered) - 1,
            )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.poll_cameras()

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self.camera_timer.stop()
        self.state.request_quit()
        event.accept()


def run_console(state: RolloutState, task: str, instruction: str,
                layout: str = "dual") -> None:
    """Own the Qt event loop on the calling thread until the window closes."""
    import os
    import sys

    # opencv-python ships a private Qt plugin set that breaks native PySide6.
    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        os.environ.pop(key, None)
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = ConsoleWindow(state, task, instruction, layout)
    window.show()
    app.exec()
    state.request_quit()
