"""Live monitor for a real-robot in-context rollout.

Shows the three prompt cameras next to what the model is doing with them: its
reasoning summary, the action chunk it returned, the step being executed and any
controller rejection. Without this the only visible output is a terminal that
sits silent for minutes while a request is in flight, which makes it impossible
to tell a slow request from a stalled one, or a sensible plan from a bad one
before the arm has finished moving.

Rendered with OpenCV, which is also used to decode the live camera stream.
"""

from __future__ import annotations

from collections import deque
import textwrap
import threading
import time

import numpy as np

PANEL_WIDTH = 520
CAMERA_LABELS = {
    "cam_head": "head",
    "cam_left_wrist": "left wrist",
    "cam_right_wrist": "right wrist",
}
WINDOW = "RoboICL live: cameras, reasoning, actions"

_WHITE = (240, 240, 240)
_DIM = (150, 150, 150)
_GREEN = (120, 230, 120)
_AMBER = (80, 190, 240)
_RED = (90, 90, 240)
_BG = (28, 28, 30)


class LiveMonitor:
    """Draw camera views and policy state into one window.

    All setters are cheap and safe to call from the control loop; drawing only
    happens in `render`, which the environment calls once per observation.
    """

    def __init__(self, scale: float = 0.6, enabled: bool = True, history: int = 6) -> None:
        self.enabled = bool(enabled)
        self.scale = float(scale)
        self._cv2 = None
        self._created = False
        self._closed = False

        self.step = 0
        self.step_lim = 0
        self.status = "waiting for the first observation"
        self.status_colour = _DIM
        self.reasoning: list[str] = []
        self.note = ""
        self.chunk: list[list[float]] = []
        self.chunk_index = 0
        self.turns = 0
        self.requests = 0
        self.tokens = 0
        self.rejections = 0
        self.last_rejection = ""
        self.events: deque[str] = deque(maxlen=history)
        self._request_started: float | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._frame_source = None
        self._period = 0.1
        self.quit_requested = False

        if self.enabled:
            try:
                import cv2

                self._cv2 = cv2
            except Exception as exc:
                print(f"[monitor] disabled, OpenCV unavailable ({exc})")
                self.enabled = False

    # -- state updates -----------------------------------------------------
    def set_status(self, text: str, colour=_WHITE) -> None:
        self.status = text
        self.status_colour = colour

    def request_started(self, turn: int) -> None:
        self.turns = turn
        self._request_started = time.monotonic()
        self.set_status(f"turn {turn}: waiting for the model", _AMBER)
        self.events.append(f"turn {turn}: request sent")

    def request_finished(self, payload: dict | None) -> None:
        elapsed = 0.0 if self._request_started is None else time.monotonic() - self._request_started
        self._request_started = None
        if payload is None:
            self.set_status(f"request failed after {elapsed:.0f}s", _RED)
            self.events.append(f"turn {self.turns}: request failed ({elapsed:.0f}s)")
            return
        self.requests += 1
        self.reasoning = payload.get("reasoning", [])
        self.note = payload.get("note", "")
        self.chunk = payload.get("actions", [])
        self.chunk_index = 0
        self.tokens += int(payload.get("tokens", 0))
        self.set_status(
            f"turn {self.turns}: {len(self.chunk)} actions in {elapsed:.0f}s", _GREEN
        )
        self.events.append(f"turn {self.turns}: {len(self.chunk)} actions ({elapsed:.0f}s)")

    def action_executed(self, step: int, index: int) -> None:
        self.step = step
        self.chunk_index = index
        self.set_status(f"executing action {index + 1}/{max(len(self.chunk), 1)}", _WHITE)

    def action_rejected(self, reason: str) -> None:
        self.rejections += 1
        self.last_rejection = reason
        self.set_status("action rejected by the IK guard; replanning", _RED)
        self.events.append(f"rejected: {reason}")

    # -- drawing -----------------------------------------------------------
    def _text(self, canvas, x, y, text, colour=_WHITE, size=0.42, weight=1):
        self._cv2.putText(
            canvas, text, (x, y), self._cv2.FONT_HERSHEY_SIMPLEX, size, colour, weight,
            self._cv2.LINE_AA,
        )

    def _camera_strip(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        """Head on top, the two wrists below, so the panel gets usable height.

        Three tiles in a row make a 1920px-wide image only 480 tall, which
        leaves the text panel too short for the model's reasoning. Stacking
        gives the panel 960px instead.
        """
        tiles = {}
        for name, label in CAMERA_LABELS.items():
            pixels = frames.get(name)
            tile = (
                np.zeros((480, 640, 3), dtype=np.uint8)
                if pixels is None
                else np.ascontiguousarray(pixels[:, :, :3]).copy()
            )
            self._text(tile, 12, 28, label, _GREEN, 0.7, 2)
            if pixels is None:
                self._text(tile, 12, 60, "no frame", _RED, 0.6, 2)
            tiles[name] = tile
        head = self._cv2.resize(tiles["cam_head"], (1280, 480))
        wrists = np.hstack([tiles["cam_left_wrist"], tiles["cam_right_wrist"]])
        return np.vstack([head, wrists])

    def _panel(self, height: int) -> np.ndarray:
        """Draw the status panel, dropping lower sections if space runs out.

        Camera tiles fix the available height, so the panel has to fit whatever
        it is given rather than overflow: sections are written top-down and the
        least important ones are skipped once the cursor nears the bottom.
        """
        panel = np.full((height, PANEL_WIDTH, 3), _BG, dtype=np.uint8)
        footer = height - 22
        y = 30

        def room(lines: int) -> bool:
            return y + lines * 18 <= footer

        self._text(panel, 14, y, "STATUS", _DIM, 0.45, 1)
        y += 24
        for line in textwrap.wrap(self.status, 52)[:2]:
            self._text(panel, 14, y, line, self.status_colour, 0.46, 1)
            y += 20
        if self._request_started is not None:
            waited = time.monotonic() - self._request_started
            self._text(panel, 14, y, f"elapsed {waited:.0f}s", _AMBER, 0.44, 1)
            y += 20

        y += 8
        budget = (
            f"step {self.step}/{self.step_lim}   turn {self.turns}   "
            f"ok {self.requests}   rejected {self.rejections}"
        )
        self._text(panel, 14, y, budget, _WHITE, 0.44, 1)
        y += 19
        self._text(panel, 14, y, f"tokens {self.tokens:,}", _DIM, 0.42, 1)
        y += 26

        if self.reasoning and room(3):
            self._text(panel, 14, y, "MODEL REASONING", _DIM, 0.45, 1)
            y += 21
            for item in self.reasoning:
                if not room(1):
                    break
                self._text(panel, 14, y, textwrap.shorten(item, 56), _WHITE, 0.42, 1)
                y += 18
            y += 8

        if self.note and room(3):
            self._text(panel, 14, y, "EXECUTION NOTE", _DIM, 0.45, 1)
            y += 21
            for line in textwrap.wrap(self.note, 54):
                if not room(1):
                    break
                self._text(panel, 14, y, line, _WHITE, 0.42, 1)
                y += 18
            y += 8

        if self.chunk and room(5):
            rows = np.asarray(self.chunk, dtype=float)
            self._text(panel, 14, y, "ACTION CHUNK", _DIM, 0.45, 1)
            y += 21
            self._text(
                panel, 14, y,
                f"{rows.shape[0]} steps, executing {self.chunk_index + 1}", _WHITE, 0.42, 1,
            )
            y += 19
            # ee_delta rows are [left dxyz, left drot, left grip, right ...].
            current = rows[min(self.chunk_index, rows.shape[0] - 1)]
            for side, offset in (("L", 0), ("R", 7)):
                if rows.shape[1] < offset + 7 or not room(2):
                    continue
                total = rows[:, offset : offset + 3].sum(axis=0) * 1000.0
                self._text(
                    panel, 14, y,
                    f"{side} chunk {total[0]:+6.1f} {total[1]:+6.1f} {total[2]:+6.1f} mm"
                    f"  grip {rows[-1, offset + 6]:.2f}",
                    _WHITE, 0.40, 1,
                )
                y += 18
                step = current[offset : offset + 3] * 1000.0
                self._text(
                    panel, 14, y,
                    f"{side} step  {step[0]:+6.1f} {step[1]:+6.1f} {step[2]:+6.1f} mm",
                    _AMBER, 0.40, 1,
                )
                y += 18
            y += 8

        if self.last_rejection and room(2):
            self._text(panel, 14, y, "LAST REJECTION", _DIM, 0.45, 1)
            y += 21
            for line in textwrap.wrap(self.last_rejection, 54):
                if not room(1):
                    break
                self._text(panel, 14, y, line, _RED, 0.40, 1)
                y += 18
            y += 8

        if self.events and room(2):
            self._text(panel, 14, y, "RECENT", _DIM, 0.45, 1)
            y += 21
            for item in list(self.events)[::-1]:
                if not room(1):
                    break
                self._text(panel, 14, y, textwrap.shorten(item, 56), _DIM, 0.40, 1)
                y += 17

        self._text(panel, 14, height - 8, "q or ESC closes this window", _DIM, 0.40, 1)
        return panel

    def render(self, frames: dict[str, np.ndarray]) -> bool:
        """Draw one frame. Returns True when the user asked to close the window."""
        if not self.enabled or self._closed:
            return False
        cv2 = self._cv2
        strip = self._camera_strip(frames)
        canvas = np.hstack([strip, self._panel(strip.shape[0])])
        if self.scale != 1.0:
            canvas = cv2.resize(
                canvas, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA
            )
        if not self._created:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, canvas.shape[1], canvas.shape[0])
            self._created = True
        cv2.imshow(WINDOW, canvas)
        return (cv2.waitKey(1) & 0xFF) in (27, ord("q"))

    def close(self) -> None:
        self.stop()
        if self._created and not self._closed:
            try:
                self._cv2.destroyWindow(WINDOW)
            except Exception:
                pass
        self._closed = True

    # -- self-driven rendering ---------------------------------------------
    def start(self, frame_source, fps: float = 10.0) -> None:
        """Render continuously from `frame_source` on a worker thread.

        A request can take minutes, and `get_obs` is only called between
        actions, so driving the window from the control loop alone would leave
        it frozen exactly when the operator wants to see progress. OpenCV GUI
        calls are confined to this one thread.
        """
        if not self.enabled or self._thread is not None:
            return
        self._frame_source = frame_source
        self._period = 1.0 / max(float(fps), 1.0)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="roboicl-monitor", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            start = time.monotonic()
            try:
                frames = self._frame_source()
            except Exception:
                frames = {}
            try:
                if self.render(frames):
                    self.quit_requested = True
                    break
            except Exception as exc:
                print(f"[monitor] stopped after a draw error: {exc}")
                break
            elapsed = time.monotonic() - start
            if elapsed < self._period:
                time.sleep(self._period - elapsed)

    def stop(self) -> None:
        self._running = False
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
