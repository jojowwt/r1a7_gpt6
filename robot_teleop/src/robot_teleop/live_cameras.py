"""Camera transport and calibrated geometry used by live RoboICL inference."""

from __future__ import annotations

import json
import struct
import threading
import time
from pathlib import Path

import numpy as np

from .camera_convention import opencv_to_usd
from .config import CONFIG_DIR, TeleopConfig


CAMERA_MOUNT_DEFAULTS = {
    "head": ("head_yaw_link", "cam_head"),
    "left_wrist": ("left_wrist_yaw_link", "cam_left_wrist"),
    "right_wrist": ("right_wrist_yaw_link", "cam_right_wrist"),
}


class FrameGeometry:
    """Resolve end-effector poses and camera geometry from the current joints."""

    def __init__(self, ik_solver, cfg: TeleopConfig, calibration_path: Path | None = None) -> None:
        self._ik = ik_solver
        self._cfg = cfg
        self._cameras: dict[str, dict] = {}
        self._world_link = ""
        self._world_inverse: np.ndarray | None = None
        path = Path(calibration_path) if calibration_path else CONFIG_DIR / "camera_calibration.yaml"
        if path.is_file():
            self._load_calibration(path)

    def _load_calibration(self, path: Path) -> None:
        import yaml

        data = yaml.safe_load(path.read_text()) or {}
        self._world_link = str(data.get("world_frame_link", "") or "")
        for name, record in (data.get("cameras") or {}).items():
            transform = record.get("link_to_camera")
            if transform is None:
                continue
            self._cameras[name] = {
                "roboicl_name": record.get("roboicl_name", f"cam_{name}"),
                "mount_link": record["mount_link"],
                "link_to_camera": np.asarray(transform, dtype=float).reshape(4, 4),
                "intrinsic_matrix": np.asarray(record["intrinsic_matrix"], dtype=float).reshape(3, 3),
                "image_size": list(record.get("image_size", [640, 480])),
                "calibrated": True,
            }
        if self._cameras:
            self._add_placeholder_cameras()

    def _add_placeholder_cameras(self) -> None:
        for name, (mount_link, camera_name) in CAMERA_MOUNT_DEFAULTS.items():
            if name in self._cameras:
                continue
            reference = next(iter(self._cameras.values()))
            self._cameras[name] = {
                "roboicl_name": camera_name,
                "mount_link": mount_link,
                "link_to_camera": np.eye(4),
                "intrinsic_matrix": reference["intrinsic_matrix"].copy(),
                "image_size": list(reference["image_size"]),
                "calibrated": False,
            }

    @property
    def world_frame_link(self) -> str:
        return self._world_link

    @staticmethod
    def _matrix(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        out = np.eye(4)
        out[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
        out[:3, 3] = np.asarray(pos, dtype=float).reshape(3)
        return out

    def _world_frame_inverse(self) -> np.ndarray | None:
        if not self._world_link:
            return None
        if self._world_inverse is None:
            try:
                pos, quat = self._ik.frame_pose(self._world_link)
            except Exception as exc:
                print(f"[camera] world frame link {self._world_link!r} unavailable: {exc}")
                self._world_link = ""
                return None
            self._world_inverse = np.linalg.inv(self._matrix(pos, quat))
        return self._world_inverse

    def _pose_in_world(self, frame_name: str) -> list[float] | None:
        from scipy.spatial.transform import Rotation

        try:
            pos, quat_xyzw = self._ik.frame_pose(frame_name)
        except Exception:
            return None
        transform = self._matrix(pos, quat_xyzw)
        reference = self._world_frame_inverse()
        if reference is not None:
            transform = reference @ transform
        quat = Rotation.from_matrix(transform[:3, :3]).as_quat()
        return [
            float(transform[0, 3]), float(transform[1, 3]), float(transform[2, 3]),
            float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2]),
        ]

    def ee_poses(self) -> dict[str, list[float] | None]:
        return {
            "left": self._pose_in_world(self._cfg.left_ee_body),
            "right": self._pose_in_world(self._cfg.right_ee_body),
        }

    def camera_geometry(
        self, image_sizes: dict[str, tuple[int, int]] | None = None
    ) -> dict[str, dict]:
        out: dict[str, dict] = {}
        reference = self._world_frame_inverse()
        for record in self._cameras.values():
            try:
                pos, quat = self._ik.frame_pose(record["mount_link"])
            except Exception:
                continue
            transform = self._matrix(pos, quat) @ record["link_to_camera"]
            if reference is not None:
                transform = reference @ transform
            camera_name = record["roboicl_name"]
            intrinsic = record["intrinsic_matrix"].copy()
            source_width, source_height = record["image_size"]
            target_size = (image_sizes or {}).get(camera_name)
            if target_size is not None:
                target_width, target_height = map(int, target_size)
                if min(source_width, source_height, target_width, target_height) <= 0:
                    raise ValueError(f"Invalid calibrated or delivered size for {camera_name}")
                intrinsic[0, :] *= target_width / source_width
                intrinsic[1, :] *= target_height / source_height
                image_size = [target_width, target_height]
            else:
                image_size = [int(source_width), int(source_height)]
            out[camera_name] = {
                "intrinsic_matrix": intrinsic.tolist(),
                "camera_to_world": transform.tolist(),
                "image_size": image_size,
                "calibrated": bool(record.get("calibrated", False)),
            }
        return out


class FourCameraZMQReceiver:
    """Subscribe to the image server's four-camera stream for model input."""

    STREAMS = ["head_left", "head_right", "wrist_left", "wrist_right"]

    def __init__(self, host: str, port: int) -> None:
        self.endpoint = f"tcp://{host}:{int(port)}"
        self._frames: tuple[np.ndarray, ...] | None = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_error = ""

    def start(self) -> None:
        try:
            import cv2  # noqa: F401
            import zmq  # noqa: F401
        except Exception as exc:
            raise RuntimeError(f"live camera input requires pyzmq and OpenCV: {exc}") from exc
        self._running = True
        self._thread = threading.Thread(target=self._run, name="r1a7-camera-sub", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import zmq

        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.RCVHWM, 2)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 500)
        try:
            sock.connect(self.endpoint)
            print(f"[camera] connected to image server: {self.endpoint}")
            while self._running:
                try:
                    frames = self._decode_parts(sock.recv_multipart())
                except zmq.Again:
                    continue
                except Exception as exc:
                    if self._running and str(exc) != self._last_error:
                        self._last_error = str(exc)
                        print(f"[camera] dropped invalid image frame: {exc}")
                    continue
                with self._lock:
                    self._frames = frames
                    self._frame_id += 1
        finally:
            sock.close(0)

    @classmethod
    def _decode_parts(cls, parts: list[bytes]) -> tuple[np.ndarray, ...]:
        import cv2

        if len(parts) == 6:
            topic, meta_raw, *jpeg_parts = parts
            if topic.decode(errors="replace") != "r1a7/head_stereo":
                raise ValueError(f"unexpected camera topic {topic!r}")
            meta = json.loads(meta_raw)
            if meta.get("streams") != cls.STREAMS:
                raise ValueError("camera stream order does not match head L/R, wrist L/R")
            frames = tuple(
                cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                for data in jpeg_parts
            )
            if any(frame is None for frame in frames):
                raise ValueError("failed to decode a multipart JPEG")
            return frames

        if len(parts) != 1:
            raise ValueError(f"expected one legacy frame or six multipart frames, got {len(parts)}")
        message = bytes(parts[0])
        header_size = struct.calcsize("II")
        if len(message) < header_size:
            raise ValueError("image message is shorter than its header")
        color_size, ir_size = struct.unpack("II", message[:header_size])
        color_end = header_size + color_size
        if color_size <= 0 or color_end > len(message):
            raise ValueError("invalid JPEG size in image header")
        if ir_size and color_end + ir_size > len(message):
            raise ValueError("invalid IR size in image header")
        combined = cv2.imdecode(
            np.frombuffer(message[header_size:color_end], dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if combined is None or combined.shape[1] % 4:
            raise ValueError("combined camera JPEG must decode to a width divisible by four")
        return tuple(np.hsplit(combined, 4))

    def wait_first_frame(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(float(timeout), 0.0)
        while time.monotonic() < deadline:
            with self._lock:
                if self._frames is not None:
                    return True
            time.sleep(0.05)
        with self._lock:
            return self._frames is not None

    def get(self) -> tuple[tuple[np.ndarray, ...] | None, int]:
        with self._lock:
            if self._frames is None:
                return None, self._frame_id
            return tuple(frame.copy() for frame in self._frames), self._frame_id

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
