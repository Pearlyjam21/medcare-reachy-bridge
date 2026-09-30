"""Robot facade: camera frames, prerecorded audio, head motion.

`reachy_mini` (and its GStreamer/WebRTC stack) is imported lazily so tests and the
video backend never need it.
"""

import logging
import math
import threading
import time
import wave
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

FRAME_WIDTH, FRAME_HEIGHT = 640, 480
SCAN_YAWS_DEG = (0, 25, 40, 25, 0, -25, -40, -25)   # gentle sweep while searching


class Robot(Protocol):
    def get_frame(self) -> np.ndarray | None: ...        # BGR 640x480, or None when no new frame
    def play_clip(self, path: Path) -> bool: ...
    def set_head(self, pose: np.ndarray) -> None: ...
    def hold_head(self) -> None: ...
    def look_around(self, step: int) -> None: ...
    def wake(self) -> None: ...
    def sleep(self) -> None: ...
    def is_reachable(self) -> bool: ...


def crop_4_3(frame: np.ndarray) -> np.ndarray:
    """Centre-crop a wide frame to 4:3 and resize to 640x480 (spec 00 §6 M2)."""
    import cv2

    height, width = frame.shape[:2]
    target_width = int(height * 4 / 3)
    if width > target_width:
        x = (width - target_width) // 2
        frame = frame[:, x:x + target_width]
    if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    return frame


def head_pose(yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    """4x4 head pose, same convention as reachy_mini.utils.create_head_pose (extrinsic xyz, degrees)."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    pose = np.eye(4)
    pose[:3, :3] = rz @ ry
    return pose


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV -> mono float32 in [-1, 1], sample rate."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM clips are supported")
        rate, channels = wav.getframerate(), wav.getnchannels()
        data = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def resample(samples: np.ndarray, rate: int, target: int) -> np.ndarray:
    if rate == target or len(samples) == 0:
        return samples
    count = max(1, round(len(samples) * target / rate))
    positions = np.linspace(0, len(samples) - 1, count)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)


class ReachyRobot:
    """Reachy Mini Wireless over the SDK's WebRTC media backend."""

    CHUNK = 1024

    def __init__(self, host: str):
        self.host = host
        self._mini = None
        self._owned = True              # False when a Reachy Mini app runtime owns the connection
        self._lock = threading.Lock()   # the SDK is driven from worker threads

    @classmethod
    def attach(cls, mini) -> "ReachyRobot":
        """Use a connection the Reachy Mini app runtime already opened (and will close)."""
        robot = cls(host="localhost")
        robot._mini = mini
        robot._owned = False
        return robot

    def connect(self) -> None:
        from reachy_mini import ReachyMini   # lazy: pulls in GStreamer

        mini = ReachyMini(host=self.host, media_backend="webrtc")
        mini.__enter__()
        self._mini = mini
        log.info("connected to Reachy Mini at %s", self.host)

    def close(self) -> None:
        if self._mini is not None and self._owned:
            try:
                self._mini.__exit__(None, None, None)
            finally:
                self._mini = None

    def is_reachable(self) -> bool:
        if self._mini is None:
            try:
                self.connect()
            except Exception as exc:   # SDK raises assorted connection errors
                log.warning("robot unreachable: %r", exc)
                return False
        return True

    def get_frame(self) -> np.ndarray | None:
        if self._mini is None:
            return None
        frame = self._mini.media.get_frame()
        return None if frame is None else crop_4_3(frame)

    def play_clip(self, path: Path) -> bool:
        if self._mini is None:
            return False
        samples, rate = read_wav(Path(path))
        media = self._mini.media
        samples = resample(samples, rate, int(media.get_output_audio_samplerate()))
        with self._lock:
            media.start_playing()
            for start in range(0, len(samples), self.CHUNK):
                media.push_audio_sample(samples[start:start + self.CHUNK])
        # Playback is asynchronous in the SDK; keep the robot quiet for the clip's length.
        time.sleep(len(samples) / int(media.get_output_audio_samplerate()))
        return True

    def set_head(self, pose: np.ndarray, duration: float = 1.0) -> None:
        if self._mini is not None:
            self._mini.goto_target(head=pose, duration=duration)

    def hold_head(self) -> None:
        self.set_head(head_pose(), duration=0.8)

    def look_around(self, step: int) -> None:
        self.set_head(head_pose(SCAN_YAWS_DEG[step % len(SCAN_YAWS_DEG)], pitch_deg=-5), duration=1.5)

    def wake(self) -> None:
        if self._mini is None:
            return
        if hasattr(self._mini, "enable_motors"):
            self._mini.enable_motors()
        self._mini.wake_up()

    def sleep(self) -> None:
        if self._mini is not None:
            self._mini.goto_sleep()


class VideoFileRobot:
    """Replays a video file as the camera (hardware-free testing); motion and audio are logged only."""

    def __init__(self, path: str, fps: float = 15.0, loop: bool = True):
        import cv2

        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            raise FileNotFoundError(f"cannot open video {path}")
        self._loop = loop
        self._interval = 1.0 / fps
        self._next_at = 0.0
        self.actions: list[tuple] = []

    def get_frame(self) -> np.ndarray | None:
        now = time.monotonic()
        if now < self._next_at:
            return None
        self._next_at = now + self._interval
        ok, frame = self._capture.read()
        if not ok and self._loop:
            import cv2

            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
        return crop_4_3(frame) if ok else None

    def play_clip(self, path: Path) -> bool:
        self.actions.append(("play_clip", Path(path).name))
        log.info("[video robot] play %s", path)
        return True

    def set_head(self, pose) -> None:
        self.actions.append(("set_head",))

    def hold_head(self) -> None:
        self.actions.append(("hold_head",))

    def look_around(self, step: int) -> None:
        self.actions.append(("look_around", step))

    def wake(self) -> None:
        self.actions.append(("wake",))

    def sleep(self) -> None:
        self.actions.append(("sleep",))

    def is_reachable(self) -> bool:
        return self._capture.isOpened()
