"""Robot facade: camera frames, prerecorded audio, head motion, check-in gestures.

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

from medcare_reachy.bridge import moves
from medcare_reachy.bridge.gestures import RECENTER_SECONDS, Gestures, Pose

log = logging.getLogger(__name__)

FRAME_WIDTH, FRAME_HEIGHT = 640, 480
SCAN_YAWS_DEG = (0, 25, 40, 25, 0, -25, -40, -25)   # gentle sweep while searching
NEUTRAL_ANTENNAS = (-0.1745, 0.1745)   # rad, [right, left]: reachy_mini's INIT_ANTENNAS_JOINT_POSITIONS (~10° off)
# Where a gesture starts from, by how far the measured pose is from the neutral pose (`_start_pose`, which logs
# each measurement as "gesture from the pose measured ... off neutral"):
# - within NEUTRAL_TOLERANCE, from the neutral pose itself. Idle near it on 3 Oct 2026 (motors on), the robot
#   measured its head 3.6° and 4.4 mm off, its body 0.2° and its antennas 0.6°.
# - within RECENTER_LIMIT and with the motors on, from the pose measured, easing back to the neutral pose over
#   gestures.RECENTER_SECONDS. At the check-in of 3 Oct 14:53 the head measured 8.6° (3.8 mm) off, and the single
#   8° limit of 0.5.2 kept Reachy still for the whole conversation.
# - further, or with the motors off or in an unknown state: none until the next goto to the neutral pose. A head
#   that fell with the motors off is up to ~24° and ~44 mm off (the sleep pose); so is a head someone holds.
# Provisional (the robot's journal could not be read): re-derive both from those log lines.
NEUTRAL_TOLERANCE = {"head_deg": 4.5, "head_mm": 6.0, "body_deg": 1.5, "antenna_deg": 3.0}
RECENTER_LIMIT = {"head_deg": 15.0, "head_mm": 20.0, "body_deg": 8.0, "antenna_deg": 30.0}


class Robot(Protocol):
    def get_frame(self) -> np.ndarray | None: ...        # BGR 640x480, or None when no new frame
    def get_camera_frame(self) -> np.ndarray | None: ...   # the camera's own BGR frame (any size), or None
    def get_audio(self) -> tuple[np.ndarray, int] | None: ...   # microphone samples since the last call, rate
    def play_clip(self, path: Path, wait: bool = True) -> bool: ...
    def set_head(self, pose: np.ndarray) -> None: ...
    def hold_head(self) -> None: ...
    def look_around(self, step: int) -> None: ...
    def wake(self) -> None: ...
    def sleep(self) -> None: ...
    def is_reachable(self) -> bool: ...
    def gesture(self, mode: str | None) -> None: ...   # check-in body language: "think", "speak", None (still)


def crop_4_3(frame: np.ndarray) -> np.ndarray:
    """Centre-crop a wide frame to 4:3 and resize to 640x480 (spec 00 §6 M2)."""
    height, width = frame.shape[:2]
    target_width = int(height * 4 / 3)
    if width > target_width:
        x = (width - target_width) // 2
        frame = frame[:, x:x + target_width]
    if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
        from PIL import Image   # Pillow, not OpenCV: nothing on the robot needs OpenCV

        resized = Image.fromarray(np.ascontiguousarray(frame)).resize(
            (FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.BOX)
        frame = np.asarray(resized)
    return frame


def head_pose(yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """4x4 head pose, same convention as reachy_mini.utils.create_head_pose (extrinsic xyz, degrees).

    Positive pitch looks down, positive roll tilts the head to the robot's left.
    """
    yaw, pitch, roll = math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg)
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    pose = np.eye(4)
    pose[:3, :3] = rz @ ry @ rx
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


def thinking_move():
    """Pollen's "inquiring3", which Reachy's thinking starts with (gestures.py); None (the sway alone) when it can't
    be loaded."""
    try:
        return moves.load(moves.THINKING_MOVE)
    except Exception as exc:   # a missing or altered file costs the move, never the gestures
        log.warning("thinking move unavailable, Reachy only sways while thinking: %r", exc)
        return None


def _beyond(off: dict, limits: dict) -> dict:
    """The measured offsets over their limit, rounded for the log."""
    return {name: round(value, 1) for name, value in off.items() if value > limits[name]}


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


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
        self._lock = threading.Lock()   # audio: the SDK is driven from worker threads
        self._quiet_at = 0.0            # when the last clip handed to the speaker has finished playing
        # Gestures start only after hold_head() and stop for every other move (wake, sleep, look_around, close).
        self._gestures = Gestures(self._send_gesture, ready=self._motors_on, start_from=self._start_pose,
                                  move=thinking_move())

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
        self._gestures.halt()   # a bridge that stops must not leave a gesture streaming
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
        frame = self.get_camera_frame()
        return None if frame is None else crop_4_3(frame)

    def get_camera_frame(self) -> np.ndarray | None:
        """The camera's 1280x720 BGR frame, or None when no new one came within ~20 ms.

        The daemon shares the camera with apps at its own rate: 10 fps on reachy_mini 1.11 (IPC_FPS), whatever
        this app asks for.
        """
        if self._mini is None:
            return None
        return self._mini.media.get_frame()

    def get_audio(self) -> tuple[np.ndarray, int] | None:
        """The microphone samples received since the last call (float32) and their rate, or None."""
        if self._mini is None:
            return None
        media = self._mini.media
        samples = media.get_audio_sample()
        if samples is None or len(samples) == 0:
            return None
        return np.asarray(samples, dtype=np.float32), int(media.get_input_audio_samplerate())

    def play_clip(self, path: Path, wait: bool = True) -> bool:
        """Play a WAV, blocking for its length; `wait=False` returns as soon as it is handed to the speaker.

        A clip never starts while the previous one is still playing: the SDK's start_playing() restarts the
        stream's timestamps, so overlapping clips would clip or mix.
        """
        if self._mini is None:
            return False
        samples, rate = read_wav(Path(path))
        media = self._mini.media
        out_rate = int(media.get_output_audio_samplerate())
        samples = resample(samples, rate, out_rate)
        seconds = len(samples) / out_rate
        with self._lock:
            time.sleep(max(0.0, self._quiet_at - time.monotonic()))
            media.start_playing()
            for start in range(0, len(samples), self.CHUNK):
                media.push_audio_sample(samples[start:start + self.CHUNK])
            self._quiet_at = time.monotonic() + seconds
        if wait:
            # Playback is asynchronous in the SDK; keep the robot quiet for the clip's length.
            time.sleep(seconds)
        return True

    def gesture(self, mode: str | None) -> None:
        """Check-in body language (gestures.py): "think", "speak", or None to come back to the neutral pose.

        Never blocks. Ignored until hold_head() has put the robot in the neutral pose; any other move (set_head,
        look_around, wake, sleep, close) stops it first and ignores it again until the next hold_head().
        """
        if self._mini is not None:
            self._gestures.set(mode)

    def _send_gesture(self, pose: Pose) -> None:
        head = head_pose(pose.yaw, pose.pitch, pose.roll)
        head[:3, 3] = (pose.x / 1000, pose.y / 1000, pose.z / 1000)
        self._mini.set_target(head=head,
                              antennas=[NEUTRAL_ANTENNAS[0] + math.radians(pose.right),
                                        NEUTRAL_ANTENNAS[1] + math.radians(pose.left)],
                              body_yaw=math.radians(pose.body_yaw))

    def _motor_mode(self) -> str | None:
        """The motors' mode in the daemon's last status (pushed every second): "enabled", "disabled",
        "gravity_compensation"; None when unknown."""
        try:
            mode = self._mini.client.get_status(wait=False).backend_status.motor_control_mode
        except Exception:
            return None
        return str(getattr(mode, "value", mode))

    def _motors_on(self) -> bool:
        """Whether the daemon's last status has the motors on; unknown counts as on."""
        mode = self._motor_mode()
        return mode is None or mode == "enabled"

    def _start_pose(self) -> Pose | None:
        """Where a gesture's first target must be, from the robot's measured pose (the daemon pushes it every control
        tick): the neutral pose when the robot is near it; the pose measured (offsets from the neutral pose), which
        the stream eases back from, when it is a little further and the motors are on; None when it is further still
        or the motors are not known to be on.

        A target is sent straight to the servos: a first target away from where the robot is would make it jump.
        """
        head = np.asarray(self._mini.get_current_head_pose(), dtype=float)
        joints, antennas = self._mini.get_current_joint_positions()
        antenna_offsets = [math.degrees(a - n) for a, n in zip(antennas, NEUTRAL_ANTENNAS)]
        off = {"head_deg": math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(head[:3, :3]) - 1) / 2)))),
               "head_mm": float(np.linalg.norm(head[:3, 3])) * 1000,
               "body_deg": abs(math.degrees(joints[0])),   # the head's first joint is the body's rotation
               "antenna_deg": max(abs(offset) for offset in antenna_offsets)}
        if not _beyond(off, NEUTRAL_TOLERANCE):
            start, note = Pose(), ""
        elif away := _beyond(off, RECENTER_LIMIT):
            start, note = None, f"; too far: {away}"
        elif (motors := self._motor_mode()) != "enabled":
            start, note = None, f"; motors {motors or 'unknown'}: not easing back"
        else:
            yaw, pitch, roll = moves.head_angles(head)
            x, y, z = (float(value) * 1000 for value in head[:3, 3])
            start = Pose(roll, pitch, yaw, math.degrees(joints[0]), *antenna_offsets, x, y, z)
            note = f"; easing back from there over {RECENTER_SECONDS} s"
        log.info("gesture from the pose measured %s off neutral%s", {k: round(v, 1) for k, v in off.items()}, note)
        return start

    def set_head(self, pose: np.ndarray, duration: float = 1.0) -> None:
        if self._mini is not None:
            # While a goto runs the daemon ignores targets, so the gesture stream stops first; the goto then
            # also brings the antennas and the body back to neutral.
            self._gestures.halt()
            self._mini.goto_target(head=pose, antennas=list(NEUTRAL_ANTENNAS), duration=duration, body_yaw=0.0)

    def hold_head(self) -> None:
        self.set_head(head_pose(), duration=0.8)
        if self._mini is not None:
            self._gestures.placed()   # the goto has ended in the neutral pose: gestures may start from it

    def look_around(self, step: int) -> None:
        self.set_head(head_pose(SCAN_YAWS_DEG[step % len(SCAN_YAWS_DEG)], pitch_deg=-5), duration=1.5)

    def wake(self) -> None:
        if self._mini is None:
            return
        self._gestures.halt()
        if hasattr(self._mini, "enable_motors"):
            self._mini.enable_motors()
        self._mini.wake_up()

    def sleep(self) -> None:
        if self._mini is not None:
            self._gestures.halt()   # also from another thread, no gesture starts until the next hold_head()
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
        frame = self.get_camera_frame()
        return None if frame is None else crop_4_3(frame)

    def get_camera_frame(self) -> np.ndarray | None:
        now = time.monotonic()
        if now < self._next_at:
            return None
        self._next_at = now + self._interval
        ok, frame = self._capture.read()
        if not ok and self._loop:
            import cv2

            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
        return frame if ok else None

    def get_audio(self) -> None:
        return None   # no microphone when replaying a file

    def play_clip(self, path: Path, wait: bool = True) -> bool:
        self.actions.append(("play_clip", Path(path).name))
        log.info("[video robot] play %s", path)
        return True

    def gesture(self, mode: str | None) -> None:
        self.actions.append(("gesture", mode))

    def set_head(self, pose, duration: float = 1.0) -> None:
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
