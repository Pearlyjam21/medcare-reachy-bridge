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

import httpx
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
DAEMON_PORT = 8000
MEDIA_STATUS_TIMEOUT = 2.0   # seconds: the camera watchdog logs the daemon's media status with every stall
RESTORE_RETRY_SECONDS = 60.0   # restore_media(): while the daemon refuses, ask again at most this often
RECORDING_RETRY_SECONDS = 5.0  # get_audio(): after the microphone failed to start, try again this often
LIVE_RATE = 16000   # the live chat's audio (bridge/live.py): the robot's own rate, so the SDK needs no resampling
# stop_live_audio(): the SDK still plays what the live chat pushed last, about 0.17 s through its pipeline plus its
# 50 ms sink buffer (audio_gstreamer.py PLAYBACK_SINK_BUFFER_TIME_US); the next clip waits that out.
LIVE_TAIL_SECONDS = 0.3
# The SDK keeps its playback timestamps continuous from one push to the next, unless a push comes more than this
# long after the end of the previous block (reachy_mini 1.8.3 audio_base.AudioBase.GAP_RESET_NS): then it starts the
# timeline afresh at the pipeline's running time. Blocks that fall behind by less than this stay behind for good,
# and the speaker drops every one of them once the lag passes its ~100 ms of latency. Measured on 13 Oct 2026
# through the ALSA loopback: after a 105-150 ms stall of the live chat's loop the rest of the chat was silent (the
# chats of 19:56 and 22:43 that day); after a 300 ms stall the SDK re-anchored and the sound came back.
# fill_live_audio() covers such a gap with silence, so the next block is on time again.
LIVE_PTS_RESYNC_SECONDS = 0.2
# The SDK's recorded moves (reachy_mini.motion.recorded_move.RecordedMoves): Hugging Face datasets the daemon keeps
# in its cache on the robot. "dance": 19 moves of 1.8-5 s (chicken_peck, jackson_square, side_to_side_sway...);
# "emotion": Pollen's 85 emotion moves of 2-20 s with their own sounds (downcast1, calming1, cheerful1...), played
# here without the sounds. The emotion cache held empty files on 13 Oct 2026 until it was downloaded again.
MOVE_LIBRARIES = {"dance": "pollen-robotics/reachy-mini-dances-library",
                  "emotion": "pollen-robotics/reachy-mini-emotions-library"}
DANCE_LIBRARY = MOVE_LIBRARIES["dance"]
DANCE_RATE_HZ = 100.0        # how often a move's trajectory is sent (the SDK's play_move plays at 100 Hz)
DANCE_START_SECONDS = 0.5    # the goto to a move's first pose before its trajectory starts


class Robot(Protocol):
    def get_frame(self) -> np.ndarray | None: ...        # BGR 640x480, or None when no new frame
    def get_camera_frame(self) -> np.ndarray | None: ...   # the camera's own BGR frame (any size), or None
    def get_audio(self) -> tuple[np.ndarray, int] | None: ...   # microphone samples since the last call, rate
    def play_clip(self, path: Path, wait: bool = True) -> bool: ...
    def set_head(self, pose: np.ndarray) -> None: ...
    def hold_head(self) -> None: ...
    def look_around(self, step: int) -> None: ...
    def look_at(self, u: float, v: float, duration: float) -> None: ...   # face centre in the streamed frame (0-1)
    def wake(self) -> None: ...
    def sleep(self) -> None: ...
    def is_reachable(self) -> bool: ...
    def restore_media(self) -> bool: ...   # ask for released media back (speaker, microphone, camera); never raises
    def gesture(self, mode: str | None) -> None: ...   # check-in body language: "think", "speak", None (still)
    # The live chat (bridge/live.py) owns the speaker from start_live_audio() to stop_live_audio(); no clip plays then.
    def start_live_audio(self) -> bool: ...   # False: no speaker now
    def push_audio(self, samples: np.ndarray) -> bool: ...   # one block, float32 mono at LIVE_RATE, not waited for
    def fill_live_audio(self) -> float: ...   # the chat's loop fell behind: silence for the gap; the seconds pushed
    def clear_live_audio(self) -> None: ...   # drop what the speaker still has queued
    def stop_live_audio(self) -> None: ...
    # The GPT-Live agent's dances and emotion moves (bridge/reachy_hardware.py): the SDK's recorded moves.
    def move_names(self, library: str) -> dict[str, str]: ...   # {name: description} the library has here; {}
    def play_move(self, library: str, name: str, should_stop) -> str: ...   # blocking: "completed" / "cancelled"
    def express(self, affect: str | None) -> None: ...   # a sustained posture under the gestures (gestures.AFFECTS)


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
        daemon_host = "127.0.0.1" if host == "localhost" else host
        self._media_status_url = f"http://{daemon_host}:{DAEMON_PORT}/api/media/status"
        self._lock = threading.Lock()   # audio: the SDK is driven from worker threads
        self._quiet_at = 0.0            # when the last clip handed to the speaker has finished playing
        self._restore_failing = False   # restore_media(): the daemon refused, already logged for this streak
        self._restore_retry_at = 0.0    # restore_media(): when a refused streak may ask the daemon again
        self._now = time.monotonic      # clock for restore_media()'s retry pace (tests replace it)
        self._recording_media = None    # get_audio(): the media manager whose microphone this app has started
        self._recording_retry_at = 0.0  # get_audio(): after a failed start, when to try again
        self._recording_failed = False  # get_audio(): a failed start is already logged
        # push_audio(): when the SDK's playback timeline (its continuous timestamps) expects the next block, on
        # self._now; None once start_playing() or clear_player() made the next push anchor it afresh
        self._live_next_at: float | None = None
        self._libraries: dict = {}      # the SDK's RecordedMoves by MOVE_LIBRARIES key, loaded on first use
        self._dancing = False           # play_move() runs: look_at() and gesture() leave the servos to it
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

    def _media(self):
        """The SDK's media manager now, or None: no connection, or this app's media is released (during a camera
        recovery, or after one whose acquire failed; the SDK would log a warning on every call then).

        acquire_media() replaces the manager, so it is looked up on every call, never kept.
        """
        if self._mini is None or getattr(self._mini, "media_released", False):
            return None
        return self._mini.media

    def get_camera_frame(self) -> np.ndarray | None:
        """The camera's 1280x720 BGR frame, or None when no new one came within ~20 ms.

        The daemon shares the camera with apps at its own rate: 10 fps on reachy_mini 1.11 (IPC_FPS), whatever
        this app asks for.
        """
        media = self._media()
        return None if media is None else media.get_frame()

    def get_audio(self) -> tuple[np.ndarray, int] | None:
        """The microphone samples received since the last call (float32) and their rate, or None."""
        media = self._media()
        if media is None:
            return None
        self._ensure_recording(media)
        samples = media.get_audio_sample()
        if samples is None or len(samples) == 0:
            return None
        return np.asarray(samples, dtype=np.float32), int(media.get_input_audio_samplerate())

    def _ensure_recording(self, media) -> None:
        """Start the SDK's microphone pipeline once per media manager (acquire_media() builds a new one).

        Reachy Mini SDK 1.8.3, which the robot came back with after its reset on 12 Oct 2026, keeps the microphone
        in a pipeline of its own that runs only after start_recording(), and nothing else calls it: every read was
        empty, so neither 「我吃完了」 nor a check-in was ever heard. Starting it again is harmless (it only sets the
        pipeline playing). A failed start is logged once and tried again after RECORDING_RETRY_SECONDS."""
        if self._recording_media is media:
            return
        now = self._now()
        if now < self._recording_retry_at:
            return
        try:
            media.start_recording()
        except Exception:
            if not self._recording_failed:
                log.exception("the microphone did not start; trying again every %.0f s", RECORDING_RETRY_SECONDS)
            self._recording_failed = True
            self._recording_retry_at = now + RECORDING_RETRY_SECONDS
            return
        if self._recording_failed:
            log.info("the microphone started")
        self._recording_media, self._recording_failed, self._recording_retry_at = media, False, 0.0

    def play_clip(self, path: Path, wait: bool = True) -> bool:
        """Play a WAV, blocking for its length; `wait=False` returns as soon as it is handed to the speaker.

        A clip never starts while the previous one is still playing: the SDK's start_playing() restarts the
        stream's timestamps, so overlapping clips would clip or mix.
        """
        media = self._media()
        if media is None:
            return False
        samples, rate = read_wav(Path(path))
        out_rate = int(media.get_output_audio_samplerate())
        samples = resample(samples, rate, out_rate)
        seconds = len(samples) / out_rate
        with self._lock:
            time.sleep(max(0.0, self._quiet_at - time.monotonic()))
            media = self._media()   # a camera recovery may have replaced the manager while this waited
            if media is None:
                return False
            media.start_playing()
            self._live_next_at = None   # start_playing() anchors the SDK's timeline afresh at the next push
            for start in range(0, len(samples), self.CHUNK):
                media.push_audio_sample(samples[start:start + self.CHUNK])
            self._quiet_at = time.monotonic() + seconds
        if wait:
            # Playback is asynchronous in the SDK; keep the robot quiet for the clip's length.
            time.sleep(seconds)
        return True

    # ── the live chat's speaker (bridge/live.py) ─────────────────────────
    def start_live_audio(self) -> bool:
        """Start the speaker for a live chat, which owns it until stop_live_audio(): the SDK's start_playing() once,
        after any clip still playing, then push_audio() blocks paced by the chat. False when this app has no media."""
        with self._lock:
            time.sleep(max(0.0, self._quiet_at - time.monotonic()))   # never cut a clip short
            media = self._media()
            if media is None:
                return False
            media.start_playing()
            self._live_next_at = None
        return True

    def _follow_timeline(self, seconds: float) -> None:
        """Keep step with the SDK's playback timeline across a push of `seconds` of audio: it continues from the
        previous block unless the push comes LIVE_PTS_RESYNC_SECONDS after that block's end (then it starts afresh,
        at now)."""
        now = self._now()
        if self._live_next_at is None or now - self._live_next_at > LIVE_PTS_RESYNC_SECONDS:
            self._live_next_at = now
        self._live_next_at += seconds

    def push_audio(self, samples: np.ndarray) -> bool:
        """One block of the live chat's audio (float32 mono at LIVE_RATE), handed to the speaker without waiting: the
        SDK plays pushed blocks back to back, so the caller paces them. False when this app has no media."""
        with self._lock:
            media = self._media()
            if media is None:
                return False
            samples = np.asarray(samples, dtype=np.float32)
            out_rate = int(media.get_output_audio_samplerate())
            self._follow_timeline(len(samples) / LIVE_RATE)
            media.push_audio_sample(resample(samples, LIVE_RATE, out_rate))   # 16 kHz on the robot: no change
        return True

    def fill_live_audio(self) -> float:
        """The live chat's loop fell behind its clock (live.py LATE_RESET_SECONDS): push silence for the time the
        SDK's timeline is behind now, so the next block is on time again, and return those seconds. 0.0 when the
        timeline is not behind, or behind by more than LIVE_PTS_RESYNC_SECONDS (the SDK starts it afresh by itself at
        the next push), or without media."""
        with self._lock:
            media = self._media()
            if media is None or self._live_next_at is None:
                return 0.0
            lag = self._now() - self._live_next_at
            if lag <= 0.0 or lag > LIVE_PTS_RESYNC_SECONDS:
                return 0.0
            count = int(lag * LIVE_RATE)
            out_rate = int(media.get_output_audio_samplerate())
            media.push_audio_sample(resample(np.zeros(count, np.float32), LIVE_RATE, out_rate))
            self._live_next_at += count / LIVE_RATE
            return count / LIVE_RATE

    def clear_live_audio(self) -> None:
        """Drop what the speaker still has queued (the live chat's flush), with the SDK's clear_player(). In reachy_mini
        1.11.0 that call is on the media manager's audio object (the manager has none): GStreamerAudio pauses its one
        pipeline, flushes the player and plays again (and anchors its timeline afresh). Does nothing without media."""
        with self._lock:
            media = self._media()
            clear = getattr(getattr(media, "audio", None), "clear_player", None)
            if clear is not None:
                clear()
                self._live_next_at = None

    def stop_live_audio(self) -> None:
        """The live chat is over. Never the SDK's stop_playing(): GStreamerAudio keeps the microphone and the speaker in
        one pipeline, which stop_playing() sets to NULL, and the speech listener would then hear nothing. The next clip
        starts the playback itself (play_clip) and waits LIVE_TAIL_SECONDS for the audio the chat pushed last."""
        with self._lock:
            self._quiet_at = max(self._quiet_at, time.monotonic() + LIVE_TAIL_SECONDS)

    def media_status(self) -> dict | None:
        """The daemon's GET /api/media/status ({"available", "released", "no_media"}), or None on any error."""
        try:
            response = httpx.get(self._media_status_url, timeout=MEDIA_STATUS_TIMEOUT)
            response.raise_for_status()
            status = response.json()
        except Exception as exc:   # the daemon busy or gone, or not JSON: the status is only logged
            log.warning("daemon media status unavailable: %r", exc)
            return None
        return status if isinstance(status, dict) else None

    def reopen_camera(self) -> str:
        """Get the camera feed back after it stalled (the camera watchdog's recovery, spec 2026-10-11 A2).

        Uses the SDK's public media calls (reachy_mini 1.11.0, reachy_mini.py:203-240):
        - release_media() closes this app's media manager and marks it released; it also asks the daemon to release
          its media, which the daemon ignores when it already has (daemon.py:145-146), e.g. after the TCU companion
          released it;
        - acquire_media() asks the daemon for its media back (daemon.py:155-169 restarts its pipeline) and builds a
          fresh media manager. On its own it does nothing unless this app's media is marked released, hence the
          release first.
        Returns "acquired" (the daemon had released its media, or lost it), "reopened" (the daemon had it: the cycle
        renewed this app's reader) or "failed: <reason>". Never raises.

        The media manager holds the speaker and the microphone too, so this waits for a clip still playing, holds the
        playback lock throughout, and starts the fresh manager's audio pipeline, which every clip used to leave
        running (play_clip and get_audio look the manager up on every call).
        """
        mini = self._mini
        if mini is None:
            return "failed: no robot connection"
        try:
            with self._lock:
                time.sleep(max(0.0, self._quiet_at - time.monotonic()))   # never cut a clip short
                return self._renew_media(mini)
        except Exception as exc:   # a failed attempt: the watchdog tries again later
            log.exception("camera recovery failed")
            return f"failed: {exc!r}"

    def _renew_media(self, mini) -> str:
        """reopen_camera's release/acquire cycle, with the playback lock held."""
        status = None
        if getattr(mini, "media_released", False):
            outcome = "acquired"   # an earlier attempt released this app's media and the daemon kept it: ask again
        else:
            status = self.media_status()
            if status is not None and status.get("no_media"):
                return "failed: the daemon runs without media"
            daemon_has_it = status is None or (status.get("available") and not status.get("released"))
            outcome = "reopened" if daemon_has_it else "acquired"
            mini.release_media()
        mini.acquire_media()
        if getattr(mini, "media_released", False):
            return "failed: the daemon did not give the media back"
        try:
            mini.media.start_recording()
        except Exception:   # the camera is back; the microphone restarts with the next clip anyway
            log.exception("the microphone did not restart after the camera recovery")
        # The fresh manager is LOCAL when the daemon's camera socket is there; a "webrtc" here means it wasn't yet.
        log.info("camera recovery: %s with the SDK's release_media() and acquire_media() (daemon media status %s, "
                 "media backend %s)", outcome, status, getattr(mini.media, "backend", None))
        return outcome

    def media_released(self) -> bool:
        """Whether the SDK has this app's media released: camera, speaker and microphone all off (a camera recovery
        whose acquire the daemon refused). The camera watchdog reads it when it gives up."""
        return self._mini is not None and bool(getattr(self._mini, "media_released", False))

    def restore_media(self) -> bool:
        """Ask for this app's media back when the SDK says it is released; returns whether this app has it now.

        A camera recovery releases the whole media manager, the speaker and the microphone with the camera. When the
        daemon refused to give it back, only an attached, stalled session's watchdog used to ask again, so after a
        give-up Reachy stayed mute and deaf (final review of 11 Oct). The runner calls this at every heartbeat and
        before a task wakes the robot. It only ever calls the SDK's acquire_media(), never release_media(), so media
        this app has is never touched; it holds the playback lock, as the recovery does, and starts the fresh
        manager's microphone. Never raises: a refusal or an SDK error logs one WARNING per streak of failures.
        While the daemon keeps refusing it asks again at most once every RESTORE_RETRY_SECONDS: the SDK logs a WARNING
        and an ERROR for each refused acquire and its request can take 10 s, so asking at every heartbeat filled the
        robot's log and delayed the heartbeat.
        """
        mini = self._mini
        if mini is None:
            return False
        if not getattr(mini, "media_released", False):
            self._restore_failing = False
            return True
        if self._restore_failing and self._now() < self._restore_retry_at:
            return False   # refused a moment ago: wait before asking the daemon again
        failure, restored = None, False
        try:
            with self._lock:
                if getattr(mini, "media_released", False):   # a camera recovery may have got it back meanwhile
                    mini.acquire_media()
                    if getattr(mini, "media_released", False):
                        failure = "the daemon did not give the media back"
                    else:
                        restored = True
                        try:
                            mini.media.start_recording()
                        except Exception:   # the speaker works again; only the microphone is missing
                            log.exception("the microphone did not restart after the media came back")
        except Exception as exc:   # asked again at the next heartbeat
            failure = repr(exc)
        if failure is not None:
            if not self._restore_failing:
                log.warning("this app's media is still released (%s): the speaker and microphone stay off; asking "
                            "again about once a minute", failure)
            self._restore_failing = True
            self._restore_retry_at = self._now() + RESTORE_RETRY_SECONDS
            return False
        self._restore_failing = False
        if restored:
            log.info("this app's media is back with the SDK's acquire_media(): camera, speaker and microphone")
        return True

    def gesture(self, mode: str | None) -> None:
        """Check-in body language (gestures.py): "think", "speak", or None to come back to the neutral pose.

        Never blocks. Ignored until hold_head() has put the robot in the neutral pose; any other move (set_head,
        look_around, wake, sleep, close) stops it first and ignores it again until the next hold_head(). Ignored
        while a recorded move plays (play_move ends with hold_head).
        """
        if self._mini is not None and not self._dancing:
            self._gestures.set(mode)

    def express(self, affect: str | None) -> None:
        """A sustained posture under the gestures (gestures.set_affect: "concern", "joy", or None for none), from
        the live chat's `express` message. Never blocks; it fades in once the robot is placed in the neutral pose,
        and stays through gestures and recorded moves until lifted."""
        if self._mini is not None:
            self._gestures.set_affect(affect)

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

    def goto(self, pose: np.ndarray, antennas_deg: tuple[float, float] = (0.0, 0.0), duration: float = 0.5) -> None:
        """One move of the head to `pose` (head_pose; its [:3, 3] is the position in metres) and of the antennas to
        offsets from the neutral antennas in degrees ([right, left], as the gesture stream's Pose), over `duration`
        seconds. Blocks until the move has ended (the SDK's goto_target), so call it from a worker thread
        (reachy_hardware.py's motion worker). Stops the gesture stream like set_head; hold_head() lets it start
        again from the neutral pose."""
        if self._mini is not None:
            self._gestures.halt()
            antennas = [NEUTRAL_ANTENNAS[0] + math.radians(antennas_deg[0]),
                        NEUTRAL_ANTENNAS[1] + math.radians(antennas_deg[1])]
            self._mini.goto_target(head=pose, antennas=antennas, duration=max(0.2, float(duration)), body_yaw=0.0)

    def hold_head(self) -> None:
        self.set_head(head_pose(), duration=0.8)
        if self._mini is not None:
            self._gestures.placed()   # the goto has ended in the neutral pose: gestures may start from it

    def look_around(self, step: int) -> None:
        self.set_head(head_pose(SCAN_YAWS_DEG[step % len(SCAN_YAWS_DEG)], pitch_deg=-5), duration=1.5)

    def look_at(self, u: float, v: float, duration: float = 0.35) -> None:
        """Turn the head towards a point of the streamed frame (normalised 0-1, see runner.encode_stream_jpeg):
        the server's face box is in that 4:3 centre crop, the SDK's look_at_image wants a pixel of the camera's
        own image."""
        media = self._media()
        if self._mini is None or media is None or self._dancing:   # a dance has the servos until it ends
            return
        camera = getattr(media, "camera", None)
        width, height = (getattr(camera, "resolution", None) or (1280, 720))[:2]
        crop_width = min(width, int(height * 4 / 3))
        x0 = (width - crop_width) // 2
        px = int(round(x0 + min(1.0, max(0.0, u)) * crop_width))
        py = int(round(min(1.0, max(0.0, v)) * height))
        px, py = min(width - 1, max(1, px)), min(height - 1, max(1, py))
        self._gestures.halt()
        self._mini.look_at_image(px, py, duration=duration)

    # ── the GPT-Live agent's dances and emotion moves (bridge/reachy_hardware.py) ──────────
    def _library(self, library: str):
        """The SDK's recorded moves of one MOVE_LIBRARIES entry, loaded once from the Hugging Face cache the daemon
        keeps on the robot (RecordedMoves asks the network only when the cache lacks it)."""
        if library not in MOVE_LIBRARIES:
            raise ValueError(f"unknown move library {library}")
        if library not in self._libraries:
            from reachy_mini.motion.recorded_move import RecordedMoves   # lazy: the SDK and huggingface_hub

            moves = RecordedMoves(MOVE_LIBRARIES[library])
            valid = []
            for name in moves.list_moves():   # a half-downloaded cache holds empty files: those are left out
                try:
                    moves.get(name)
                    valid.append(name)
                except Exception:
                    log.warning("[MOVES] %s: %s is unreadable in the cache", library, name)
            moves.moves = {name: moves.moves[name] for name in valid}
            self._libraries[library] = moves
            log.info("[MOVES] library %s (%s): %d moves", library, MOVE_LIBRARIES[library], len(valid))
        return self._libraries[library]

    def move_names(self, library: str) -> dict[str, str]:
        """The recorded moves this robot has in `library` ("dance" or "emotion"), name -> the library's one-line
        description; {} when it cannot be loaded (logged). Blocks on the first call (read from disk)."""
        try:
            moves = self._library(library)
        except Exception as exc:
            log.warning("[MOVES] the %s library is not available: %r", library, exc)
            return {}
        return {name: str((moves.moves.get(name) or {}).get("description") or "") for name in moves.list_moves()}

    def dance_names(self) -> dict[str, str]:
        return self.move_names("dance")

    def play_dance(self, name: str, should_stop) -> str:
        return self.play_move("dance", name, should_stop)

    def play_move(self, library: str, name: str, should_stop) -> str:
        """Play one recorded move, blocking for its length (call it from reachy_hardware's motion worker): the
        gesture stream stops, the head goes to the move's first pose (DANCE_START_SECONDS), then its trajectory is
        sent at DANCE_RATE_HZ with the SDK's set_target, as the SDK's own play_move does (without the move's sound:
        GPT-Live's voice is the sound), until it ends or `should_stop()` says so. Face following and gestures leave
        the servos alone meanwhile; it ends with hold_head(), the neutral pose, so they may start again (a posture
        set with express() fades in again then). Returns "completed" or "cancelled". ValueError for a name the
        library lacks, RuntimeError without a connection."""
        if self._mini is None:
            raise RuntimeError("no_robot")
        move = self._library(library).get(name)   # ValueError when unknown
        self._gestures.halt()
        self._dancing = True
        outcome = "cancelled"
        try:
            head, antennas, body_yaw = move.evaluate(0.0)
            self._mini.goto_target(head=head, antennas=list(antennas), duration=DANCE_START_SECONDS,
                                   body_yaw=float(body_yaw))
            period = 1.0 / DANCE_RATE_HZ
            started = time.monotonic()
            while not should_stop():
                elapsed = time.monotonic() - started
                if elapsed >= move.duration:
                    outcome = "completed"
                    break
                head, antennas, body_yaw = move.evaluate(min(elapsed, move.duration - 1e-2))
                self._mini.set_target(head=head, antennas=list(antennas), body_yaw=float(body_yaw))
                time.sleep(max(0.001, period - (time.monotonic() - started - elapsed)))
        finally:
            self._dancing = False
            self.hold_head()
        return outcome

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

    def restore_media(self) -> bool:
        return True   # a video file has no media to give back

    def play_clip(self, path: Path, wait: bool = True) -> bool:
        self.actions.append(("play_clip", Path(path).name))
        log.info("[video robot] play %s", path)
        return True

    def start_live_audio(self) -> bool:
        return False   # no microphone to stream: a live chat falls back to the text chat

    def push_audio(self, samples: np.ndarray) -> bool:
        return False

    def fill_live_audio(self) -> float:
        return 0.0

    def clear_live_audio(self) -> None:
        pass

    def stop_live_audio(self) -> None:
        pass

    def move_names(self, library: str) -> dict[str, str]:
        return {}   # no move library without the SDK

    def play_move(self, library: str, name: str, should_stop) -> str:
        self.actions.append(("play_move", library, name))
        return "completed"

    def express(self, affect: str | None) -> None:
        self.actions.append(("express", affect))

    def gesture(self, mode: str | None) -> None:
        self.actions.append(("gesture", mode))

    def set_head(self, pose, duration: float = 1.0) -> None:
        self.actions.append(("set_head",))

    def goto(self, pose, antennas_deg=(0.0, 0.0), duration: float = 0.5) -> None:
        self.actions.append(("goto", tuple(antennas_deg), duration))

    def hold_head(self) -> None:
        self.actions.append(("hold_head",))

    def look_around(self, step: int) -> None:
        self.actions.append(("look_around", step))

    def look_at(self, u: float, v: float, duration: float = 0.35) -> None:
        self.actions.append(("look_at", round(u, 2), round(v, 2)))

    def wake(self) -> None:
        self.actions.append(("wake",))

    def sleep(self) -> None:
        self.actions.append(("sleep",))

    def is_reachable(self) -> bool:
        return self._capture.isOpened()
