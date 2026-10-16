"""Non-blocking Reachy Mini hardware for the GPT-Live agent layer: camera, microphone, speaker, head and antennas.

Adapted from the reference reachy_gpt_live_agent/reachy_hardware.py. The SDK is never called from here: every call
goes through the ReachyRobot facade (media.py), which already holds the measured SDK 1.8.3 behaviour (the microphone
runs only after start_recording(), one playback pipeline for the speaker and the microphone, the gesture stream that
a goto must stop first). What this module adds is the isolation the reference describes:
  * the camera on its own single worker (a capture never runs on the asyncio loop);
  * the emotes on one serialized motion worker, so two gestures never fight over the servos and a tool call returns
    at once with "queued";
  * microphone and speaker conversions (float32 <-> PCM16, resampling with media.resample, so no scipy) off the loop.

During a live chat the LiveChat (live.py) owns the microphone and the speaker; it uses this adapter only for the
server's tool requests: `camera` (a fresh frame as JPEG), `trigger_antenna_emote` (an EMOTE_PRESETS gesture),
`dance` / `stop_dance` (one of the SDK's recorded dances, media.MOVE_LIBRARIES, on the same motion worker; a stop
event ends it within a trajectory step) and `express` (one of Pollen's recorded emotion moves, the same way; the
sustained posture that comes with it is the gesture stream's, media.Robot.express). The audio methods are for an
agent that runs the whole GPT-Live session on the robot itself.

Emotes are short gotos (ReachyRobot.goto: head_pose plus antenna offsets in degrees, [right, left]) that end with
hold_head(), which brings the robot back to the neutral pose and lets the check-in gesture stream start again. The
angles stay modest: within gestures.LIMITS for the head and a little beyond it for the antennas (the recorded
thinking move swings the left antenna to 100°, MOVE_LIMITS). Positive pitch looks down (media.head_pose).
"""

import asyncio
import logging
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from medcare_reachy.bridge.media import LIVE_RATE, head_pose, resample
from medcare_reachy.bridge.runner import encode_jpeg

log = logging.getLogger(__name__)

EMOTE_PRESETS = ("applaud_cheer", "sympathy_droop_warm", "curious_tilt", "alert_attention", "listening_attentive",
                 "calming_sway")
MAX_PENDING_EMOTES = 3        # queued gestures beyond this are refused (the motion worker is busy)
MAX_DANCE_REPEAT = 4          # a dance tool call plays its move at most this many times in a row
CAPTURE_TRIES = 10            # get_frame() answers None when no new frame came within ~20 ms: ask again this often
CAPTURE_RETRY_SECONDS = 0.05  # ... after this long (the daemon shares the camera at 10 fps)
HEAD_LIMIT_DEG = 12.0         # an emote's head angles are clamped to this ...
ANTENNA_LIMIT_DEG = 40.0      # ... and its antenna offsets to this
Z_LIMIT_MM = 5.0


class HardwareError(RuntimeError):
    """Reachy's hardware did not do what was asked (the camera gave no frame, an unknown emote, the worker is busy)."""


@dataclass
class CapturedFrame:
    """One BGR camera frame (640x480, media.crop_4_3) with when it was taken (time.monotonic) and its size."""
    image: np.ndarray
    captured_at_monotonic: float
    width: int
    height: int


class ReachyMiniHardware:
    """Own the camera and motion workers over a ReachyRobot (or any media.Robot with goto and get_frame).

    `robot` is the facade; `clock` the monotonic clock (tests replace it); `run_blocking` runs a robot call off the
    loop (asyncio.to_thread; tests run it inline). Nothing here connects or disconnects the SDK: the app runtime
    owns that (main.py, ReachyRobot.attach)."""

    def __init__(self, robot, *, clock=time.monotonic, run_blocking=asyncio.to_thread):
        self.robot = robot
        self.clock = clock
        self._run_blocking = run_blocking
        self._camera_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reachy-camera")
        self._motion_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reachy-motion")
        self._motion_guard = threading.Lock()
        self._motion_futures: set[Future] = set()
        self._dance: tuple[Future, threading.Event, str] | None = None   # the dance running or queued, its stop
        self._last_dance: str | None = None                               # event and its name
        self._closed = False

    # ── camera ───────────────────────────────────────────────────────────
    def _capture_sync(self) -> CapturedFrame:
        frame = None
        for attempt in range(CAPTURE_TRIES):
            frame = self.robot.get_frame()
            if frame is not None:
                break
            if attempt + 1 < CAPTURE_TRIES:
                time.sleep(CAPTURE_RETRY_SECONDS)
        if frame is None:
            raise HardwareError("no_camera_frame")
        image = np.asarray(frame)
        if image.ndim != 3:
            raise HardwareError(f"unexpected_frame_shape:{image.shape}")
        height, width = image.shape[:2]
        return CapturedFrame(image=image.copy(), captured_at_monotonic=self.clock(), width=int(width),
                             height=int(height))

    async def capture_frame(self) -> CapturedFrame:
        """A fresh camera frame, taken on the camera worker; HardwareError when the camera gives none."""
        if self._closed:
            raise HardwareError("hardware_closed")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._camera_executor, self._capture_sync)

    async def capture_jpeg(self) -> tuple[bytes, CapturedFrame]:
        """A fresh frame as JPEG bytes (runner.encode_jpeg) plus the frame, both made off the loop."""
        frame = await self.capture_frame()
        jpeg = await self._run_blocking(encode_jpeg, frame.image)
        return jpeg, frame

    # ── microphone and speaker (an agent running on the robot; the live chat has its own loops) ──────────────
    async def start_audio(self) -> bool:
        return bool(await self._run_blocking(self.robot.start_live_audio))

    async def stop_audio(self) -> None:
        await self._run_blocking(self.robot.stop_live_audio)

    async def read_input_pcm16(self, target_rate: int = LIVE_RATE) -> bytes:
        """The microphone samples since the last call as mono PCM16 at `target_rate`; b"" when none came yet."""
        chunk = await self._run_blocking(self.robot.get_audio)
        if chunk is None:
            return b""
        samples, rate = chunk

        def convert() -> bytes:
            mono = np.asarray(samples, dtype=np.float32)
            if mono.ndim == 2:
                mono = mono[:, 0]   # both channels carry the XVF3800 beam (live.mic_pcm)
            if int(rate) != target_rate:
                mono = resample(mono, int(rate), target_rate)
            return (np.clip(mono, -1.0, 32767 / 32768) * 32768).astype("<i2").tobytes()

        return await self._run_blocking(convert)

    async def enqueue_output_pcm16(self, pcm: bytes, source_rate: int = LIVE_RATE) -> bool:
        """Queue mono PCM16 for the speaker (ReachyRobot.push_audio takes float32 at LIVE_RATE, not waited for)."""
        if not pcm:
            return True

        def convert() -> np.ndarray:
            samples = np.frombuffer(pcm[:len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float32) / 32768
            if source_rate != LIVE_RATE:
                samples = resample(samples, source_rate, LIVE_RATE)
            return np.ascontiguousarray(samples, dtype=np.float32)

        block = await self._run_blocking(convert)
        return bool(await self._run_blocking(self.robot.push_audio, block))

    async def clear_output_audio(self) -> None:
        await self._run_blocking(self.robot.clear_live_audio)

    # ── head and antennas ────────────────────────────────────────────────
    @property
    def pending_motion_jobs(self) -> int:
        with self._motion_guard:
            return sum(not future.done() for future in self._motion_futures)

    async def trigger_emote(self, preset: str) -> dict:
        """Queue an EMOTE_PRESETS gesture on the motion worker and return at once: {status: queued, preset,
        pending_motion_jobs}. HardwareError for an unknown preset or when MAX_PENDING_EMOTES are already waiting."""
        if preset not in EMOTE_PRESETS:
            raise HardwareError(f"unknown_emote:{preset}")
        if self._closed:
            raise HardwareError("hardware_closed")
        with self._motion_guard:
            pending = sum(not future.done() for future in self._motion_futures)
            if pending >= MAX_PENDING_EMOTES:
                raise HardwareError("motion_busy")
            future = self._motion_executor.submit(self._run_emote_sync, preset)
            self._motion_futures.add(future)
        # Outside the guard: a future already done runs the callback here and now, and it takes the guard itself
        future.add_done_callback(self._motion_done)
        return {"status": "queued", "preset": preset, "pending_motion_jobs": pending + 1}

    def _motion_done(self, future: Future) -> None:
        with self._motion_guard:
            self._motion_futures.discard(future)
        if future.cancelled():
            return
        error = future.exception()
        if error is not None:
            log.error("Reachy emote failed: %r", error)

    async def wait_motion_idle(self, timeout: float = 10.0) -> bool:
        """Wait until every queued emote has run (tests, shutdown); False when `timeout` passed first."""
        deadline = self.clock() + timeout
        while self.pending_motion_jobs:
            if self.clock() >= deadline:
                return False
            await asyncio.sleep(0.02)
        return True

    def _goto(self, *, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
              antennas: tuple[float, float] = (0.0, 0.0), duration: float = 0.45, z_mm: float = 0.0) -> None:
        clamp = lambda value, limit: max(-limit, min(limit, float(value)))   # noqa: E731
        pose = head_pose(clamp(yaw, HEAD_LIMIT_DEG), clamp(pitch, HEAD_LIMIT_DEG), clamp(roll, HEAD_LIMIT_DEG))
        pose[2, 3] = clamp(z_mm, Z_LIMIT_MM) / 1000
        self.robot.goto(pose, (clamp(antennas[0], ANTENNA_LIMIT_DEG), clamp(antennas[1], ANTENNA_LIMIT_DEG)),
                        duration)

    def _run_emote_sync(self, preset: str) -> None:
        """One preset's choreography on the motion worker (each goto blocks for its duration); it ends in the
        neutral pose with hold_head(), so the check-in gestures may start again."""
        try:
            if preset == "applaud_cheer":
                for _ in range(2):
                    self._goto(pitch=-5, antennas=(-32, 32), duration=0.28)
                    self._goto(pitch=2, antennas=(24, -24), duration=0.28)
                self._goto(pitch=-4, antennas=(-18, 18), duration=0.38)
            elif preset == "sympathy_droop_warm":
                self._goto(pitch=11, roll=-5, antennas=(20, -20), duration=0.8)
                time.sleep(0.15)
                self._goto(pitch=8, roll=3, antennas=(16, -16), duration=0.7)
            elif preset == "curious_tilt":
                self._goto(yaw=9, roll=9, pitch=-2, antennas=(-20, 8), duration=0.55)
                time.sleep(0.2)
            elif preset == "alert_attention":
                self._goto(pitch=-8, z_mm=4, antennas=(-35, 35), duration=0.35)
                time.sleep(0.15)
                self._goto(pitch=-4, antennas=(-28, 28), duration=0.4)
            elif preset == "listening_attentive":
                self._goto(yaw=4, roll=3, antennas=(-13, 13), duration=0.5)
                time.sleep(0.3)
            elif preset == "calming_sway":
                self._goto(yaw=-7, roll=-4, antennas=(-10, 10), duration=0.75)
                self._goto(yaw=7, roll=4, antennas=(-10, 10), duration=0.9)
                self._goto(yaw=-4, roll=-2, antennas=(-8, 8), duration=0.75)
        finally:
            self.robot.hold_head()

    # ── recorded moves: dances and emotion moves (media.MOVE_LIBRARIES) ──
    async def moves(self, library: str) -> dict[str, str]:
        """The recorded moves this robot has in `library` ("dance" or "emotion"), name -> description
        (media.Robot.move_names, off the loop: the library loads on the first call)."""
        return dict(await self._run_blocking(self.robot.move_names, library))

    async def dances(self) -> dict[str, str]:
        return await self.moves("dance")

    async def start_move(self, library: str, name: str | None = None, repeat: int = 1) -> dict:
        """Start a recorded move on the motion worker and return at once: {status: started, library, move
        (and `dance`, the same, for a dance), description, repeat, replaced, available: [names]}. `name` None: one
        of the library's, not the one played last (so "dance again" varies). A recorded move still running or
        queued is stopped first (replaced: true). HardwareError: no_<library>_library, unknown_<library>:<name>
        (the answer's names are the real ones), hardware_closed."""
        if self._closed:
            raise HardwareError("hardware_closed")
        tag = "DANCE" if library == "dance" else "EMOTION"
        moves = await self.moves(library)
        if not moves:
            raise HardwareError(f"no_{library}_library")
        if name is None:
            choices = [move for move in moves if move != self._last_dance] or list(moves)
            name = random.choice(choices)
        elif name not in moves:
            raise HardwareError(f"unknown_{library}:{name}; available: {', '.join(sorted(moves))}")
        times = max(1, min(MAX_DANCE_REPEAT, int(repeat)))
        log.info("[%s] selected=%s repeat=%d", tag, name, times)
        replaced = await self.stop_dance()
        stop = threading.Event()
        with self._motion_guard:
            future = self._motion_executor.submit(self._run_move_sync, library, name, times, stop)
            self._motion_futures.add(future)
            self._dance = (future, stop, name)
        future.add_done_callback(self._motion_done)
        self._last_dance = name
        result = {"status": "started", "library": library, "move": name, "description": moves[name],
                  "repeat": times, "replaced": replaced["status"] == "cancelled", "available": sorted(moves)}
        if library == "dance":
            result["dance"] = name
        return result

    async def start_dance(self, name: str | None = None, repeat: int = 1) -> dict:
        """The dance tool: start_move("dance", ...)."""
        return await self.start_move("dance", name, repeat)

    async def play_emotion(self, name: str) -> dict:
        """One of Pollen's recorded emotion moves once (the live chat's `express`, the play_emotion tool): the
        answer as it starts, as start_move gives it."""
        return await self.start_move("emotion", name, 1)

    async def stop_dance(self) -> dict:
        """Stop the recorded move running or queued now (a dance or an emotion move): {status: cancelled, dance}
        when there was one, else {status: idle}. It stops within one trajectory step and the head returns to the
        neutral pose."""
        with self._motion_guard:
            dance = self._dance
            self._dance = None
        if dance is None or dance[0].done():
            return {"status": "idle"}
        future, stop, name = dance
        stop.set()
        future.cancel()   # a move still queued behind an emote never starts
        log.info("[DANCE] cancelled %s", name)
        return {"status": "cancelled", "dance": name}

    def _run_move_sync(self, library: str, name: str, times: int, stop: threading.Event) -> None:
        """The recorded move on the motion worker: `times` in a row, unless the stop event ends it (media.Robot.
        play_move ends with hold_head either way)."""
        tag = "DANCE" if library == "dance" else "EMOTION"
        log.info("[%s] started %s x%d", tag, name, times)
        outcome = "cancelled"
        try:
            for _ in range(times):
                if stop.is_set():
                    break
                outcome = self.robot.play_move(library, name, stop.is_set)
                if outcome != "completed":
                    break
        except Exception:
            log.exception("[%s] failed %s", tag, name)
            outcome = "failed"
        else:
            log.info("[%s] %s %s", tag, "completed" if outcome == "completed" else "cancelled", name)
        finally:
            with self._motion_guard:
                if self._dance is not None and self._dance[1] is stop:
                    self._dance = None

    # ── shutdown ─────────────────────────────────────────────────────────
    async def shutdown(self) -> None:
        """Stop taking work, stop a dance, and let the queued emotes finish (the servos are never left mid-move)."""
        if self._closed:
            return
        self._closed = True
        await self.stop_dance()
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: self._camera_executor.shutdown(wait=True, cancel_futures=True))
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: self._motion_executor.shutdown(wait=True, cancel_futures=False))
