"""Long-running loops: task polling, heartbeat, and the camera -> landmarks/JPEG pipeline."""

import asyncio
import logging
import time
from collections import deque

from medcare_reachy.bridge import __version__
from medcare_reachy.bridge.app_client import BridgeError, NotAuthorised
from medcare_reachy.bridge.camera_watch import NOT_WORKING, CameraWatchdog
from medcare_reachy.bridge.session import SlotSession

log = logging.getLogger(__name__)

CAPTURE_FPS = 15.0
FRAME_WAIT_STEPS = 100         # up to ~0.5 s (5 ms polls) for a free slot before dropping a frame
# On a phone hotspot one 45 KB frame took ~0.2 s to reach the laptop and come back, so with one frame in flight the
# stream topped out near 5 fps. Smaller frames and a few in flight keep it near the 15 fps the server needs.
MAX_FRAMES_IN_FLIGHT = 3
STREAM_SIZE = (480, 360)
STREAM_JPEG_QUALITY = 70
# Server vision follows the camera: a frame may go out this share of the frame interval before it is due. The
# camera's own timing jitters by a few ms, and a strict 1/15 s gate would drop every frame that came early.
PACE_EARLY = 0.2
FRAME_POLL_SECONDS = 0.005     # between reads that found no new frame (the SDK itself waits ~20 ms for one)
SERVER_FPS_MIN = 12.0          # the server records only at >= 12 fps (app/services/monitor_service.py FPS_MIN)
STALL_SECONDS = 0.25           # a longer gap between streamed frames makes the server call the stream degraded
STALL_LOG_SECONDS = 1.0        # at most one stall line per this long (a broken camera would log every frame)
VISION_INTERVAL = 0.5          # identity-only snapshots, 2 fps (the server re-checks identity every 0.5 s)
EMOTION_INTERVAL = 0.5         # emotion is scored on the robot, at most 2 fps, only for the verified face
JPEG_QUALITY = 75
FPS_WINDOW = 10.0
HEARTBEAT_SECONDS = 10.0
TICK_SECONDS = 0.2
TASK_WAIT = 25
RESUMABLE = frozenset({"app_unreachable"})   # slot results that leave the task leased to us


def _cannot_reopen() -> str:
    """The camera watchdog's recovery for a robot without reopen_camera() (VideoFileRobot, test cameras)."""
    return "failed: this robot can't reopen its camera"


def _no_media_status() -> None:
    return None


def encode_stream_jpeg(frame) -> bytes:
    """BGR camera frame -> its 4:3 centre at STREAM_SIZE as a JPEG, for the server-side landmark stream.

    One pass from the camera's own frame: the Wireless's 1280x720 has a 960x720 centre that halves exactly to
    480x360. On the Pi CM4 that takes ~12 ms, against ~38 ms for crop_4_3 to 640x480 and then a second resize.
    """
    import io

    import numpy as np
    from PIL import Image

    height, width = frame.shape[:2]
    crop_width = min(width, int(height * 4 / 3))
    x = (width - crop_width) // 2
    crop = np.ascontiguousarray(frame[:, x:x + crop_width])
    image = Image.frombuffer("RGB", (crop_width, height), crop, "raw", "BGR", 0, 1)
    factor = crop_width // STREAM_SIZE[0]
    if factor >= 2 and image.size == (STREAM_SIZE[0] * factor, STREAM_SIZE[1] * factor):
        image = image.reduce(factor)
    if image.size != STREAM_SIZE:
        image = image.resize(STREAM_SIZE, Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=STREAM_JPEG_QUALITY)
    return buffer.getvalue()


def encode_jpeg(frame) -> bytes:
    """BGR frame -> JPEG bytes."""
    import io

    import numpy as np
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1])).save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()


class MonitorStream:
    """Streams frames to the attached monitor session and keeps its latest public() state.

    With no `engine`, the server computes the landmarks: each frame goes out as a JPEG to
    /monitor/frame, with up to MAX_FRAMES_IN_FLIGHT requests outstanding (the server keeps them in
    order and ignores a frame older than one it already has). The loop follows the camera rather
    than a fixed timer, taking every frame up to `fps`: a timer only slightly slower than the camera
    loses frames, and the server records nothing below 12 fps.
    With an `engine`, the robot computes landmarks itself and posts them plus a 2 fps JPEG; then at
    most one landmark request and one vision request are in flight.
    A frame that finds no free slot is dropped rather than queued. frame_seq restarts at 1 for every
    session; timestamps are capture times.
    A gap over STALL_SECONDS between streamed frames is logged with what held it up (see _note_stall).
    `camera` (camera_watch.CameraWatchdog) follows every read of an attached session: frame() for a new frame,
    tick() (off the event loop: a recovery blocks) for none. By default it recovers with the robot's
    reopen_camera() and logs its media_status(); the heartbeat sends its snapshot().
    """

    def __init__(self, app, robot, engine, *, clock=time.monotonic, run_blocking=asyncio.to_thread,
                 encode=encode_jpeg, stream_encode=encode_stream_jpeg, fps=CAPTURE_FPS,
                 vision_interval=VISION_INTERVAL, emotion=None, emotion_interval=EMOTION_INTERVAL,
                 max_in_flight=MAX_FRAMES_IN_FLIGHT, camera=None):
        self.app, self.robot, self.engine = app, robot, engine
        self.camera = camera if camera is not None else CameraWatchdog(
            getattr(robot, "reopen_camera", _cannot_reopen), getattr(robot, "media_status", _no_media_status),
            clock=clock, media_released=getattr(robot, "media_released", None))
        self.emotion, self.emotion_interval = emotion, emotion_interval
        self._last_emotion = float("-inf")
        self.clock, self.run_blocking, self.encode, self.stream_encode = clock, run_blocking, encode, stream_encode
        self.max_in_flight = max_in_flight
        self._frames_in_flight = 0
        self.interval = 1.0 / fps
        self.vision_interval = vision_interval
        self.session: dict | None = None
        self.latest: dict | None = None
        self.frame_seq = 0
        self._latest_seq = 0
        self._error: Exception | None = None
        self._landmark_busy = False
        self._vision_busy = False
        self._last_vision = float("-inf")
        self._landmark_times: deque = deque()
        self._vision_times: deque = deque()
        self._camera_times: deque = deque()
        self._next_due = float("-inf")   # when the server stream's next frame is due
        self._streaming_since: float | None = None
        self._slow_camera_logged = False
        self._tasks: set = set()
        # What held the server stream up, for _note_stall.
        self._sent_at: dict = {}         # one key per unanswered request -> when it went out
        self._last_streamed: float | None = None
        self._last_stall_log = float("-inf")
        self._stalls = 0
        self._camera_asked: float | None = None   # since when the stream has waited for a new camera frame
        self._clear_stall_notes()

    # ── the SlotSession-facing interface ─────────────────────────────────
    def attach(self, monitor: dict) -> None:
        self.session = {"session_id": monitor["session_id"], "generation": monitor["generation"]}
        self.latest = dict(monitor)
        self.frame_seq = 0
        self._latest_seq = 0
        self._error = None
        self._streaming_since = None
        self._last_streamed = None
        self._stalls = 0
        self._camera_asked = None
        self._clear_stall_notes()
        self.camera.attach()

    def detach(self) -> None:
        self.session = None
        self.latest = None
        self.camera.detach()

    def take_error(self) -> Exception | None:
        error, self._error = self._error, None
        return error

    # ── rates for the heartbeat ──────────────────────────────────────────
    def _rate(self, times: deque) -> float:
        cutoff = self.clock() - FPS_WINDOW
        while times and times[0] < cutoff:
            times.popleft()
        return round(len(times) / FPS_WINDOW, 1)

    def landmark_fps(self) -> float:
        return self._rate(self._landmark_times)

    def vision_fps(self) -> float:
        return self._rate(self._vision_times)

    def camera_fps(self) -> float:
        """New frames the camera handed the server stream: the ceiling for landmark_fps."""
        return self._rate(self._camera_times)

    # ── pipeline ─────────────────────────────────────────────────────────
    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            started = self.clock()
            try:
                await self.step()
            except Exception:
                log.exception("capture step failed")
                await asyncio.sleep(0.5)
            pause = self._pause(started)
            asleep = self.clock()
            await asyncio.sleep(pause)
            # Waking late means something else held this process's event loop, or the CPU, meanwhile.
            self._loop_late = max(self._loop_late, self.clock() - asleep - pause)

    def _pause(self, started: float) -> float:
        if self.session is not None and self.engine is None:
            # Sleep until the next frame may go out, then take the first one the camera hands over.
            return max(FRAME_POLL_SECONDS, self._next_due - self.interval * PACE_EARLY - self.clock())
        return max(0.0, self.interval - (self.clock() - started))

    async def step(self) -> None:
        session = self.session
        if session is not None and self.engine is None:
            await self._stream_frame(session)
            return
        if session is None or self._landmark_busy:
            return
        frame = await self.run_blocking(self.robot.get_frame)
        captured = self.clock()
        if self.session is not session:
            return
        if frame is None:
            await self.run_blocking(self.camera.tick)   # a recovery blocks for a while: never on the event loop
            return
        self._camera_seen()
        self.frame_seq += 1
        seq = self.frame_seq
        packet = await self.run_blocking(self.engine.process, frame, seq, captured * 1000.0)
        if self.session is not session:
            return   # session changed while the landmark models ran: this frame is stale
        packet = await self._with_emotion(frame, packet, captured)
        jpeg = None
        if not self._vision_busy and captured - self._last_vision >= self.vision_interval:
            self._last_vision = captured
            jpeg = await self.run_blocking(self.encode, frame)
        self._landmark_busy = True
        self._spawn(self._send_landmarks(session, {**session, **packet}, seq, jpeg))

    async def _stream_frame(self, session) -> None:
        # The camera's own frame: encode_stream_jpeg crops and scales it in one pass.
        asked = self.clock()
        frame = await self.run_blocking(self.robot.get_camera_frame)
        captured = self.clock()
        if self.session is not session:
            return
        if frame is None:
            if self._camera_asked is None:
                self._camera_asked = asked
            await self.run_blocking(self.camera.tick)   # a recovery blocks for a while: never on the event loop
            return
        self._camera_wait = max(self._camera_wait, captured - (asked if self._camera_asked is None
                                                               else self._camera_asked))
        self._camera_asked = None
        self._camera_seen()
        self._note_camera_frame(captured)
        if captured < self._next_due - self.interval * PACE_EARLY:
            return   # the camera runs faster than the stream: skip this one
        jpeg = await self.run_blocking(self.stream_encode, frame)
        for _ in range(FRAME_WAIT_STEPS):
            if self._frames_in_flight < self.max_in_flight or self.session is not session:
                break
            await asyncio.sleep(0.005)
        if self.session is not session:
            return
        if self._frames_in_flight >= self.max_in_flight:
            self._no_slot += 1
            if self._sent_at:
                self._oldest_request = max(self._oldest_request, self.clock() - min(self._sent_at.values()))
            return
        self.frame_seq += 1
        self._frames_in_flight += 1
        self._note_stall(captured)
        # Keep to the schedule (a 15 fps camera streamed at 10 fps sends 2 frames in 3), unless the camera fell
        # more than a frame behind it: then start again from this frame.
        on_time = captured - self._next_due <= self.interval
        self._next_due = (self._next_due if on_time else captured) + self.interval
        request = object()
        self._sent_at[request] = self.clock()
        self._spawn(self._send_frame(session, self.frame_seq, captured, jpeg, request))

    def _clear_stall_notes(self) -> None:
        """Start the notes on what holds up the next streamed frame."""
        self._camera_wait = 0.0       # longest wait for a new camera frame
        self._loop_late = 0.0         # latest the event loop woke up
        self._no_slot = 0             # frames dropped while every request was unanswered
        self._oldest_request = 0.0    # the oldest unanswered request at such a drop

    def _note_stall(self, captured: float) -> None:
        """Log a gap the server will see before this frame (it trusts no gap over 0.25 s), with what held it up.

        On 2 Oct 2026 the stream stopped for ~1 s at a time in every slot state, then 4-6 answers came within
        0.2 s, and nothing said where it had waited. Frames dropped behind old unanswered requests point at Wi-Fi
        or the server (whose log names its slow frames), a late event loop at this process, and a long camera
        wait at the camera feed (or a starved CPU).
        """
        gap = 0.0 if self._last_streamed is None else captured - self._last_streamed
        self._last_streamed = captured
        if gap > STALL_SECONDS:
            self._stalls += 1
            if captured - self._last_stall_log >= STALL_LOG_SECONDS:
                self._last_stall_log = captured
                log.warning("server stream: %.2f s between frames before frame %d (%d gap(s) over %.2f s since the "
                            "last report); longest wait for a camera frame %.2f s, event loop up to %.2f s late, "
                            "frames dropped for want of a free request slot: %d (oldest unanswered request %.2f s)",
                            gap, self.frame_seq, self._stalls, STALL_SECONDS, self._camera_wait, self._loop_late,
                            self._no_slot, self._oldest_request)
                self._stalls = 0
        self._clear_stall_notes()

    def _camera_seen(self) -> None:
        """A new camera frame for the watchdog. After a stall the slow-camera warning may fire again, measured from
        the recovery."""
        recovered = self.camera.state in NOT_WORKING
        self.camera.frame()
        if recovered:
            self._slow_camera_logged = False
            self._streaming_since = None

    def _note_camera_frame(self, captured: float) -> None:
        """Count camera frames, and say once in the log when the camera alone keeps the server from recording."""
        self._camera_times.append(captured)
        fps = self.camera_fps()   # also forgets frames older than the window
        if self._streaming_since is None:
            self._streaming_since = captured
        if not self._slow_camera_logged and captured - self._streaming_since >= FPS_WINDOW and fps < SERVER_FPS_MIN:
            self._slow_camera_logged = True
            # Reachy Mini daemon 1.11 caps the camera feed it shares with apps at 10 fps (IPC_FPS in
            # reachy_mini/media/media_server.py); tools/deploy_to_robot.py --camera-ipc-fps 15 lifts it.
            log.warning("the camera hands this app only %.1f fps; the server records doses only at >= %.0f fps, "
                        "so every dose goes to family confirmation. Check the daemon's IPC_FPS.",
                        fps, SERVER_FPS_MIN)

    async def _send_frame(self, session, seq: int, captured: float, jpeg: bytes, request: object) -> None:
        try:
            response = await self.app.monitor_frame(session["session_id"], session["generation"], seq, captured, jpeg)
            self._landmark_times.append(self.clock())
            self._accept(session, response)
        except BridgeError as exc:
            self._fail(session, exc)
        finally:
            self._sent_at.pop(request, None)
            self._frames_in_flight -= 1

    async def _with_emotion(self, frame, packet: dict, captured: float) -> dict:
        """Attach the robot-scored emotion of the server-verified face (the server re-checks it)."""
        target = (self.latest or {}).get("target_box")
        if self.emotion is None or not target or captured - self._last_emotion < self.emotion_interval:
            return packet
        self._last_emotion = captured
        try:
            report = await self.run_blocking(self.emotion.score, frame, packet, target)
        except Exception:   # emotion is optional; never lose the landmark frame over it
            log.exception("emotion scoring failed")
            return packet
        return packet if report is None else {**packet, "emotion": report}

    async def drain(self) -> None:
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks))

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _send_landmarks(self, session, packet, seq, jpeg) -> None:
        try:
            response = await self.app.monitor_landmarks(packet)
            self._landmark_times.append(self.clock())
            self._accept(session, response)
            # Like the browser: the JPEG follows its landmark packet, so the server can pair them.
            if jpeg is not None and self.session is session and not self._vision_busy:
                self._vision_busy = True
                self._spawn(self._send_vision(session, seq, jpeg))
        except BridgeError as exc:
            self._fail(session, exc)
        finally:
            self._landmark_busy = False

    async def _send_vision(self, session, seq, jpeg) -> None:
        try:
            response = await self.app.monitor_vision(session["session_id"], session["generation"], seq, jpeg)
            self._vision_times.append(self.clock())
            self._accept(session, response)
        except BridgeError as exc:
            self._fail(session, exc)
        finally:
            self._vision_busy = False

    def _accept(self, session, response) -> None:
        # Mirrors Intake.tsx acceptStatus: never move back to an older frame's state,
        # and never let a same-frame response without `recorded` hide one with it.
        if self.session is not session or not response:
            return
        seq = int(response.get("frame_seq") or 0)
        if seq < self._latest_seq:
            return
        if seq == self._latest_seq and self.latest and self.latest.get("recorded") and not response.get("recorded"):
            return
        self._latest_seq = seq
        self.latest = response

    def _fail(self, session, exc: Exception) -> None:
        if self.session is session:
            self._error = exc


class Runner:
    def __init__(self, *, app, robot, clips, stream, voice=None, speaker=None, language="zh-TW",
                 heartbeat_seconds=HEARTBEAT_SECONDS, tick_seconds=TICK_SECONDS, run_blocking=asyncio.to_thread,
                 ack=True, gestures=True):
        self.app, self.robot, self.clips, self.stream, self.voice = app, robot, clips, stream, voice
        self.speaker, self.language = speaker, language
        self.ack, self.gestures = ack, gestures   # check-in 「嗯」 and body language (SlotSession)
        self.heartbeat_seconds = heartbeat_seconds
        self.tick_seconds = tick_seconds
        self.run_blocking = run_blocking
        self.slot: SlotSession | None = None
        self.stopping = asyncio.Event()
        self.robot_reachable: bool | None = None

    def request_shutdown(self) -> None:
        """Process exit: leave the task leased so it is resumed (tasks/current) or re-queued on lease expiry."""
        self.stopping.set()
        if self.slot:
            self.slot.stop("bridge_shutdown", abort=False)

    async def run(self) -> None:
        loops = [asyncio.create_task(self.stream.run(self.stopping)),
                 asyncio.create_task(self._heartbeat_loop())]
        tasks = asyncio.create_task(self._task_loop())
        stopping = asyncio.create_task(self.stopping.wait())
        try:
            await asyncio.wait({tasks, stopping}, return_when=asyncio.FIRST_COMPLETED)
            if not tasks.done():
                # Don't sit out a 25 s long poll on shutdown; a running slot ends itself on cancellation
                # (session ended, robot asleep, task left leased for recovery).
                tasks.cancel()
                await asyncio.gather(tasks, return_exceptions=True)
            elif not tasks.cancelled():
                tasks.result()
        finally:
            stopping.cancel()
            self.stopping.set()
            for loop in loops:
                loop.cancel()
            await asyncio.gather(*loops, return_exceptions=True)

    async def _task_loop(self) -> None:
        # Restart recovery: a task this device already holds comes first. The same applies after
        # failing closed on app loss: the lease is still ours, so tasks/next would never return it.
        check_current = True
        while not self.stopping.is_set():
            try:
                if check_current:
                    task = await self.app.tasks_current()
                    check_current = False
                else:
                    task = await self.app.tasks_next(TASK_WAIT)
            except NotAuthorised as exc:
                log.error("device token rejected (%s); pair the robot again or restore consent", exc.detail)
                await self._pause(60)
                continue
            except BridgeError as exc:
                log.warning("task poll failed: %s", exc)
                await self._pause(5)
                continue
            if task:
                slot = await self.run_slot(task)
                check_current = slot.result in RESUMABLE

    async def _pause(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self.stopping.wait(), seconds)
        except asyncio.TimeoutError:
            pass

    async def run_slot(self, task: dict) -> SlotSession:
        log.info("task %s (%s): %d dose(s)", task.get("task_id"), task.get("reason"), len(task.get("doses", [])))
        slot = SlotSession(task, app=self.app, robot=self.robot, clips=self.clips, stream=self.stream,
                           voice=self.voice, speaker=self.speaker, language=self.language,
                           run_blocking=self.run_blocking, ack=self.ack, gestures=self.gestures)
        self.slot = slot
        if self.stopping.is_set():
            slot.stop("bridge_shutdown", abort=False)
        try:
            while not slot.done:
                await slot.tick()
                if not slot.done:
                    await asyncio.sleep(self.tick_seconds)
        except asyncio.CancelledError:
            slot.stop("bridge_shutdown", abort=False)   # end the session and sleep the robot, keep the lease
            await slot.tick()
            raise
        finally:
            self.slot = None
        return slot

    async def heartbeat_once(self) -> dict | None:
        try:
            self.robot_reachable = bool(await self.run_blocking(self.robot.is_reachable))
        except Exception:
            self.robot_reachable = False
        if self.robot_reachable:
            # A camera recovery the daemon refused leaves this app's media released, the speaker and the microphone
            # with it, and outside a stalled session nothing else asks for it back (restore_media never raises).
            await self.run_blocking(self.robot.restore_media)
        payload = {"robot_reachable": self.robot_reachable, "landmark_fps": self.stream.landmark_fps(),
                   "vision_fps": self.stream.vision_fps(), "bridge_version": __version__,
                   "missing_clips": self.clips.missing_count}
        try:
            response = await self.app.heartbeat(payload, camera=self.stream.camera.snapshot())
        except NotAuthorised:
            response = {"stop_all": True, "microphone": False}   # revoked device or withdrawn consent
        except BridgeError as exc:
            log.warning("heartbeat failed: %s", exc)
            return None
        if response.get("stop_all") and self.slot:
            log.warning("stop_all from app: stopping the current slot")
            self.slot.stop("stop_all")
        if self.slot and "microphone" in response:
            # Withdrawn microphone consent takes effect within one heartbeat, not only at the next task.
            self.slot.task["microphone"] = bool(response["microphone"])
        return response

    async def _heartbeat_loop(self) -> None:
        while not self.stopping.is_set():
            await self.heartbeat_once()
            await self._pause(self.heartbeat_seconds)
