"""Long-running loops: task polling, heartbeat, and the camera -> landmarks/JPEG pipeline."""

import asyncio
import logging
import time
from collections import deque

from reachy_bridge import __version__
from reachy_bridge.app_client import BridgeError, NotAuthorised
from reachy_bridge.session import SlotSession

log = logging.getLogger(__name__)

CAPTURE_FPS = 15.0
VISION_INTERVAL = 0.2          # JPEG uploads at most 5 fps, like the browser's 200 ms gate
JPEG_QUALITY = 75
FPS_WINDOW = 10.0
HEARTBEAT_SECONDS = 10.0
TICK_SECONDS = 0.2
TASK_WAIT = 25
RESUMABLE = frozenset({"app_unreachable"})   # slot results that leave the task leased to us


def encode_jpeg(frame) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buffer.tobytes()


class MonitorStream:
    """Streams frames to the attached monitor session and keeps its latest public() state.

    At most one landmark and one vision request are in flight; a frame that arrives while the
    landmark request is busy is dropped rather than queued. frame_seq restarts at 1 for every
    session; timestamps are capture times.
    """

    def __init__(self, app, robot, engine, *, clock=time.monotonic, run_blocking=asyncio.to_thread,
                 encode=encode_jpeg, fps=CAPTURE_FPS, vision_interval=VISION_INTERVAL):
        self.app, self.robot, self.engine = app, robot, engine
        self.clock, self.run_blocking, self.encode = clock, run_blocking, encode
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
        self._tasks: set = set()

    # ── the SlotSession-facing interface ─────────────────────────────────
    def attach(self, monitor: dict) -> None:
        self.session = {"session_id": monitor["session_id"], "generation": monitor["generation"]}
        self.latest = dict(monitor)
        self.frame_seq = 0
        self._latest_seq = 0
        self._error = None

    def detach(self) -> None:
        self.session = None
        self.latest = None

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

    # ── pipeline ─────────────────────────────────────────────────────────
    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            started = self.clock()
            try:
                await self.step()
            except Exception:
                log.exception("capture step failed")
                await asyncio.sleep(0.5)
            await asyncio.sleep(max(0.0, self.interval - (self.clock() - started)))

    async def step(self) -> None:
        session = self.session
        if session is None or self._landmark_busy:
            return
        frame = await self.run_blocking(self.robot.get_frame)
        captured = self.clock()
        if frame is None or self.session is not session:
            return
        self.frame_seq += 1
        seq = self.frame_seq
        packet = await self.run_blocking(self.engine.process, frame, seq, captured * 1000.0)
        if self.session is not session:
            return   # session changed while MediaPipe ran: this frame is stale
        jpeg = None
        if not self._vision_busy and captured - self._last_vision >= self.vision_interval:
            self._last_vision = captured
            jpeg = await self.run_blocking(self.encode, frame)
        self._landmark_busy = True
        self._spawn(self._send_landmarks(session, {**session, **packet}, seq, jpeg))

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
    def __init__(self, *, app, robot, clips, stream, heartbeat_seconds=HEARTBEAT_SECONDS,
                 tick_seconds=TICK_SECONDS, run_blocking=asyncio.to_thread):
        self.app, self.robot, self.clips, self.stream = app, robot, clips, stream
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
        try:
            await self._task_loop()
        finally:
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
                           run_blocking=self.run_blocking)
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
        payload = {"robot_reachable": self.robot_reachable, "landmark_fps": self.stream.landmark_fps(),
                   "vision_fps": self.stream.vision_fps(), "bridge_version": __version__,
                   "missing_clips": self.clips.missing_count}
        try:
            response = await self.app.heartbeat(payload)
        except NotAuthorised:
            response = {"stop_all": True}   # revoked device or withdrawn consent
        except BridgeError as exc:
            log.warning("heartbeat failed: %s", exc)
            return None
        if response.get("stop_all") and self.slot:
            log.warning("stop_all from app: stopping the current slot")
            self.slot.stop("stop_all")
        return response

    async def _heartbeat_loop(self) -> None:
        while not self.stopping.is_set():
            await self.heartbeat_once()
            await self._pause(self.heartbeat_seconds)
