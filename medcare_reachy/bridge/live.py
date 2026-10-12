"""Live voice check-ins on the robot (spec 2026-10-12 F1): one WebSocket to the server's GPT-Live relay.

The robot streams its microphone to the server and plays what comes back; only the server talks to OpenAI. During
the chat this module is the robot's one audio owner: it reads the microphone (robot.get_audio) and feeds the speaker
(robot.start_live_audio once, then robot.push_audio in 20 ms blocks on an absolute-deadline clock). The speech
listener (voice.py) must be paused meanwhile, and no clip may play: play_clip restarts the SDK's playback.

Messages (JSON text frames carry a "type"; binary frames are PCM16 mono at 16 kHz, both ways):
- robot -> server: start {task_id, language, rate, bridge_version}; then 40 ms microphone frames (silence while
  Reachy says a line of its own); playing / idle when Reachy's speaker starts and stops making sound; end {reason}
  when the robot ends the chat itself.
- server -> robot: ready {conversation_id}; GPT-Live's voice; flush (drop what is queued, now); say {speech_text,
  reason} (Reachy says it with its own Matcha voice, through the same queue); fallback {reason, conversation_id};
  end {reason, risk}.

Ported from the TCU companion (reachy-mini-tcu, companion/audio/reachy_audio.py, base.py, protected_playback.py and
companion/session.py): the paced playback loop with absolute deadlines, the prebuffer that waits again after an
underrun, and the onset level 120. Not ported: its media release (it takes the daemon's camera away, CLAUDE.md), its
24 kHz resampling (the robot and the session both run at 16 kHz), and its fades (the SDK's pipeline has no volume
element an app can reach; a flush cuts at once).
"""

import asyncio
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from medcare_reachy.bridge import __version__
from medcare_reachy.bridge.media import resample
from medcare_reachy.bridge.speech import chunks

log = logging.getLogger(__name__)

RATE = 16000                   # the robot's audio and the GPT-Live session, both ways (spike: no resampling)
FRAME_SAMPLES = 640            # 40 ms microphone frames to the server (spec A2: 20-40 ms)
BLOCK_SAMPLES = 320            # 20 ms speaker blocks
BLOCK_SECONDS = BLOCK_SAMPLES / RATE
JITTER_START_SECONDS = 0.3     # GPT-Live's voice waits until this much is queued (its deltas came up to 0.37 s apart)
JITTER_STEP_SECONDS = 0.2      # ... and this much longer after each underrun,
JITTER_MAX_SECONDS = 1.0       # ... up to this
UNDERRUN_GAP_SECONDS = 1.0     # GPT-Live's voice coming back this soon after the queue ran dry: it ran dry mid-reply
ONSET_LEVEL = 120              # a block with a sample above this (of 32767) makes sound (TCU session.py:561)
IDLE_QUIET_SECONDS = 0.6       # 'idle' this long after the last block with sound (TCU session.py:608)
IDLE_QUIET_BLOCKS = round(IDLE_QUIET_SECONDS / BLOCK_SECONDS)
# The microphone stays silent this long after Reachy's own line was pushed: about 0.22 s until the SDK has played it
# (its pipeline and sink buffer), then voice.py's 0.35 s for the echo to die down.
SAY_TAIL_SECONDS = 0.6
SAY_TAIL_BLOCKS = math.ceil(SAY_TAIL_SECONDS / BLOCK_SECONDS)
CONNECT_TIMEOUT = 5.0          # opening the WebSocket
CLOSE_TIMEOUT = 2.0
READY_TIMEOUT = 15.0           # 'ready' or 'fallback' after 'start': the server's checks plus its 8 s GPT-Live start
POLL_SECONDS = 0.1             # how often the chat looks at its time limits and its loops
DRAIN_LIMIT_SECONDS = 30.0     # after the server's end or fallback the queue (the help line) plays, this long at most
MAX_SECONDS = 660.0            # the server ends a chat at 600 s (LIVE_MAX_SECONDS); this only guards against it not
MIC_IDLE_SECONDS = 0.005       # no microphone samples yet: look again after this (TCU reachy_audio.py:94)
LATE_RESET_SECONDS = 0.1       # the speaker's clock this late: start afresh, never burst (TCU reachy_audio.py:125)

# What a queued sound is: GPT-Live's voice, Reachy's own line, Reachy's own line said without moving.
LIVE, SAY, STILL = "live", "say", "still"
_STOP = object()               # stop() wakes the chat with this in its inbox

# Why the robot left the live chat by itself (LiveOutcome.reason; the server's reasons are passed on as it sent them).
NO_AUDIO = "no_audio"               # this app has no speaker now (its media is released)
CONNECT_FAILED = "connect_failed"   # the WebSocket did not open
NO_ANSWER = "no_answer"             # neither 'ready' nor 'fallback' within READY_TIMEOUT
DISCONNECTED = "disconnected"       # the connection closed without 'end' or 'fallback'
ROBOT_ERROR = "robot_error"         # the robot's microphone, speaker or socket loop failed
MAX_LENGTH = "max_length"           # MAX_SECONDS passed
CLOSE_REASONS = {4401: "not_authorised", 4403: "consent_required", 4409: "live_busy"}   # the server's close codes


@dataclass
class LiveOutcome:
    """How a live chat ended. "ended": it is over (the server's end, or the robot's own end after stop()); "fallback":
    go on with today's text chat, on `conversation_id` when the server's fallback named one (POST
    /conversations/{id}/turn), else on a new one (POST /conversations): the server ends a conversation whose robot
    connection closed (spec C2), so a fallback the robot decides by itself never names one. `risk`: the server ended
    it for a risk, after Reachy said the help line."""
    kind: str
    reason: str
    conversation_id: str | None
    risk: bool = False


def to_pcm(samples) -> np.ndarray:
    """float32 samples in [-1, 1] -> PCM16."""
    return (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 32767 / 32768) * 32768).astype("<i2")


def mic_pcm(samples, rate: int) -> np.ndarray:
    """One channel of the SDK's microphone samples, as PCM16 at RATE. Both channels carry the XVF3800 chip's processed
    beam (spike robot-audio.md 1.4), so the first one is enough."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples[:, 0]
    if int(rate) != RATE:
        samples = resample(samples, int(rate), RATE)
    return to_pcm(samples)


class Player:
    """The chat's one playback queue: GPT-Live's voice and Reachy's own lines in the order they came, handed out in
    BLOCK_SAMPLES blocks.

    GPT-Live's voice starts once `target` seconds of it are queued, or once nothing more has come for `target` seconds
    (the end of a reply), or once finish() says nothing more will come. When it runs dry and more comes within
    UNDERRUN_GAP_SECONDS, that was an underrun: it waits for the buffer again, JITTER_STEP_SECONDS longer each time, up
    to JITTER_MAX_SECONDS. Reachy's own lines are queued whole and play as soon as everything before them has.
    """

    def __init__(self):
        self.target = JITTER_START_SECONDS
        self.underruns = 0
        self.queued = 0                     # samples
        self._items: deque = deque()        # [samples (PCM16), samples played, kind]
        self._ready = False                 # GPT-Live's voice may play without waiting
        self._complete = False              # finish(): nothing more will come
        self._arrived: float | None = None  # when the last frame of GPT-Live's voice came
        self._dry_at: float | None = None   # when GPT-Live's voice last ran out while playing

    def add_live(self, pcm: bytes, now: float) -> None:
        """A frame of GPT-Live's voice (PCM16 mono at RATE) that came at `now`; a stray last byte is dropped."""
        samples = np.frombuffer(pcm, dtype="<i2", count=len(pcm) // 2)
        if not len(samples):
            return
        if self._dry_at is not None and now - self._dry_at <= UNDERRUN_GAP_SECONDS:
            self.underruns += 1
            self.target = min(JITTER_MAX_SECONDS, self.target + JITTER_STEP_SECONDS)
        self._dry_at = None
        self._arrived = now
        self._append(samples, LIVE)

    def add_say(self, samples: np.ndarray, still: bool) -> None:
        """Reachy's own line (PCM16 at RATE); `still`: said without moving (the help line after a risk)."""
        self._append(np.asarray(samples, dtype="<i2"), STILL if still else SAY)

    def _append(self, samples: np.ndarray, kind: str) -> None:
        if len(samples):
            self._items.append([samples, 0, kind])
            self.queued += len(samples)

    def has(self, *kinds: str) -> bool:
        return any(item[2] in kinds for item in self._items)

    def flush(self) -> None:
        """Drop everything queued; GPT-Live's voice waits for the buffer again."""
        self._items.clear()
        self.queued = 0
        self._ready = False
        self._dry_at = None

    def finish(self) -> None:
        """Nothing more will come: what is queued plays without waiting for the buffer."""
        self._complete = True

    def block(self, now: float) -> tuple[np.ndarray, str | None]:
        """The next BLOCK_SAMPLES samples to play (silence where nothing may play yet), and the kind of sound it starts
        with: LIVE, SAY, STILL, or None for silence."""
        out = np.zeros(BLOCK_SAMPLES, dtype="<i2")
        filled, kind = 0, None
        while self._items and filled < BLOCK_SAMPLES:
            item = self._items[0]
            samples, played, item_kind = item
            if item_kind == LIVE and not self._may_play(now):
                break
            count = min(BLOCK_SAMPLES - filled, len(samples) - played)
            out[filled:filled + count] = samples[played:played + count]
            filled += count
            kind = kind or item_kind
            self.queued -= count
            item[1] += count
            if item[1] == len(samples):
                self._items.popleft()
                if item_kind == LIVE and not self.has(LIVE):
                    self._ready, self._dry_at = False, now   # GPT-Live's voice ran out
        return out, kind

    def _may_play(self, now: float) -> bool:
        if not self._ready:
            live = sum(len(samples) - played for samples, played, kind in self._items if kind == LIVE)
            paused = self._arrived is not None and now - self._arrived >= self.target
            self._ready = self._complete or paused or live >= round(self.target * RATE)
        return self._ready


async def connect_websocket(url: str, headers: dict) -> "WebSocket":
    """Open the live chat's WebSocket with websockets' asyncio client (15.0.1 in the robot's apps_venv)."""
    from websockets.asyncio.client import connect   # lazy: the tests use a fake socket
    from websockets.exceptions import ConnectionClosed

    # No compression: PCM hardly compresses, and deflating 32 KB/s each way would cost the Pi's CPU for nothing.
    connection = await connect(url, additional_headers=headers, open_timeout=CONNECT_TIMEOUT, compression=None,
                               close_timeout=CLOSE_TIMEOUT)
    return WebSocket(connection, ConnectionClosed)


class WebSocket:
    """A websockets connection in the shape the chat uses (the tests' FakeSocket has it too): send() and receive()
    never raise for a closed connection; receive() gives None then, and close_code is the server's close code."""

    def __init__(self, connection, closed_error):
        self._connection, self._closed_error = connection, closed_error
        self.close_code: int | None = None

    async def send(self, data: str | bytes) -> bool:
        try:
            await self._connection.send(data)
            return True
        except self._closed_error:
            return False

    async def receive(self) -> str | bytes | None:
        try:
            return await self._connection.recv()
        except self._closed_error as exc:
            self.close_code = exc.rcvd.code if exc.rcvd is not None else None
            return None

    async def close(self, code: int = 1000) -> None:
        try:
            await self._connection.close(code)
        except Exception:   # already closed or broken: nothing left to close
            log.debug("live chat: closing the WebSocket failed", exc_info=True)


class LiveChat:
    """One live check-in: run() opens the WebSocket, streams the microphone, plays what comes back, and returns how it
    ended. Use each LiveChat once. stop() ends it from outside (the task stopped, the bridge shuts down).

    `connect(url, headers)` opens the socket (default: connect_websocket); `clock` paces the speaker and times the
    limits; `sleep` and `run_blocking` (robot calls, Matcha) are replaced in the tests; `gestures`: the settings
    page's check-in gestures switch (Reachy's "speak" gesture while it makes sound, still during the help line).
    """

    def __init__(self, app, robot, speaker, *, connect=None, clock=time.monotonic, sleep=asyncio.sleep,
                 run_blocking=asyncio.to_thread, gestures=True):
        self.app, self.robot, self.speaker = app, robot, speaker
        self._connect = connect or connect_websocket
        self.clock, self._sleep, self._run_blocking = clock, sleep, run_blocking
        self.gestures = gestures
        self.player = Player()
        self.conversation_id: str | None = None
        self.frames_sent = 0          # microphone frames sent
        self.dropped_audio = 0        # frames of GPT-Live's voice that came while Reachy said its own line
        self._socket = None
        self._inbox: asyncio.Queue = asyncio.Queue()   # the server's messages; None once the connection closed
        self._ready = False
        self._stop_reason: str | None = None
        self._workers: list[asyncio.Task] = []
        self._says: set[asyncio.Task] = set()
        self._saying = 0              # Reachy's own lines still being synthesised
        self._say_tail = 0            # blocks the microphone still stays silent after Reachy's own line
        self._playing = False         # 'playing' was reported, 'idle' not yet
        self._quiet = 0               # blocks without sound since the last one with sound
        self._gesture_mode: str | None = None
        self._speaker_lock = asyncio.Lock()   # a flush never lands between taking a block and pushing it

    def stop(self, reason: str) -> None:
        """End the chat now: run() drops what is queued, tells the server `end {reason}` and returns an "ended"
        outcome. Call it on the chat's event loop (the bridge's stop paths run there)."""
        if self._stop_reason is None:
            self._stop_reason = reason
            self._inbox.put_nowait(_STOP)

    async def run(self, task_id: str, language: str) -> LiveOutcome:
        """The whole live chat. A failure of the server, the connection or the robot's audio never raises: it ends
        the chat with a "fallback" outcome. Cancelling it stops everything, without a word to the server."""
        try:
            started = bool(await self._run_blocking(self.robot.start_live_audio))
        except Exception:
            log.exception("live chat: the speaker did not start")
            started = False
        if not started:
            log.warning("live chat: no speaker now; Reachy uses the text chat")
            return LiveOutcome("fallback", NO_AUDIO, None)
        try:
            try:
                self._socket = await asyncio.wait_for(self._connect(self.app.live_url(), self.app.auth_header()),
                                                      CONNECT_TIMEOUT + 1)
            except Exception as exc:   # refused, timed out, an HTTP answer instead of a WebSocket (an older server)
                log.warning("live chat: no connection to the server (%r); Reachy uses the text chat", exc)
                return LiveOutcome("fallback", CONNECT_FAILED, None)
            await self._send({"type": "start", "task_id": task_id, "language": language, "rate": RATE,
                              "bridge_version": __version__})
            self._workers = [asyncio.create_task(self._read(), name="live-read"),
                             asyncio.create_task(self._mic_loop(), name="live-microphone"),
                             asyncio.create_task(self._play_loop(), name="live-speaker")]
            outcome, drain = await self._converse()
            if drain:
                await self._drain()
            if self._stop_reason is not None and not (outcome.kind == "ended" and outcome.reason == self._stop_reason):
                # stop() came during the drain: the server's earlier fallback or end no longer decides; only a risk
                # end keeps its flag
                outcome = LiveOutcome("ended", self._stop_reason, self.conversation_id,
                                      outcome.kind == "ended" and outcome.risk)
            log.info("live chat %s: %s (%s); %d microphone frames sent, %d underruns, buffer %.1f s",
                     self.conversation_id, outcome.kind, outcome.reason, self.frames_sent, self.player.underruns,
                     self.player.target)
            return outcome
        finally:
            tasks = [*self._workers, *self._says]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self._socket is not None:
                await self._socket.close()
            self._set_gesture(None)
            try:
                await self._run_blocking(self.robot.stop_live_audio)
            except Exception:
                log.exception("live chat: the speaker did not stop cleanly")

    # ── the server's messages ────────────────────────────────────────────
    async def _converse(self) -> tuple[LiveOutcome, bool]:
        """Handle what the server sends until the chat ends: (how it ended, whether what is queued still plays)."""
        started = self.clock()
        while True:
            ending = self._robot_ending(started)
            if ending is not None:
                await self._flush(report=False)
                await self._send({"type": "end", "reason": ending.reason})
                return ending, False
            try:
                message = await asyncio.wait_for(self._inbox.get(), POLL_SECONDS)
            except asyncio.TimeoutError:
                continue
            if message is _STOP:
                continue
            if message is None:
                code = self._socket.close_code
                log.warning("live chat %s: the connection closed (code %s)", self.conversation_id, code)
                return LiveOutcome("fallback", CLOSE_REASONS.get(code, DISCONNECTED), None), True
            if isinstance(message, (bytes, bytearray)):
                self._audio(bytes(message))
                continue
            outcome = await self._control(message)
            if outcome is not None:
                return outcome, True

    def _robot_ending(self, started: float) -> LiveOutcome | None:
        """Whether the robot ends the chat itself now: stop(), a failed audio or socket loop, no answer to 'start', or
        MAX_SECONDS."""
        if self._stop_reason is not None:
            return LiveOutcome("ended", self._stop_reason, self.conversation_id)
        for task in self._workers:
            if task.done() and not task.cancelled() and task.exception() is not None:
                log.error("live chat %s: %s failed: %r", self.conversation_id, task.get_name(), task.exception())
                return LiveOutcome("fallback", ROBOT_ERROR, None)
        elapsed = self.clock() - started
        if not self._ready and elapsed >= READY_TIMEOUT:
            log.warning("live chat: no answer from the server within %.0f s; Reachy uses the text chat", READY_TIMEOUT)
            return LiveOutcome("fallback", NO_ANSWER, None)
        if elapsed >= MAX_SECONDS:
            return LiveOutcome("ended", MAX_LENGTH, self.conversation_id)
        return None

    async def _control(self, text: str) -> LiveOutcome | None:
        try:
            message = json.loads(text)
        except ValueError:
            message = None
        kind = message.get("type") if isinstance(message, dict) else None
        if kind == "ready":
            conversation_id = message.get("conversation_id")
            self.conversation_id = None if conversation_id is None else str(conversation_id)
            self._ready = True
            log.info("live chat %s: ready", self.conversation_id)
        elif kind == "flush":
            await self._flush()
        elif kind == "say":
            self._start_say(message.get("speech_text"), message.get("reason"))
        elif kind == "fallback":
            conversation_id = message.get("conversation_id") or self.conversation_id   # never lose the stored turns
            return LiveOutcome("fallback", str(message.get("reason") or "fallback"),
                               None if conversation_id is None else str(conversation_id))
        elif kind == "end":
            return LiveOutcome("ended", str(message.get("reason") or "end"), self.conversation_id,
                               bool(message.get("risk")))
        else:
            log.warning("live chat: ignored a message it does not know (%s)", kind or "not a JSON object")
        return None

    def _audio(self, pcm: bytes) -> None:
        if self._say_active():
            self.dropped_audio += 1   # GPT-Live's voice never follows or interrupts Reachy's own line
            return
        self.player.add_live(pcm, self.clock())

    async def _read(self) -> None:
        while True:
            message = await self._socket.receive()
            await self._inbox.put(message)
            if message is None:
                return

    async def _send(self, message: dict) -> bool:
        return await self._socket.send(json.dumps(message))

    # ── Reachy's own lines ───────────────────────────────────────────────
    def _start_say(self, text, reason) -> None:
        pieces = chunks(text if isinstance(text, str) else None)
        if not pieces or self.speaker is None or not self.speaker.available:
            log.warning("live chat %s: Reachy cannot say the server's %s line (no voice or no text)",
                        self.conversation_id, reason)
            return
        self._saying += 1
        task = asyncio.create_task(self._say(pieces, still=reason == "risk"), name="live-say")
        self._says.add(task)
        task.add_done_callback(self._says.discard)

    async def _say(self, pieces: list[str], still: bool) -> None:
        """Synthesise one chunk at a time and queue each as it is ready, so the first sound waits for one chunk."""
        try:
            for piece in pieces:
                voiced = await self._run_blocking(self.speaker.synth, piece)
                if voiced is not None:
                    samples, rate = voiced
                    self.player.add_say(to_pcm(resample(np.asarray(samples, dtype=np.float32), int(rate), RATE)),
                                        still)
        except Exception:
            log.exception("live chat %s: Reachy's own line could not be synthesised", self.conversation_id)
        finally:
            self._saying -= 1

    def _say_active(self) -> bool:
        return self._saying > 0 or self.player.has(SAY, STILL)

    # ── microphone and speaker ───────────────────────────────────────────
    async def _mic_loop(self) -> None:
        """The microphone to the server in FRAME_SAMPLES frames, from 'ready' on (nothing would hear them before), and
        silence instead while Reachy says its own line. The microphone's own pace paces the stream, which GPT-Live
        needs: no input audio, no output (spike robot-audio.md section 4)."""
        pending = np.zeros(0, dtype="<i2")
        while True:
            chunk = await self._run_blocking(self.robot.get_audio)
            if chunk is None:
                await self._sleep(MIC_IDLE_SECONDS)
                continue
            if self._ready:
                pending = np.concatenate([pending, mic_pcm(*chunk)])
                while len(pending) >= FRAME_SAMPLES:
                    frame, pending = pending[:FRAME_SAMPLES], pending[FRAME_SAMPLES:]
                    if self._say_active() or self._say_tail:
                        frame = np.zeros(FRAME_SAMPLES, dtype="<i2")
                    if not await self._socket.send(frame.tobytes()):
                        return   # the connection closed: _converse ends the chat
                    self.frames_sent += 1
            await self._sleep(0)

    async def _play_loop(self) -> None:
        """One block every BLOCK_SECONDS on an absolute-deadline clock (silence when nothing may play), as TCU's
        reachy_audio.py:96-127: small sleep overshoots never add up to a slower-than-real-time speaker."""
        next_tick = self.clock()
        while True:
            async with self._speaker_lock:
                block, kind = self.player.block(self.clock())
                await self._run_blocking(self.robot.push_audio, block.astype(np.float32) / 32768)
            await self._after_block(block, kind)
            next_tick += BLOCK_SECONDS
            now = self.clock()
            if now - next_tick > LATE_RESET_SECONDS:
                next_tick = now
            await self._sleep(max(0.0, next_tick - now))

    async def _after_block(self, block: np.ndarray, kind: str | None) -> None:
        """'playing' at the first block with sound, 'idle' IDLE_QUIET_BLOCKS after the last one; the gestures follow."""
        if kind in (SAY, STILL):
            self._say_tail = SAY_TAIL_BLOCKS
        elif self._say_tail:
            self._say_tail -= 1
        if kind is not None and int(np.abs(block.astype(np.int32)).max()) > ONSET_LEVEL:
            self._quiet = 0
            self._set_gesture(None if kind == STILL else "speak")
            if not self._playing:
                self._playing = True
                await self._send({"type": "playing"})
        elif self._playing:
            self._quiet += 1
            if self._quiet >= IDLE_QUIET_BLOCKS:
                self._playing = False
                self._set_gesture(None)
                await self._send({"type": "idle"})

    async def _flush(self, report: bool = True) -> None:
        """Drop everything queued, here and in the SDK, at once; `report`: tell the server Reachy is idle again."""
        async with self._speaker_lock:
            self.player.flush()
            try:
                await self._run_blocking(self.robot.clear_live_audio)
            except Exception:
                log.exception("live chat %s: clearing the speaker failed", self.conversation_id)
        if self._playing:
            self._playing, self._quiet = False, 0
            self._set_gesture(None)
            if report:
                await self._send({"type": "idle"})

    async def _drain(self) -> None:
        """After the server's end or fallback, or a dropped connection, nothing more comes: what is queued (Reachy's
        help line above all) plays now without waiting for the buffer, for DRAIN_LIMIT_SECONDS at most, until stop()."""
        self.player.finish()
        deadline = self.clock() + DRAIN_LIMIT_SECONDS
        speaker = self._workers[2]
        while ((self._saying or self.player.queued) and self.clock() < deadline and not speaker.done()
               and self._stop_reason is None):
            await self._sleep(BLOCK_SECONDS)

    def _set_gesture(self, mode: str | None) -> None:
        if not self.gestures or mode == self._gesture_mode:
            return
        self._gesture_mode = mode
        try:
            self.robot.gesture(mode)   # never blocks
        except Exception:
            log.exception("live chat: gesture %s failed", mode)
