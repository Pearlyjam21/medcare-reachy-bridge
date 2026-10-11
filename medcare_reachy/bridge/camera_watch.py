"""Camera watchdog (spec 2026-10-11 A2, A3): notices when the camera stops handing this app new frames while a
monitor session streams, asks the robot for its camera back, and gives up after a while, so a search ends with a
reason instead of looking around blind for 10 minutes.

On 11 Oct 2026 the TCU companion connected with media_backend="no_media", which in reachy_mini 1.11.0 releases the
daemon's camera. The SDK's camera reader then gave one stale frame and None for ever, and the app searched for 10
minutes without a log line.

Pure logic: the clock, the recovery and the daemon's media status are passed in, so the tests run on a fake clock.
tick() may block for as long as `recover` takes (a daemon release/acquire cycle): call it off the event loop.
"""

import logging
import threading
import time
from collections import deque
from typing import Callable

log = logging.getLogger(__name__)

CAMERA_STALL_SECONDS = 2.0       # no new frame this long while a session streams: the camera has stalled
CAMERA_RECOVER_INTERVAL = 5.0    # at most one recovery attempt per this long
CAMERA_GIVE_UP_SECONDS = 15.0    # no new frame this long after the stall began: the camera is unavailable
CAMERA_HEALTHY_SECONDS = 5.0     # frames must come without a gap this long before a stall is over
FPS_WINDOW = 10.0                # the heartbeat's camera fps, over the same window as the stream's other rates

IDLE, OK, STALLED, RECOVERING, UNAVAILABLE = "idle", "ok", "stalled", "recovering", "unavailable"
NOT_WORKING = frozenset({STALLED, RECOVERING, UNAVAILABLE})   # the Reachy card warns about these


class CameraWatchdog:
    """The camera's state while a monitor session streams:

    - idle: no session, so the camera isn't read (0 fps is normal);
    - ok: new frames keep coming (a slow camera is ok: only a gap of stall_seconds counts);
    - stalled: no new frame for stall_seconds, or the last recovery attempt failed;
    - recovering: a recovery attempt is running, or succeeded and the first new frame is awaited;
    - unavailable: the camera has not stayed healthy for give_up_seconds since the stall began. It outlasts the
      session (detach() keeps it, so the heartbeat still reports it while the robot sleeps) until the next session
      starts a fresh try.

    A stall is an episode, not a moment: it ends only when frames have come without a gap of stall_seconds for
    healthy_seconds. Until then a new frame shows "ok" but changes nothing else, a new stall continues the same
    episode, and give_up_seconds count from the episode's first stall. On 11 Oct the camera gave one stale frame
    after each recovery and then nothing: with every frame ending the stall, the recoveries came every ~2 s and the
    state never left ok/recovering. The recovery limit (once per recover_interval) holds for the whole session; only
    attach() lifts it.

    The stream calls attach()/detach() with its session, frame() for every new frame and tick() for every read that
    found none. Only tick() calls `recover` and `media_status`, outside the lock, so the heartbeat's snapshot() and the
    slot session's `state` never wait for a recovery. `media_released` (optional, a quick read) says whether a
    recovery left the app's media released: the give-up then also says the speaker and microphone are off.
    """

    def __init__(self, recover: Callable[[], str], media_status: Callable[[], dict | None],
                 clock: Callable[[], float] = time.monotonic, stall_seconds: float = CAMERA_STALL_SECONDS,
                 recover_interval: float = CAMERA_RECOVER_INTERVAL, give_up_seconds: float = CAMERA_GIVE_UP_SECONDS,
                 *, wall: Callable[[], float] = time.time, healthy_seconds: float = CAMERA_HEALTHY_SECONDS,
                 media_released: Callable[[], bool] | None = None):
        self._recover, self._media_status, self._media_released = recover, media_status, media_released
        self.clock, self.wall = clock, wall
        self.stall_seconds, self.recover_interval = stall_seconds, recover_interval
        self.give_up_seconds, self.healthy_seconds = give_up_seconds, healthy_seconds
        self._lock = threading.Lock()
        self._frames: deque = deque()   # when each new frame came, for the fps
        self._state = IDLE
        self._since = wall()            # unix seconds of the last state change
        self._media: dict | None = None   # the daemon's media status, as read during the current stall episode
        self._generation = 0            # bumped by attach() and detach(): a late recovery result is ignored
        self._reset(clock())

    # ── the stream's interface ───────────────────────────────────────────
    @property
    def state(self) -> str:
        with self._lock:
            self._give_up_if_due(self.clock())   # also while a recovery blocks in tick()
            return self._state

    def attach(self) -> None:
        """A monitor session starts streaming: a fresh try, whatever happened before."""
        with self._lock:
            self._generation += 1
            self._reset(self.clock())
            self._frames.clear()
            self._media = None
            self._set(OK)

    def detach(self) -> None:
        """No session any more: the camera isn't read. "unavailable" stays until the next attach()."""
        with self._lock:
            self._generation += 1
            self._give_up_if_due(self.clock())
            if self._state != UNAVAILABLE:
                self._media = None
                self._set(IDLE)

    def frame(self) -> None:
        """A new frame came. It shows "ok", but an open stall episode ends only after healthy_seconds of frames
        without a gap (a gap of stall_seconds starts the count again)."""
        with self._lock:
            now = self.clock()
            self._frames.append(now)
            self._prune(now)
            if self._stall_since is not None:
                if self._healthy_since is None or now - self._last_frame >= self.stall_seconds:
                    self._healthy_since = now   # the healthy stretch starts with this frame
                elif now - self._healthy_since >= self.healthy_seconds:
                    self._end_episode(now)
                if self._state in NOT_WORKING:
                    self._set(OK)
            self._last_frame = now

    def tick(self) -> str:
        """A read found no new frame. Notices a stall (one ERROR line, with the daemon's media status), tries a
        recovery at most once per recover_interval, and gives up give_up_seconds after the stall began.
        Never raises; returns the state."""
        try:
            return self._tick()
        except Exception:   # the watchdog must never break the stream
            log.exception("camera watchdog failed")
            return self._state

    def snapshot(self) -> dict:
        """For the heartbeat: the state, new frames per second over the last FPS_WINDOW s, the unix time of the last
        state change, and the daemon's media status read during the current stall episode (None while the state is
        ok)."""
        with self._lock:
            now = self.clock()
            self._give_up_if_due(now)
            self._prune(now)
            return {"state": self._state, "fps": round(len(self._frames) / FPS_WINDOW, 1), "since": self._since,
                    "media": None if self._media is None or self._state == OK else dict(self._media)}

    # ── inside ───────────────────────────────────────────────────────────
    def _tick(self) -> str:
        with self._lock:
            now = self.clock()
            self._give_up_if_due(now)
            if self._state in (IDLE, UNAVAILABLE):
                return self._state
            stalled = self._state == OK and now - self._last_frame >= self.stall_seconds
            if self._state == OK and not stalled:
                return OK
            new_stall = False
            if stalled:
                self._healthy_since = None
                new_stall = self._stall_since is None   # else it continues the open episode
                if new_stall:
                    self._stall_since = now
                else:
                    log.warning("camera stalled again: no new frame for %.1f s, %.1f s after the stall began "
                                "(%d recovery attempt(s), %d failed)", now - self._last_frame, now - self._stall_since,
                                self._attempts, self._failures)
                self._set(STALLED)
                self._give_up_if_due(now)
                if self._state == UNAVAILABLE:
                    return UNAVAILABLE
            if now - self._last_attempt < self.recover_interval:
                return self._state
            self._last_attempt = now
            self._attempts += 1
            attempt, generation, gap = self._attempts, self._generation, now - self._last_frame
            self._set(RECOVERING)
        media = self._read_media_status()   # HTTP and the recovery run without the lock
        if new_stall:
            log.error("camera stalled: no new frame for %.1f s while a session streams; daemon media status: %s",
                      gap, media)
        result = self._attempt_recovery()
        failed = result.startswith("failed")
        if failed:
            log.warning("camera recovery attempt %d failed: %s", attempt, result)
        else:
            log.info("camera recovery attempt %d: %s; waiting for a new frame", attempt, result)
        with self._lock:
            if generation != self._generation:
                return self._state   # the session ended or restarted meanwhile: this result no longer applies
            self._media = media
            if failed:
                self._failures += 1
            self._give_up_if_due(self.clock())
            if failed and self._state == RECOVERING:
                self._set(STALLED)
            return self._state

    def _read_media_status(self) -> dict | None:
        try:
            status = self._media_status()
        except Exception as exc:
            log.warning("could not read the daemon's media status: %r", exc)
            return None
        return status if isinstance(status, dict) else None

    def _attempt_recovery(self) -> str:
        try:
            return str(self._recover())
        except Exception as exc:   # counted as a failed attempt; the stream goes on
            log.exception("camera recovery raised")
            return f"failed: {exc!r}"

    def _give_up_if_due(self, now: float) -> None:
        """With the lock held."""
        if self._state in (STALLED, RECOVERING) and now - self._stall_since >= self.give_up_seconds:
            log.error("camera unavailable: no steady frames for %.1f s since the stall began (%d recovery attempt(s), "
                      "%d failed); a search ends with camera_unavailable",
                      now - self._stall_since, self._attempts, self._failures)
            self._set(UNAVAILABLE)
            if self._media_off():
                log.error("this app's media is still released: the speaker and microphone are off until the robot's "
                          "media comes back (asked for again at every heartbeat)")

    def _media_off(self) -> bool:
        """With the lock held: whether a recovery left the app's media released. Only for a log line."""
        try:
            return self._media_released is not None and bool(self._media_released())
        except Exception:
            return False

    def _reset(self, now: float) -> None:
        """A fresh session (and the constructor): everything, the recovery limit included."""
        self._last_frame = now
        self._last_attempt = float("-inf")   # only a new session lifts the limit of one recovery per interval
        self._clear_episode()

    def _end_episode(self, now: float) -> None:
        """With the lock held: the feed has been healthy for healthy_seconds. The recovery limit stays."""
        log.info("camera recovered: new frames for %.1f s without a gap; the stall began %.1f s ago "
                 "(%d recovery attempt(s), %d failed)", now - self._healthy_since, now - self._stall_since,
                 self._attempts, self._failures)
        self._clear_episode()

    def _clear_episode(self) -> None:
        self._stall_since = None
        self._healthy_since = None   # when the frames of the current healthy stretch began
        self._attempts = 0
        self._failures = 0
        self._media = None

    def _set(self, state: str) -> None:
        if state != self._state:
            self._state, self._since = state, self.wall()

    def _prune(self, now: float) -> None:
        while self._frames and self._frames[0] < now - FPS_WINDOW:
            self._frames.popleft()
