"""Where emotion is computed: on the robot by default, on the laptop when the robot can't keep up.

The stream tells the governor about every scoring opportunity (`due`: a frame with a verified face when a score
is due), every completed score (`scored`, with its latency) and every failure (`failed`). In "auto" mode the
robot is abandoned for the laptop when, over the last WINDOW seconds of opportunities, it completed less than
MIN_SHARE of them (the scorer was still busy when the next frame was due), when the last MAX_SLOW scores each
took longer than one frame interval at the target rate, or when MAX_FAILURES scores in a row raised. On the
laptop the robot streams its frames (it already does when landmarks run on the server) and the server scores
them with its own copy of the face model. After RECOVER_AFTER seconds the robot is tried again, with fresh
counters, so a passing load spike doesn't cost the rest of the day.

"robot" and "laptop" pin the choice; the governor then only measures.
"""

from __future__ import annotations

import logging
import time
from collections import deque

log = logging.getLogger(__name__)

ROBOT, LAPTOP = "robot", "laptop"
MODES = ("auto", ROBOT, LAPTOP)
TARGET_FPS = 2.0           # the server accepts a robot score at most twice a second (runner.EMOTION_INTERVAL)
WINDOW = 10.0              # seconds of opportunities the share is measured over
MIN_SHARE = 0.5            # completed / due, below which the robot is too slow
MAX_SLOW = 3               # consecutive scores slower than one frame interval
MAX_FAILURES = 3           # consecutive exceptions
RECOVER_AFTER = 60.0       # seconds on the laptop before the robot is tried again (auto mode)


class ComputeGovernor:
    def __init__(self, mode: str = "auto", *, target_fps: float = TARGET_FPS, window: float = WINDOW,
                 min_share: float = MIN_SHARE, max_slow: int = MAX_SLOW, max_failures: int = MAX_FAILURES,
                 recover_after: float = RECOVER_AFTER, clock=time.monotonic) -> None:
        if mode not in MODES:
            raise ValueError(f"emotion compute must be one of {MODES}")
        self.mode = mode
        self.budget = 1.0 / target_fps
        self.window, self.min_share = window, min_share
        self.max_slow, self.max_failures, self.recover_after = max_slow, max_failures, recover_after
        self.clock = clock
        self.compute = LAPTOP if mode == LAPTOP else ROBOT
        self.reason = "setting" if mode != "auto" else "default"
        self.fell_back_at: float | None = None
        self.fallbacks = 0
        self.recoveries = 0
        self._reset()

    def _reset(self) -> None:
        self._due: deque = deque()
        self._done: deque = deque()
        self._slow = 0
        self._failures = 0
        self._first_due: float | None = None

    # ── what the stream asks ──
    def on_robot(self, now: float | None = None) -> bool:
        """Whether the next score runs on the robot. Also where a laptop spell ends (auto mode)."""
        if self.mode == "auto" and self.compute == LAPTOP and self.fell_back_at is not None:
            now = self.clock() if now is None else now
            if now - self.fell_back_at >= self.recover_after:
                self.compute, self.reason, self.fell_back_at = ROBOT, "retry", None
                self.recoveries += 1
                self._reset()
                log.info("emotion: trying the robot again after %.0f s on the laptop", self.recover_after)
        return self.compute == ROBOT

    # ── what the stream reports ──
    def due(self, now: float | None = None) -> None:
        """A frame with a verified face came when a score was due (whether or not the scorer was free)."""
        now = self.clock() if now is None else now
        if self._first_due is None:
            self._first_due = now
        self._due.append(now)
        self._trim(now)
        self._check(now)

    def scored(self, latency: float, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        self._done.append(now)
        self._failures = 0
        self._slow = self._slow + 1 if latency > self.budget else 0
        self._trim(now)
        self._check(now)

    def failed(self, exc: BaseException | None = None, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        self._failures += 1
        self._trim(now)
        self._check(now, exc)

    # ── numbers for the status page / heartbeat ──
    def share(self, now: float | None = None) -> float | None:
        """Completed scores per opportunity over the window, or None before there were any."""
        now = self.clock() if now is None else now
        self._trim(now)
        return None if not self._due else min(1.0, len(self._done) / len(self._due))

    def status(self) -> dict:
        return {"mode": self.mode, "compute": self.compute, "reason": self.reason, "fallbacks": self.fallbacks,
                "recoveries": self.recoveries}

    # ── the rule ──
    def _trim(self, now: float) -> None:
        cutoff = now - self.window
        for times in (self._due, self._done):
            while times and times[0] < cutoff:
                times.popleft()

    def _check(self, now: float, exc: BaseException | None = None) -> None:
        if self.mode != "auto" or self.compute != ROBOT:
            return
        reason = None
        if self._failures >= self.max_failures:
            reason = f"{self._failures} scores in a row failed ({exc!r})" if exc else f"{self._failures} scores in a row failed"
        elif self._slow >= self.max_slow:
            reason = f"{self._slow} scores in a row took over {self.budget:.2f} s"
        elif (self._first_due is not None and now - self._first_due >= self.window and self._due
              and len(self._done) < self.min_share * len(self._due)):
            reason = (f"scored {len(self._done)} of {len(self._due)} due frames in the last {self.window:.0f} s "
                      f"(below {self.min_share:.0%})")
        if reason is None:
            return
        self.compute, self.reason, self.fell_back_at = LAPTOP, reason, now
        self.fallbacks += 1
        log.warning("emotion: the robot can't keep up (%s); scoring on the laptop from the streamed frames for "
                    "the next %.0f s", reason, self.recover_after)
