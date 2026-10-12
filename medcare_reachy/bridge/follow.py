"""Face following while a dose is watched: the head turns to keep the verified patient's face centred.

The server names the patient's face in every monitor reply (`target_box`, normalised to the streamed 4:3 frame).
`FaceFollower` turns that into head moves with the SDK's `look_at_image` (a pixel of the camera's own image), at
most once per INTERVAL and only when the face has left a DEADBAND around the last target, so a patient who sits
still gets a still robot, and the hand-to-mouth detector never sees a head that twitches with every landmark.
No box (the patient looked away, or is not verified) leaves the head where it is; `release()` recentres it.

Pure logic: the clock and the robot call are passed in, so the tests run on a fake clock.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

INTERVAL = 0.4            # seconds between head moves
DEADBAND = 0.06           # normalised distance the face must move before the head follows
MOVE_SECONDS = 0.35       # one move's duration: smooth, and done before the next one is due
RELEASE_AFTER = 3.0       # no verified face this long: the head goes back to neutral (keeps the camera's view wide)


class FaceFollower:
    def __init__(self, look_at, recentre, *, clock=time.monotonic, interval: float = INTERVAL,
                 deadband: float = DEADBAND, release_after: float = RELEASE_AFTER) -> None:
        """`look_at(u, v, duration)` takes the face centre in normalised stream coordinates (0-1); `recentre()`
        brings the head back to neutral. Both are called on the caller's thread."""
        self.look_at, self.recentre = look_at, recentre
        self.clock, self.interval, self.deadband, self.release_after = clock, interval, deadband, release_after
        self._last_move = float("-inf")
        self._target: tuple[float, float] | None = None
        self._seen = float("-inf")
        self.active = False
        self.moves = 0

    def start(self) -> None:
        self.active = True
        self._target = None
        self._seen = self.clock()

    def stop(self) -> None:
        """Stop following; the head stays put (the next prompt or sleep moves it)."""
        self.active = False
        self._target = None

    def update(self, target_box, now: float | None = None) -> bool:
        """One monitor reply. Returns True when the head was moved (or recentred)."""
        if not self.active:
            return False
        now = self.clock() if now is None else now
        if not target_box:
            if self._target is not None and now - self._seen >= self.release_after:
                self._target = None
                self._call(self.recentre)
                return True
            return False
        self._seen = now
        x, y, w, h = (float(v) for v in target_box[:4])
        centre = (min(1.0, max(0.0, x + w / 2)), min(1.0, max(0.0, y + h / 2)))
        if now - self._last_move < self.interval:
            return False
        if self._target is not None and max(abs(centre[0] - self._target[0]), abs(centre[1] - self._target[1])) < self.deadband:
            return False
        self._target = centre
        self._last_move = now
        self.moves += 1
        self._call(self.look_at, centre[0], centre[1], MOVE_SECONDS)
        return True

    @staticmethod
    def _call(fn, *args) -> None:
        try:
            fn(*args)
        except Exception:   # the head is a nicety; the dose goes on
            log.exception("face following failed")
