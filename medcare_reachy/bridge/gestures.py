"""Reachy's body language in a check-in conversation: it visibly thinks while a reply is being prepared, and its
antennas stay lively while it speaks.

One thread streams targets: small offsets from the neutral pose, at GESTURE_HZ (MOVE_HZ while a recorded move plays).
Every motion fades in and out over FADE_SECONDS, so one motion flows into the next and the robot comes back to the
neutral pose without a jump (the servos have no speed limit of their own: the daemon sends each target straight to
them). Each offset is clamped to LIMITS whatever the mix, or to MOVE_LIMITS while the recorded move plays.

Thinking starts with Pollen's recorded "inquiring3" move (moves.py: played as offsets from its first frame, without
its sound), then flows from its last frame into a slow sway until Reachy speaks. It plays at most once every
MOVE_AGAIN_SECONDS. Cut short (Reachy starts speaking, or the patient goes on talking), it fades out over
MOVE_FADE_SECONDS and slows down as it does (it plays at the square of its fade level's speed), so a cut at any
moment moves no part faster than the recording itself does. (A plain 0.5 s fade, cut while the move lowers its
antenna at the recording's 280°/s, made 430°/s.) Speaking moves the antennas only.

A stream's first target is where the robot is, so a stream only starts from where it knows the robot is: after a
goto put it in the neutral pose (`placed()`), and only if it is measured there or close to it (`start_from()`).
Measured near the neutral pose, the first target is that pose itself; a little further (the head sank a few degrees),
the first target is the pose measured, and the stream eases back to the neutral pose over RECENTER_SECONDS, under the
motion. A goto elsewhere (`halt()` must come first: while a goto runs, the daemon ignores every target), a stream
that ended early (motors off, the daemon gone) and a robot measured far from that pose (the motors switched off and
on while it was still, a hand holding the head) all stop motion until the next `placed()`.
"""

import logging
import math
import threading
import time
from typing import NamedTuple

log = logging.getLogger(__name__)

GESTURE_HZ = 25.0          # targets per second; the daemon's control loop runs at 50 Hz
MOVE_HZ = 50.0             # while the recorded move plays: its own frame rate, so no step is larger than recorded
FADE_SECONDS = 0.5         # every motion fades in and out over this long
MOVE_FADE_SECONDS = 0.75   # ... except thinking while its recorded move plays, which fades out over this long
RECENTER_SECONDS = 0.9    # from a pose measured a little off neutral, the stream eases back to it over this long
MOVE_AGAIN_SECONDS = 10.0  # thinking again sooner than this after the move started goes straight to the sway
READY_CHECK_SECONDS = 1.0  # how often the stream checks that the motors are on
MODES = ("think", "speak")

# Thinking (degrees): the head tilts and looks a little up, the whole robot turns slowly to and fro, and the
# antennas swing in mirror (Pollen's "breathing" pattern, slower).
THINK_ROLL, THINK_PITCH = 7.0, -4.0
THINK_TURN, THINK_TURN_HZ = 4.0, 0.15
THINK_ANTENNA, THINK_ANTENNA_HZ = 12.0, 0.3
# Speaking: antennas only, a little quicker and out of step with each other.
SPEAK_ANTENNA, SPEAK_ANTENNA_HZ = 8.0, 0.7
# Hard limits on each offset from the neutral pose (degrees; the head's position in mm) for the sway and speaking.
LIMITS = {"roll": 10.0, "pitch": 10.0, "yaw": 8.0, "body_yaw": 5.0, "right": 20.0, "left": 20.0,
          "x": 0.0, "y": 0.0, "z": 0.0}
# The same while the recorded move plays. Its offsets from its first frame: head roll -6..17°, pitch -6..4°,
# yaw -7..1°, body 0, right antenna 0, left antenna 0..100°, head position within 5 mm.
MOVE_LIMITS = {"roll": 20.0, "pitch": 10.0, "yaw": 10.0, "body_yaw": 5.0, "right": 20.0, "left": 110.0,
               "x": 6.0, "y": 6.0, "z": 6.0}


class Pose(NamedTuple):
    """Offsets from the neutral pose: head roll, pitch and yaw in degrees (world frame), body yaw, the antennas
    (degrees), and the head's position in mm (x forward, y left, z up)."""
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    body_yaw: float = 0.0
    right: float = 0.0
    left: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


def pattern(mode: str, t: float) -> Pose:
    """A motion at full strength, `t` seconds into the stream."""
    if mode == "think":
        turn = THINK_TURN * math.sin(2 * math.pi * THINK_TURN_HZ * t)
        swing = THINK_ANTENNA * math.sin(2 * math.pi * THINK_ANTENNA_HZ * t)
        # The head target is in the world frame: turning it with the body turns the whole robot.
        return Pose(THINK_ROLL, THINK_PITCH, turn, turn, swing, -swing)
    if mode == "speak":
        phase = 2 * math.pi * SPEAK_ANTENNA_HZ * t
        return Pose(right=SPEAK_ANTENNA * math.sin(phase), left=SPEAK_ANTENNA * math.sin(phase + math.pi / 2))
    raise ValueError(f"unknown gesture {mode}")


def smooth(level: float) -> float:
    """0..1 -> 0..1, starting and ending with zero speed (smoothstep)."""
    return level * level * (3 - 2 * level)


def min_jerk(share: float) -> float:
    """0..1 -> 0..1, starting and ending with zero speed and acceleration (the SDK's goto profile)."""
    share = min(1.0, max(0.0, share))
    return share ** 3 * (10 - 15 * share + 6 * share * share)


def blend(a: Pose, b: Pose, weight: float) -> Pose:
    return Pose(*((1 - weight) * x + weight * y for x, y in zip(a, b)))


def thinking(move, since_move: float | None, t: float) -> Pose | None:
    """The thinking motion while its recorded move is involved, or None once only the sway is left (also when no
    move plays this time).

    `since_move`: how far into the move it is, in seconds of the recording (None: not playing); `t`: the stream's
    time, which the sway follows. The move from its first frame (exactly the neutral pose), then FADE_SECONDS from
    its last frame into the sway.
    """
    if move is None or since_move is None or since_move >= move.duration + FADE_SECONDS:
        return None
    if since_move < move.duration:
        return move.at(since_move)
    return blend(move.at(move.duration), pattern("think", t), smooth((since_move - move.duration) / FADE_SECONDS))


def easing_back(origin: Pose, t: float) -> Pose:
    """How much of the way back to the neutral pose is still to go `t` seconds into a stream that started at
    `origin` (offsets from that pose): all of it at first, none from RECENTER_SECONDS on."""
    left = 1.0 - min_jerk(t / RECENTER_SECONDS)
    return Pose(*(left * value for value in origin))


def mix(levels: dict, t: float, motions: dict | None = None, limits: dict = LIMITS) -> Pose:
    """The motions weighted by their fade levels; all levels at 0 is exactly the neutral pose. `motions` gives a
    mode's pose at this moment in place of its pattern (the thinking move)."""
    total = [0.0] * len(Pose._fields)
    for mode, level in levels.items():
        if level > 0:
            weight = smooth(level)
            motion = (motions or {}).get(mode)
            for i, value in enumerate(pattern(mode, t) if motion is None else motion):
                total[i] += weight * value
    return Pose(*(max(-limits[name], min(limits[name], value)) for name, value in zip(Pose._fields, total)))


class Gestures:
    """Streams the current motion to `send(pose)` from one background thread, only while one is wanted.

    `set()`, `halt()` and `placed()` may be called from any thread and never wait for a motion. `ready()` says
    whether the motors are on. `start_from()`, asked before a stream's first target, says where that target must be:
    the robot's offsets from the neutral pose (`Pose()` when it is at that pose), or None when no motion may start
    from where it is. `move` is the recorded move thinking starts with (moves.RecordedMove), or None for the sway
    alone. A failing check or a failing `send` (the daemon gone) ends the stream with a warning.
    """

    def __init__(self, send, *, ready=lambda: True, start_from=lambda: Pose(), move=None, clock=time.monotonic,
                 sleep=time.sleep, hz=GESTURE_HZ, move_hz=MOVE_HZ):
        self._send, self._ready, self._start_from, self._move = send, ready, start_from, move
        self.clock, self._sleep = clock, sleep
        self._period, self._move_period = 1.0 / hz, 1.0 / move_hz
        self._lock = threading.Lock()
        self._mode: str | None = None
        self._levels = dict.fromkeys(MODES, 0.0)
        self._thread: threading.Thread | None = None
        self._run_id = 0          # a stream sends only while its run is current: halt() starts a new one
        self._placed = False      # the robot is in the neutral pose, where every stream starts
        self._move_from: float | None = None   # when thinking last started the recorded move (clock time)
        self._move_time: float | None = None   # how far into it the stream is (it plays at the speed of its fade)

    @property
    def moving(self) -> bool:
        with self._lock:
            return self._thread is not None

    def set(self, mode: str | None) -> None:
        """Fade into `mode` ("think" or "speak"), or back to the neutral pose (None). Ignored until `placed()`.

        Thinking starts with the recorded move, unless that started less than MOVE_AGAIN_SECONDS ago.
        """
        if mode is not None and mode not in MODES:
            raise ValueError(f"unknown gesture {mode}")
        with self._lock:
            if not self._placed:
                return
            if mode == "think" and self._mode != "think" and self._move is not None:
                now = self.clock()
                if self._move_from is None or now - self._move_from >= MOVE_AGAIN_SECONDS:
                    self._move_from, self._move_time = now, 0.0
            self._mode = mode
            if mode is None or self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, args=(self._run_id,), name="medcare-gesture",
                                            daemon=True)
            self._thread.start()

    def placed(self) -> None:
        """A goto has just put the robot in the neutral pose: motions may start from there."""
        with self._lock:
            self._placed = True

    def halt(self) -> None:
        """Stop streaming now, without fading out, and wait until the thread has: a goto takes over the robot.

        No motion starts again until `placed()`, also when another thread asks for one meanwhile.
        """
        with self._lock:
            self._run_id += 1
            self._mode, self._placed, thread, self._thread = None, False, self._thread, None
            self._levels = dict.fromkeys(MODES, 0.0)
        if thread is not None and thread is not threading.current_thread():
            thread.join(1.0)

    @staticmethod
    def _check(check, unknown: bool) -> bool:
        try:
            return bool(check())
        except Exception:
            return unknown

    def _origin(self) -> Pose | None:
        """Where the first target must be; unknown counts as away."""
        try:
            return self._start_from()
        except Exception:
            return None

    def _end(self, run_id: int) -> None:
        """The stream stopped early, so the robot may be anywhere: no motion until the next `placed()`."""
        with self._lock:
            if run_id == self._run_id:
                self._thread, self._mode, self._placed = None, None, False
                self._levels = dict.fromkeys(MODES, 0.0)

    def _run(self, run_id: int) -> None:
        started = last = origin = None
        checked = float("-inf")
        while True:
            now = self.clock()
            if started is None:
                origin = self._origin()
                if origin is None:
                    log.warning("gestures off until the robot is put back in its neutral pose: it is not there now")
                    self._end(run_id)
                    return
                started = last = now
            if now - checked >= READY_CHECK_SECONDS:
                checked = now
                # Unknown counts as on: the session only gestures after placing the robot.
                if not self._check(self._ready, unknown=True):
                    log.warning("gestures stopped: the robot's motors are off (none until it is placed again)")
                    self._end(run_id)
                    return
            with self._lock:
                if run_id != self._run_id:   # halted
                    return
                playing = (self._move is not None and self._move_time is not None
                           and self._move_time < self._move.duration + FADE_SECONDS)
                for mode in MODES:
                    goal, level = (1.0 if mode == self._mode else 0.0), self._levels[mode]
                    fade = MOVE_FADE_SECONDS if mode == "think" and playing and goal < level else FADE_SECONDS
                    step = (now - last) / fade
                    self._levels[mode] = min(goal, level + step) if goal > level else max(goal, level - step)
                levels = dict(self._levels)
                if self._move_time is not None:   # the move plays at the speed of its fade, squared
                    self._move_time += (now - last) * smooth(levels["think"]) ** 2
                since_move = self._move_time
                idle = self._mode is None and not any(levels.values())
            last = now
            t = now - started
            think = thinking(self._move, since_move, t) if levels["think"] > 0 else None
            pose = mix(levels, t, {"think": think}, LIMITS if think is None else MOVE_LIMITS)
            easing = origin != Pose() and t < RECENTER_SECONDS
            if easing:
                pose = Pose(*(a + b for a, b in zip(pose, easing_back(origin, t))))
            try:
                self._send(pose)
            except Exception as exc:   # motion is never worth breaking the conversation over
                log.warning("gestures stopped: %r", exc)
                self._end(run_id)
                return
            if idle and not easing:   # that was exactly the neutral pose
                with self._lock:
                    if run_id != self._run_id:
                        return
                    if self._mode is None:   # nothing new was asked for while the neutral pose went out
                        self._thread = None
                        return
            self._sleep(self._period if think is None else self._move_period)
