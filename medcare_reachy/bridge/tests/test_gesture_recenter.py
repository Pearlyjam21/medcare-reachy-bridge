"""A gesture started from a pose measured a few degrees off neutral: it starts where the robot is and eases back.

On 3 Oct 2026 (14:53) the head measured 8.6° off right after a check-in began, over the single 8° limit of 0.5.2,
and Reachy kept still for the whole conversation.
"""

import logging
import threading
import time

import numpy as np
import pytest

from medcare_reachy.bridge import gestures
from medcare_reachy.bridge.gestures import (
    FADE_SECONDS, RECENTER_SECONDS, Gestures, Pose, easing_back, min_jerk, pattern)
from medcare_reachy.bridge.media import NEUTRAL_ANTENNAS, ReachyRobot, head_pose
from medcare_reachy.bridge.tests.test_gestures import Mini, awake, wait_for

ORIGIN = Pose(roll=3.5, pitch=7.5, yaw=-3.0, body_yaw=0.8, right=-0.5, left=0.4, x=2.0, y=-1.0, z=-3.0)


def run(origin, cues=(), first="think", halt_at=None, move=None):
    """A stream on a fake clock that starts at `origin` with `first`, then each (time, mode) cue (one at 0 comes
    right after the first target, before any motion has faded in); halted at `halt_at`. Returns [(time, pose)]."""
    state, sent, cues = {"t": 0.0}, [], list(cues)
    stream = None

    def sleep(seconds):
        state["t"] += seconds
        while cues and cues[0][0] <= state["t"]:
            stream.set(cues.pop(0)[1])
        if halt_at is not None and state["t"] >= halt_at:
            stream.halt()

    stream = Gestures(lambda pose: sent.append((state["t"], pose)), start_from=lambda: origin,
                      clock=lambda: state["t"], sleep=sleep, move=move)
    stream.placed()
    stream.set(first)
    assert wait_for(lambda: not stream.moving, timeout=5)
    return sent


def test_the_first_target_is_where_the_robot_was_measured():
    sent = run(ORIGIN, [(0.1, None)])
    assert sent[0] == (0.0, ORIGIN)


def test_it_comes_back_to_the_neutral_pose_within_the_recentering_time_and_stops_there():
    sent = run(ORIGIN, [(0.0, None)])                                  # the patient goes on talking at once
    assert sent[-1][1] == Pose()
    assert RECENTER_SECONDS <= sent[-1][0] <= RECENTER_SECONDS + 0.1
    assert all(abs(value) < 1e-9 for t, pose in sent if t >= RECENTER_SECONDS for value in pose)


def test_it_eases_back_no_faster_than_a_goto_of_the_same_length():
    sent = run(ORIGIN, [(0.0, None)])
    peak = 1.875 * np.abs(np.array(ORIGIN)) / RECENTER_SECONDS        # a minimum-jerk move's top speed
    for (t0, a), (t1, b) in zip(sent, sent[1:]):
        speed = np.abs(np.subtract(b, a)) / (t1 - t0)
        assert np.all(speed <= peak * 1.05 + 1e-9), (t0, speed)
    assert peak.max() < 20                                             # degrees (or mm) per second: gentle


def test_thinking_starts_at_once_on_top_of_it_and_is_the_plain_sway_once_back():
    sent = run(ORIGIN, [(4.0, None)])
    for (t0, a), (t1, b) in zip(sent, sent[1:]):                       # no jump anywhere
        assert np.abs(np.subtract(b, a)).max() < 2.0, t0
    t, pose = next((t, pose) for t, pose in sent if t >= 0.3)
    assert pose.roll > ORIGIN.roll * (1 - min_jerk(t / RECENTER_SECONDS)) + 0.1   # already tilting
    for t, pose in sent:
        if RECENTER_SECONDS + 0.01 < t < 4.0 and t > FADE_SECONDS:
            assert pose == pytest.approx(pattern("think", t), abs=1e-9)
    assert sent[-1][1] == Pose()


def test_with_the_thinking_move_too():
    from medcare_reachy.bridge import moves

    sent = run(ORIGIN, [(1.5, "speak"), (4.0, None)], move=moves.load("inquiring3"))
    assert sent[0][1] == ORIGIN and sent[-1][1] == Pose()
    for (t0, a), (t1, b) in zip(sent, sent[1:]):
        step = np.abs(np.subtract(b, a))
        assert step[:4].max() < 2.0 and step[6:].max() < 2.0, t0      # head and body: no jump


def test_halt_stops_it_at_once_mid_way():
    sent = run(ORIGIN, halt_at=0.3)
    assert sent and sent[-1][0] <= 0.3 + 1e-9
    assert sent[-1][1] != Pose()                                       # stopped where it was: a goto takes over


def test_easing_back_is_all_of_the_way_at_first_and_none_at_the_end():
    assert easing_back(ORIGIN, 0.0) == ORIGIN
    assert easing_back(ORIGIN, RECENTER_SECONDS) == Pose()
    assert easing_back(ORIGIN, 2 * RECENTER_SECONDS) == Pose()
    half = easing_back(ORIGIN, RECENTER_SECONDS / 2)
    assert half == pytest.approx(tuple(value / 2 for value in ORIGIN))


def test_an_unknown_start_pose_starts_nothing(caplog):
    sent = []

    def broken():
        raise ConnectionError("no state from the daemon")

    stream = Gestures(sent.append, start_from=broken, sleep=lambda seconds: time.sleep(0.005))
    stream.placed()
    stream.set("think")
    assert wait_for(lambda: not stream.moving)
    assert sent == [] and "not there now" in caplog.text


# ── through the robot facade ─────────────────────────────────────────────

def sunk(mini, pitch=7.3, roll=3.4, yaw=-2.9, shift=(0.002, -0.001, -0.003)):
    """The head measured as on 3 Oct 14:53: about 8.6° and 3.8 mm off the neutral pose."""
    mini.head = head_pose(yaw, pitch, roll)
    mini.head[:3, 3] = shift
    mini.joints = ([np.radians(0.8)] + [0.0] * 6, [NEUTRAL_ANTENNAS[0] - np.radians(0.5), NEUTRAL_ANTENNAS[1]])


def test_the_check_in_of_3_october_now_gestures_from_where_the_head_was(caplog):
    caplog.set_level(logging.INFO)
    mini = Mini()
    robot = awake(mini)
    sunk(mini)
    robot.gesture("think")
    assert wait_for(lambda: len(mini.targets()) >= 5)
    first = mini.targets()[0]
    assert np.allclose(first["head"], mini.head, atol=1e-9)                       # no jump at the first target
    assert first["body_yaw"] == pytest.approx(np.radians(0.8))
    assert first["antennas"] == pytest.approx(mini.joints[1])
    assert "easing back from there" in caplog.text and "head_deg': 8.6" in caplog.text
    robot.gesture(None)
    assert wait_for(lambda: not robot._gestures.moving, timeout=5)
    last = mini.targets()[-1]                                                     # and back to the neutral pose
    assert np.allclose(last["head"], np.eye(4)) and last["body_yaw"] == 0.0
    assert last["antennas"] == pytest.approx(list(NEUTRAL_ANTENNAS))


def test_with_the_motors_off_a_moderate_offset_still_starts_nothing():
    mini = Mini(motors="disabled")
    robot = awake(mini)
    sunk(mini)
    robot.gesture("think")
    time.sleep(0.1)
    assert mini.targets() == [] and not robot._gestures.moving


def test_switched_off_and_on_mid_gesture_it_waits_for_the_next_goto_even_when_close(monkeypatch):
    """The original safety rule stands: after the motors were off, no gesture until a goto put the robot back."""
    monkeypatch.setattr(gestures, "READY_CHECK_SECONDS", 0.02)
    mini = Mini()
    robot = awake(mini)
    robot.gesture("think")
    assert wait_for(lambda: len(mini.targets()) >= 10)
    mini.motors = "disabled"
    assert wait_for(lambda: not robot._gestures.moving)
    count = len(mini.targets())
    mini.motors = "enabled"
    sunk(mini)
    robot.gesture("think")
    time.sleep(0.1)
    assert len(mini.targets()) == count
    robot.hold_head()
    robot.gesture("think")
    assert wait_for(lambda: len(mini.targets()) > count)
    assert np.allclose(mini.targets()[count]["head"], mini.head, atol=1e-9)        # from where it is


def test_a_goto_while_easing_back_takes_over_at_once():
    mini = Mini()
    robot = awake(mini)
    sunk(mini)
    robot.gesture("think")
    assert wait_for(lambda: mini.targets())
    done = threading.Event()
    threading.Thread(target=lambda: (robot.sleep(), done.set())).start()
    assert done.wait(2)
    names = mini.names()
    assert "set_target" not in names[names.index("goto_sleep"):] and not robot._gestures.moving
