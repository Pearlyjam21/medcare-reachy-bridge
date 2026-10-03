"""Pollen's "inquiring3" while Reachy thinks: the vendored move, how the gesture stream plays it, and how it is cut
when Reachy starts to speak."""

import hashlib
import json
import time
import types

import numpy as np
import pytest

from medcare_reachy.bridge import gestures, moves
from medcare_reachy.bridge.gestures import (
    FADE_SECONDS, GESTURE_HZ, MOVE_AGAIN_SECONDS, MOVE_FADE_SECONDS, MOVE_HZ, MOVE_LIMITS, Gestures, Pose, pattern)
from medcare_reachy.bridge.media import NEUTRAL_ANTENNAS, ReachyRobot, head_pose
from medcare_reachy.bridge.tests.test_gestures import Mini, awake, wait_for

MOVE = moves.load()
OWN_STEP = np.abs(np.diff(MOVE.offsets, axis=0)).max(axis=0)   # the recording's largest change between frames
DRIVEN = [i for i, name in enumerate(Pose._fields) if name not in ("body_yaw", "right")]   # what the move moves
SPEAK_STEP = gestures.SPEAK_ANTENNA * 2 * np.pi * gestures.SPEAK_ANTENNA_HZ / MOVE_HZ     # speaking's own, per tick


# ── the move itself ──────────────────────────────────────────────────────

def test_the_vendored_move_is_pollens_inquiring3_without_its_sound():
    raw = (moves.MOVES_DIR / "inquiring3.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == moves.SHA256["inquiring3"]
    data = json.loads(raw)
    assert data["description"] == "A fast movement that lets you ask a question."
    assert len(data["time"]) == 147 and MOVE.duration == pytest.approx(2.92)
    assert sorted(path.name for path in moves.MOVES_DIR.iterdir()) == ["NOTICE", "inquiring3.json"]   # no .ogg
    assert "Apache License" in (moves.MOVES_DIR / "NOTICE").read_text(encoding="utf-8")


def test_an_altered_move_file_is_refused(tmp_path, monkeypatch):
    (tmp_path / "inquiring3.json").write_bytes((moves.MOVES_DIR / "inquiring3.json").read_bytes() + b" ")
    monkeypatch.setattr(moves, "MOVES_DIR", tmp_path)
    moves.load.cache_clear()
    try:
        with pytest.raises(ValueError):
            moves.load("inquiring3")
    finally:
        moves.load.cache_clear()


def test_it_starts_at_the_neutral_pose_and_stays_inside_its_limits():
    assert MOVE.at(0.0) == Pose() and MOVE.at(-1.0) == Pose()
    assert MOVE.at(99.0) == MOVE.at(MOVE.duration)
    low, high = MOVE.offsets.min(axis=0), MOVE.offsets.max(axis=0)
    for i, name in enumerate(Pose._fields):
        assert -MOVE_LIMITS[name] < low[i] and high[i] < MOVE_LIMITS[name], name
    assert high[Pose._fields.index("left")] == pytest.approx(99.9, abs=0.1)   # the left antenna's question mark
    assert np.allclose(MOVE.offsets[:, Pose._fields.index("right")], 0)
    assert np.allclose(MOVE.offsets[:, Pose._fields.index("body_yaw")], 0)


def test_it_is_played_as_the_sdk_interpolates_it():
    t = 1.01                                              # halfway between two frames
    i = int(np.searchsorted(MOVE.times, t)) - 1
    expected = (MOVE.offsets[i] + MOVE.offsets[i + 1]) / 2
    assert np.allclose(MOVE.at(t), expected)


@pytest.mark.parametrize("angles", [(0, 0, 0), (20, -5, 7), (-35, 12, -3.5), (5, 30, 15)])
def test_head_angles_undo_head_pose(angles):
    assert np.allclose(moves.head_angles(head_pose(*angles)), angles)


def test_the_offsets_are_turns_from_the_first_frame():
    data = json.loads((moves.MOVES_DIR / "inquiring3.json").read_text(encoding="utf-8"))
    first, frame = (np.asarray(data["set_target_data"][i]["head"]) for i in (0, 60))
    offset = MOVE.offsets[60]
    turn = head_pose(offset[2], offset[1], offset[0])[:3, :3]
    assert np.allclose(turn @ first[:3, :3], frame[:3, :3], atol=1e-5)   # recorded to 6 decimals
    assert np.allclose(offset[6:], (frame[:3, 3] - first[:3, 3]) * 1000)


# ── played by the gesture stream ─────────────────────────────────────────

def stream(cues, move=MOVE):
    """Runs a gesture stream with the move on a fake clock: "think" at 0, then each (time, mode) cue, also after
    the stream came to rest (a new one starts then). Returns [(time, pose)] sent."""
    state, sent, cues = {"t": 0.0}, [], sorted(cues, key=lambda cue: cue[0])
    stream = None

    def sleep(seconds):
        state["t"] += seconds
        while cues and cues[0][0] <= state["t"]:
            stream.set(cues.pop(0)[1])

    stream = Gestures(lambda pose: sent.append((state["t"], pose)), clock=lambda: state["t"], sleep=sleep,
                      move=move)
    stream.placed()
    stream.set("think")
    while True:
        assert wait_for(lambda: not stream.moving, timeout=5)
        if not cues:
            return sent
        at, mode = cues.pop(0)
        state["t"] = max(state["t"], at)
        stream.set(mode)


def test_thinking_plays_the_move_from_the_neutral_pose_then_sways():
    sent = stream([(8.0, None)])
    assert sent[0][1] == Pose() and sent[-1][1] == Pose()
    by_time = dict(sent)
    # Once faded in, the targets are the recording's own (a little behind it: it starts slowly while fading in).
    peak = max(pose.left for _, pose in sent)
    assert peak == pytest.approx(99.9, abs=0.5)
    assert max(pose.roll for _, pose in sent) == pytest.approx(MOVE.offsets[:, 0].max(), abs=0.3)
    # After the move: the sway, as without it.
    for t in (5.0, 6.0, 7.0):
        near = min(by_time, key=lambda time: abs(time - t))
        assert by_time[near] == pytest.approx(pattern("think", near), abs=1e-6)


def test_the_move_streams_at_its_own_rate_and_the_sway_as_before():
    sent = stream([(8.0, None)])
    times = np.array([t for t, _ in sent])
    gaps = np.diff(times)
    during = gaps[(times[:-1] > 0) & (times[:-1] < MOVE.duration)]   # from the first tick thinking shows
    after = gaps[(times[:-1] > MOVE.duration + 1.0) & (times[:-1] < 7.5)]
    assert np.allclose(during, 1 / MOVE_HZ) and np.allclose(after, 1 / GESTURE_HZ)


def test_every_target_stays_inside_the_limits_whatever_the_mix():
    for cut in np.arange(0.1, 4.0, 0.3):
        for _, pose in stream([(cut, "speak"), (cut + 1.5, None)]):
            for name, value in pose._asdict().items():
                assert abs(value) <= MOVE_LIMITS[name], (cut, name, value)


@pytest.mark.parametrize("then", ["speak", None])
def test_cut_at_any_moment_no_part_moves_faster_than_the_recording_does(then):
    """The reply's first audio (or the patient going on) can come at any point of the move."""
    for cut in np.arange(0.02, 4.4, 0.04):
        sent = stream([(cut, then), (cut + 2.0, None)])
        assert sent[0][1] == Pose() and sent[-1][1] == Pose()
        for (t0, a), (t1, b) in zip(sent, sent[1:]):
            step = np.abs(np.subtract(b, a))
            allowed = OWN_STEP + 1e-6
            allowed[[4, 5]] += SPEAK_STEP * (t1 - t0) * MOVE_HZ + 1e-6   # the speaking antennas' own motion
            assert np.all(step[DRIVEN] <= allowed[DRIVEN]), (cut, t0, step)
            speed = step / (t1 - t0)
            assert speed[3] <= 30 and speed[4] <= 100, (cut, t0, speed)   # body and right antenna: as before


def test_speaking_cuts_the_move_within_its_fade():
    sent = stream([(1.5, "speak"), (4.0, None)])
    after = [pose for t, pose in sent if 1.5 + MOVE_FADE_SECONDS + 0.05 <= t < 4.0]
    assert after and all(pose[:4] == (0, 0, 0, 0) and pose[6:] == (0, 0, 0) for pose in after)   # antennas only
    assert all(abs(pose.left) <= gestures.SPEAK_ANTENNA + 1e-9 for pose in after)


def test_the_move_plays_once_however_often_thinking_restarts_within_its_interval():
    # The patient goes on talking (None), pauses again (think), and so on, within MOVE_AGAIN_SECONDS.
    sent = stream([(1.0, None), (2.0, "think"), (3.0, None), (4.0, "think"), (9.0, None)])
    lefts = np.array([pose.left for _, pose in sent])
    assert lefts.max() > 60                                            # the move went on where it was
    late = [pose.left for t, pose in sent if t > 7.0]
    assert max(abs(value) for value in late) <= gestures.THINK_ANTENNA + 1e-9   # then only the sway


def test_thinking_again_later_plays_the_move_again():
    late = MOVE_AGAIN_SECONDS + 1.0
    sent = stream([(4.0, "speak"), (7.0, None), (late, "think"), (late + 4.0, None)])
    first = [pose.left for t, pose in sent if t < 4.0]
    second = [pose.left for t, pose in sent if late < t < late + 4.0]
    assert max(first) > 90 and max(second) > 90


def test_without_the_move_thinking_is_the_sway_alone():
    sent = stream([(3.0, None)], move=None)
    assert max(abs(pose.left) for _, pose in sent) <= gestures.THINK_ANTENNA + 1e-9
    assert np.allclose(np.diff([t for t, _ in sent]), 1 / GESTURE_HZ)


# ── through the robot facade ─────────────────────────────────────────────

class SoundCheckingMini(Mini):
    """Fails the stream on any call that would play a sound or a move through the SDK."""

    def __init__(self):
        super().__init__()
        self.media = types.SimpleNamespace()   # no audio methods at all

    def __getattr__(self, name):
        if name in ("play_move", "async_play_move", "cancel_move", "play_sound"):
            raise AssertionError(f"{name} called")
        raise AttributeError(name)


def test_the_robot_thinks_with_the_move_through_targets_only():
    mini = SoundCheckingMini()
    robot = awake(mini)
    assert robot._gestures._move is moves.load("inquiring3")
    robot.gesture("think")
    assert wait_for(lambda: any(np.degrees(t["antennas"][1] - NEUTRAL_ANTENNAS[1]) > 60 for t in mini.targets()),
                    timeout=5)
    robot.gesture("speak")
    robot.gesture(None)
    assert wait_for(lambda: not robot._gestures.moving, timeout=5)
    assert set(mini.names()) <= {"set_target", "goto_target", "enable_motors", "wake_up"}
    last = mini.targets()[-1]
    assert np.allclose(last["head"], np.eye(4)) and last["antennas"] == pytest.approx(list(NEUTRAL_ANTENNAS))


def test_a_missing_move_file_leaves_the_sway(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(moves, "MOVES_DIR", tmp_path)
    moves.load.cache_clear()
    try:
        robot = ReachyRobot.attach(Mini())
        assert robot._gestures._move is None and "only sways" in caplog.text
    finally:
        moves.load.cache_clear()


def test_the_move_s_head_position_reaches_the_target():
    mini = Mini()
    robot = ReachyRobot.attach(mini)
    robot._send_gesture(Pose(roll=5.0, x=1.5, y=-2.0, z=3.0))
    head = mini.targets()[-1]["head"]
    assert np.allclose(head[:3, 3], [0.0015, -0.002, 0.003])
    assert np.allclose(head[:3, :3], head_pose(roll_deg=5.0)[:3, :3])


def test_a_move_target_is_cheap_to_compute():
    """The Pi computes one at MOVE_HZ while the move plays: an interpolation of each channel per tick."""
    began = time.perf_counter()
    for i in range(500):
        gestures.thinking(MOVE, i / MOVE_HZ, i / MOVE_HZ)
    assert (time.perf_counter() - began) / 500 < 0.002
