"""Check-in gestures: the motions' bounds, the streaming thread, and how the robot facade drives the SDK."""

import threading
import time
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from medcare_reachy.bridge import gestures
from medcare_reachy.bridge.gestures import LIMITS, MODES, MOVE_LIMITS, Gestures, Pose, mix, pattern
from medcare_reachy.bridge.media import NEUTRAL_ANTENNAS, ReachyRobot, head_pose


def wait_for(predicate, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return False


# ── the motions ──────────────────────────────────────────────────────────

def test_at_rest_the_mix_is_exactly_the_neutral_pose():
    assert mix(dict.fromkeys(MODES, 0.0), 3.7) == Pose()


def test_thinking_tilts_the_head_and_turns_the_whole_robot_while_speaking_moves_only_the_antennas():
    think = pattern("think", 1.0)
    assert think.roll > 0 and think.pitch < 0 and think.yaw == think.body_yaw != 0
    assert think.right == -think.left != 0                                    # mirrored, like breathing
    assert pattern("speak", 1.0)[:4] == (0, 0, 0, 0) and pattern("speak", 1.0).right != 0
    with pytest.raises(ValueError):
        pattern("dance", 0.0)


def test_no_mix_ever_goes_past_the_limits():
    for t in np.arange(0.0, 20.0, 0.01):
        for levels in ({"think": 1.0, "speak": 0.0}, {"think": 0.0, "speak": 1.0}, {"think": 1.0, "speak": 1.0}):
            for name, value in mix(levels, t)._asdict().items():
                assert abs(value) <= LIMITS[name], (name, t, levels)


class Script:
    """Runs a gesture stream on a fake clock: every pause between targets moves it on, and modes change on cue."""

    def __init__(self, cues):
        self.t, self.cues, self.sent = 0.0, list(cues), []
        self.stream = Gestures(self.send, clock=lambda: self.t, sleep=self.sleep)
        self.stream.placed()

    def send(self, pose):
        self.sent.append((self.t, pose))

    def sleep(self, seconds):
        self.t += seconds
        while self.cues and self.cues[0][0] <= self.t:
            self.stream.set(self.cues.pop(0)[1])


def test_thinking_flows_into_speaking_and_back_to_neutral_gently():
    script = Script([(3.0, "speak"), (5.0, None)])
    script.stream.set("think")
    assert wait_for(lambda: not script.stream.moving)
    sent = script.sent
    assert sent[0][1] == Pose() and sent[-1][1] == Pose()                     # from and back to the neutral pose
    speaking = [pose for t, pose in sent if 4.0 <= t < 5.0]
    assert speaking and all(pose[:4] == (0, 0, 0, 0) for pose in speaking)    # the thinking tilt has faded out
    for (t0, a), (t1, b) in zip(sent, sent[1:]):
        speed = [abs(y - x) / (t1 - t0) for x, y in zip(a, b)]
        assert max(speed[:4]) <= 30 and max(speed[4:]) <= 100, (t0, speed)    # degrees per second
    assert 5.0 < sent[-1][0] <= 5.0 + gestures.FADE_SECONDS + 0.1


def test_no_motion_until_the_robot_is_placed_in_the_neutral_pose():
    sent = []
    stream = Gestures(sent.append, sleep=lambda seconds: time.sleep(0.005))
    stream.set("think")
    time.sleep(0.05)
    assert sent == [] and not stream.moving
    stream.placed()
    stream.set("think")
    assert wait_for(lambda: sent) and sent[0] == Pose()
    stream.halt()


def test_halt_stops_the_stream_at_once_and_the_next_motion_waits_for_the_neutral_pose():
    sent = []
    stream = Gestures(sent.append, sleep=lambda seconds: time.sleep(0.005))
    stream.placed()
    stream.set("think")
    assert wait_for(lambda: len(sent) >= 20)
    stream.halt()
    count = len(sent)
    assert not stream.moving
    stream.set("speak")                    # asked for while a goto moves the robot (or after it went elsewhere)
    time.sleep(0.05)
    assert len(sent) == count and not stream.moving
    stream.placed()                        # the goto has put it back in the neutral pose
    stream.set("speak")
    assert wait_for(lambda: len(sent) > count)
    assert sent[count] == Pose()
    stream.halt()


def test_a_failing_robot_ends_the_stream_with_a_warning_and_no_restart_from_where_it_stopped(caplog):
    sent = []

    def send(pose):
        sent.append(pose)
        if len(sent) == 10:
            raise ConnectionError("Lost connection with the server.")

    stream = Gestures(send, sleep=lambda seconds: time.sleep(0.005))
    stream.placed()
    stream.set("think")
    assert wait_for(lambda: not stream.moving)
    assert "gestures stopped" in caplog.text
    stream.set("speak")                    # the robot is somewhere in the thinking motion: no jump from there
    time.sleep(0.05)
    assert len(sent) == 10 and not stream.moving


def test_no_motion_while_the_motors_are_off():
    sent = []
    stream = Gestures(sent.append, ready=lambda: False)
    stream.placed()
    stream.set("think")
    assert wait_for(lambda: not stream.moving)
    assert sent == []
    with pytest.raises(ValueError):
        stream.set("dance")


def test_no_motion_from_a_pose_measured_away_from_neutral(caplog):
    sent, there = [], [False]
    stream = Gestures(sent.append, start_from=lambda: Pose() if there[0] else None,
                      sleep=lambda seconds: time.sleep(0.005))
    stream.placed()
    stream.set("think")
    assert wait_for(lambda: not stream.moving)
    assert sent == [] and "not there now" in caplog.text
    there[0] = True
    stream.set("think")                    # still not until a goto has put it there
    time.sleep(0.05)
    assert sent == []
    stream.placed()
    stream.set("think")
    assert wait_for(lambda: sent)
    stream.halt()


# ── the robot facade ─────────────────────────────────────────────────────

class Mini:
    """The SDK calls ReachyRobot makes, in order, and the pose and motor state the daemon reports."""

    def __init__(self, motors="enabled", sleep_seconds=0.0):
        self.log = []
        self.motors, self.sleep_seconds = motors, sleep_seconds
        self.head, self.joints = np.eye(4), ([0.0] * 7, list(NEUTRAL_ANTENNAS))   # measured: the neutral pose
        self.client = types.SimpleNamespace(get_status=lambda wait=True: types.SimpleNamespace(
            backend_status=types.SimpleNamespace(motor_control_mode=types.SimpleNamespace(value=self.motors))))

    def set_target(self, **kwargs):
        self.log.append(("set_target", kwargs))

    def goto_target(self, **kwargs):
        self.log.append(("goto_target", kwargs))

    def enable_motors(self):
        self.log.append(("enable_motors", {}))

    def wake_up(self):
        self.log.append(("wake_up", {}))

    def goto_sleep(self):
        self.log.append(("goto_sleep", {}))
        time.sleep(self.sleep_seconds)

    def get_current_head_pose(self):
        return self.head

    def get_current_joint_positions(self):
        return self.joints

    def targets(self):
        return [kwargs for name, kwargs in self.log if name == "set_target"]

    def names(self):
        return [name for name, _ in self.log]


def awake(mini):
    """A robot woken up and holding its head in the neutral pose, as at the start of a check-in."""
    robot = ReachyRobot.attach(mini)
    robot.wake()
    robot.hold_head()
    return robot


def test_gestures_only_after_hold_head_and_never_after_sleep():
    mini = Mini()
    robot = ReachyRobot.attach(mini)
    robot.gesture("think")
    robot.wake()                                               # wake_up's own moves end in an unknown pose
    robot.gesture("think")
    time.sleep(0.1)
    assert mini.targets() == []
    robot.hold_head()
    robot.gesture("think")
    assert wait_for(lambda: len(mini.targets()) >= 3)
    robot.sleep()
    names = mini.names()
    assert "set_target" not in names[names.index("goto_sleep"):]
    robot.gesture("think")
    time.sleep(0.1)
    assert mini.names() == names


def test_a_gesture_asked_for_while_the_robot_goes_to_sleep_never_starts():
    """The speaker's thread can outlive a cancelled tick and ask for "speak" while another thread puts the robot to
    sleep: nothing may pull it out of the sleep pose."""
    mini = Mini(sleep_seconds=0.2)
    robot = awake(mini)
    robot.gesture("think")
    assert wait_for(lambda: mini.targets())
    asking = threading.Event()

    def ask():
        while not asking.is_set():
            robot.gesture("speak")
            robot.gesture(None)
            time.sleep(0.001)

    thread = threading.Thread(target=ask)
    thread.start()
    try:
        robot.sleep()
        time.sleep(0.1)                                        # and well after it
    finally:
        asking.set()
        thread.join()
    names = mini.names()
    assert "set_target" not in names[names.index("goto_sleep"):] and not robot._gestures.moving


def test_a_goto_stops_the_stream_first_and_brings_the_antennas_and_body_back():
    mini = Mini()
    robot = awake(mini)
    robot.gesture("think")
    assert wait_for(lambda: len(mini.targets()) >= 10)
    robot.hold_head()
    time.sleep(0.1)
    names = mini.names()
    goto = len(names) - 1 - names[::-1].index("goto_target")
    assert goto == len(names) - 1                               # no target after the goto began
    hold = mini.log[goto][1]
    assert np.allclose(hold["head"], np.eye(4)) and hold["antennas"] == list(NEUTRAL_ANTENNAS)
    assert hold["body_yaw"] == 0.0


def test_after_the_motors_were_off_the_robot_never_jumps_back_to_neutral(monkeypatch):
    """Switched off mid-gesture, the head falls; switched on, it is held where it fell. The next gesture must not
    send it straight back to the neutral pose (the servos would get there at full speed)."""
    monkeypatch.setattr(gestures, "READY_CHECK_SECONDS", 0.02)
    mini = Mini()
    robot = awake(mini)
    robot.gesture("think")
    assert wait_for(lambda: len(mini.targets()) >= 10)
    mini.motors = "disabled"
    assert wait_for(lambda: not robot._gestures.moving)
    count = len(mini.targets())
    mini.motors = "enabled"
    robot.gesture("speak")                                      # the reply's first sound
    robot.gesture(None)
    robot.gesture("think")                                      # the patient's next pause
    time.sleep(0.1)
    assert len(mini.targets()) == count
    robot.hold_head()                                           # the next check-in: a goto from where it is
    robot.gesture("think")
    assert wait_for(lambda: len(mini.targets()) > count)


def test_no_gesture_from_a_pose_measured_away_from_neutral():
    """E.g. the motors were switched off and on while the robot was still, listening: the head fell meanwhile."""
    mini = Mini()
    robot = awake(mini)
    mini.head = head_pose(pitch_deg=24.0)
    mini.head[2, 3] = -0.044                                   # the sleep pose's head: 24° down, 44 mm lower
    robot.gesture("think")
    time.sleep(0.1)
    assert mini.targets() == [] and not robot._gestures.moving


@pytest.mark.parametrize("measured,start", [
    # Idle near the neutral pose on 3 Oct 2026 (daemon state API): head roll 2.0°, pitch 1.9°, yaw -2.2°, x -4 mm.
    # Started from the neutral pose itself.
    ((head_pose(-2.2, 1.9, 2.0), (-0.004, 0.001, -0.0016), 0.003, (-0.1841, 0.1825)), "neutral"),
    # A little further: started from the pose measured, easing back.
    ((head_pose(-3.0, 7.5, 3.5), (0.002, 0.0, -0.003), 0.0, NEUTRAL_ANTENNAS), "measured"),   # 8.6°: 3 Oct 14:53
    ((head_pose(0, 10.0, 0), (0, 0, 0), 0.0, NEUTRAL_ANTENNAS), "measured"),                  # head tilted 10°
    ((np.eye(4), (0, 0, -0.015), 0.0, NEUTRAL_ANTENNAS), "measured"),                         # head sunk 15 mm
    ((np.eye(4), (0, 0, 0), np.radians(5), NEUTRAL_ANTENNAS), "measured"),                    # body turned 5°
    ((np.eye(4), (0, 0, 0), 0.0, (NEUTRAL_ANTENNAS[0] - np.radians(10), NEUTRAL_ANTENNAS[1])), "measured"),
    # Further still: no gesture.
    ((head_pose(0, 20.0, 0), (0, 0, 0), 0.0, NEUTRAL_ANTENNAS), None),                        # held, or fallen
    ((np.eye(4), (0, 0, -0.025), 0.0, NEUTRAL_ANTENNAS), None),                               # sunk 25 mm
    ((np.eye(4), (0, 0, 0), np.radians(12), NEUTRAL_ANTENNAS), None),                         # body turned 12°
    ((np.eye(4), (0, 0, 0), 0.0, (NEUTRAL_ANTENNAS[0], NEUTRAL_ANTENNAS[1] + np.radians(45))), None),
])
def test_where_a_gesture_starts_from(measured, start, caplog):
    caplog.set_level("INFO")
    head, shift, body, antennas = measured
    mini = Mini()
    mini.head = np.array(head)
    mini.head[:3, 3] = shift
    mini.joints = ([body] + [0.0] * 6, list(antennas))
    robot = ReachyRobot.attach(mini)
    pose = robot._start_pose()
    assert "gesture from the pose measured" in caplog.text            # what the journal is searched for
    if start == "neutral":
        assert pose == Pose()
    elif start is None:
        assert pose is None
    else:                                                             # the first target is the pose measured
        robot._send_gesture(pose)
        target = mini.targets()[-1]
        assert np.allclose(target["head"], mini.head, atol=1e-9)
        assert target["body_yaw"] == pytest.approx(body)
        assert target["antennas"] == pytest.approx(list(antennas))


@pytest.mark.parametrize("motors", ["disabled", "gravity_compensation", None])
def test_no_easing_back_unless_the_motors_are_known_to_be_on(motors):
    mini = Mini(motors=motors)
    if motors is None:   # no status from the daemon yet
        mini.client = types.SimpleNamespace(get_status=lambda wait=True: None)
    mini.head = head_pose(0, 10.0, 0)
    assert ReachyRobot.attach(mini)._start_pose() is None
    mini.head = np.eye(4)                                             # at the neutral pose: as before
    assert ReachyRobot.attach(mini)._start_pose() == Pose()


def test_the_stream_sends_small_offsets_from_the_neutral_pose():
    mini = Mini()
    robot = awake(mini)
    robot.gesture("think")                                      # with the recorded move
    assert wait_for(lambda: len(mini.targets()) >= 30)
    robot.gesture(None)
    assert wait_for(lambda: not robot._gestures.moving)
    first, last = mini.targets()[0], mini.targets()[-1]
    for target in (first, last):                                # exactly neutral at both ends
        assert np.allclose(target["head"], np.eye(4)) and target["body_yaw"] == 0.0
        assert target["antennas"] == pytest.approx(list(NEUTRAL_ANTENNAS))
    for target in mini.targets():
        assert abs(np.degrees(target["body_yaw"])) <= MOVE_LIMITS["body_yaw"]
        right, left = np.degrees(np.array(target["antennas"]) - NEUTRAL_ANTENNAS)
        assert abs(right) <= MOVE_LIMITS["right"] and abs(left) <= MOVE_LIMITS["left"]
        assert target["head"].shape == (4, 4)
        assert np.linalg.norm(target["head"][:3, 3]) * 1000 <= np.linalg.norm(
            [MOVE_LIMITS["x"], MOVE_LIMITS["y"], MOVE_LIMITS["z"]])


def test_no_gestures_when_the_daemon_reports_the_motors_off():
    mini = Mini(motors="disabled")
    robot = awake(mini)
    robot.gesture("think")
    time.sleep(0.1)
    assert mini.targets() == [] and not robot._gestures.moving


def test_closing_the_bridge_stops_a_gesture_for_good():
    mini = Mini()
    robot = awake(mini)
    robot.gesture("speak")
    assert wait_for(lambda: mini.targets())
    robot.close()
    assert not robot._gestures.moving
    count = len(mini.targets())
    robot.gesture("speak")                                      # a late call from the closed bridge's threads
    time.sleep(0.05)
    assert len(mini.targets()) == count


def test_head_pose_roll_tilts_about_the_x_axis():
    pose = head_pose(roll_deg=90)
    assert np.allclose(pose[:3, :3] @ [0, 1, 0], [0, 0, 1])
    assert np.allclose(head_pose(30, -5, 0), head_pose(30, -5))


# ── a clip played without waiting (the 「嗯」) ────────────────────────────

def write_clip(path: Path, seconds: float, rate=16000):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.zeros(int(seconds * rate), "<i2").tobytes())


def test_a_clip_handed_over_without_waiting_still_plays_before_the_next_one(tmp_path):
    started = []
    media = types.SimpleNamespace(get_output_audio_samplerate=lambda: 16000,
                                  start_playing=lambda: started.append(time.monotonic()),
                                  push_audio_sample=lambda samples: None)
    mini = Mini()
    mini.media = media
    robot = ReachyRobot.attach(mini)
    write_clip(tmp_path / "ack.wav", 0.2)
    write_clip(tmp_path / "reply.wav", 0.1)
    began = time.monotonic()
    assert robot.play_clip(tmp_path / "ack.wav", wait=False) is True
    assert time.monotonic() - began < 0.1                      # returns as soon as it is handed over
    robot.play_clip(tmp_path / "reply.wav")
    assert started[1] - started[0] >= 0.19                     # start_playing() restarts the stream's timing
    assert time.monotonic() - began >= 0.29


def test_the_player_hands_over_a_clip_and_says_how_long_it_is(tmp_path):
    from medcare_reachy.bridge.clips import ClipPlayer

    class Robot:
        def __init__(self):
            self.played = []

        def play_clip(self, path, wait=True):
            self.played.append((Path(path).name, wait))
            return True

    (tmp_path / "zh-TW").mkdir()
    write_clip(tmp_path / "zh-TW" / "ack.wav", 0.4, rate=22050)
    robot = Robot()
    player = ClipPlayer(robot, tmp_path, "zh-TW")
    assert player.start("ack") == pytest.approx(0.4, abs=0.001)
    assert robot.played == [("ack.wav", False)]
    assert player.start("thanks") is None and "thanks" in player.missing
    with pytest.raises(KeyError):
        player.start("not_a_clip")
