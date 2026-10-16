"""Reachy's body language for what a conversation carries (14 Oct 2026): the sustained posture under the gestures
(bridge/gestures.py set_affect), the facade's express() and recorded emotion moves (bridge/media.py), the motion
worker's play_emotion (bridge/reachy_hardware.py), the live chat's `express` message (bridge/live.py) and the
settings API's test of it on the real robot (service.py)."""

import threading
import time

import numpy as np
import pytest

from medcare_reachy import service as service_module
from medcare_reachy.bridge import gestures
from medcare_reachy.bridge.gestures import AFFECTS, LIMITS, MODES, Gestures, Pose, affect_pattern, mix
from medcare_reachy.bridge.reachy_hardware import HardwareError, ReachyMiniHardware
from medcare_reachy.bridge.tests.fakes import FakeRobot
from medcare_reachy.bridge.tests.test_live import Chat, inline, run, until
from medcare_reachy.bridge.tests.test_reachy_hardware import _chat_with_hardware, frame


# ── the posture patterns ──
def test_concern_sinks_and_tilts_the_head_and_droops_the_antennas_while_joy_perks_them_up():
    concern = affect_pattern("concern", 0.0)
    assert concern.pitch > 0 and concern.roll > 0 and concern.z < 0          # down, tilted, lower
    assert concern.right > 0 > concern.left                                  # the emote presets' droop
    joy = affect_pattern("joy", 10.0)                                        # after the bounce died away
    assert joy.pitch < 0 and joy.z > 0 and joy.right < 0 < joy.left          # up, perked
    assert abs(affect_pattern("joy", 0.2).right - joy.right) > 1.0           # the bounce at the start
    with pytest.raises(ValueError):
        affect_pattern("rage", 0.0)


def test_the_posture_adds_under_the_motions_and_never_goes_past_the_limits():
    assert mix(dict.fromkeys(MODES, 0.0), 1.0, affect=("concern", 0.0, 0.0)) == Pose()   # level 0: neutral
    full = mix(dict.fromkeys(MODES, 0.0), 1.0, affect=("concern", 1.0, 1.0))
    assert full.pitch > 0 and full.right > 0
    for t in np.arange(0.0, 12.0, 0.05):
        for affect in AFFECTS:
            pose = mix({"think": 0.0, "speak": 1.0}, t, affect=(affect, 1.0, t))
            for name, value in pose._asdict().items():
                assert abs(value) <= LIMITS[name], (affect, name, t)


class Script:
    """A gesture stream on a fake clock whose pauses move it on; cues set a mode or a posture at a time."""

    def __init__(self, cues):
        self.t, self.cues, self.sent = 0.0, list(cues), []
        self.stream = Gestures(self.send, clock=lambda: self.t, sleep=self.sleep)
        self.stream.placed()

    def send(self, pose):
        self.sent.append((self.t, pose))

    def sleep(self, seconds):
        self.t += seconds
        while self.cues and self.cues[0][0] <= self.t:
            _, kind, value = self.cues.pop(0)
            (self.stream.set_affect if kind == "affect" else self.stream.set)(value)


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.005)
    return True


def test_a_posture_fades_in_on_its_own_stays_under_speaking_and_fades_out_when_lifted():
    script = Script([(4.0, "mode", "speak"), (7.0, "mode", None), (9.0, "affect", None)])
    script.stream.set_affect("concern")                       # no motion asked for: the posture starts the stream
    assert wait_for(lambda: not script.stream.moving, 5.0)
    sent = script.sent
    assert sent[0][1] == Pose() and sent[-1][1] == Pose()
    settled = [pose for t, pose in sent if 2.5 <= t < 4.0]     # faded in, before speaking
    assert settled and all(pose.pitch > 4 and pose.right > 10 for pose in settled)
    speaking = [pose for t, pose in sent if 5.0 <= t < 7.0]   # speaking moves the antennas on top of the droop
    assert speaking and all(pose.pitch > 4 for pose in speaking)
    assert max(pose.right for pose in speaking) - min(pose.right for pose in speaking) > 5
    assert sent[-1][0] >= 9.0 + gestures.AFFECT_FADE_SECONDS - 0.1 and sent[-1][1] == Pose()   # lifted, then stops
    for (t0, a), (t1, b) in zip(sent, sent[1:]):              # gently: the fade is slower than a motion's
        speed = [abs(y - x) / (t1 - t0) for x, y in zip(a, b)]
        assert max(speed[:4]) <= 30 and max(speed[4:6]) <= 100, (t0, speed)


def test_a_posture_survives_a_halt_and_comes_back_once_the_robot_is_placed_again():
    sent = []
    stream = Gestures(sent.append, sleep=lambda seconds: time.sleep(0.002))
    stream.placed()
    stream.set_affect("joy")
    assert wait_for(lambda: any(pose.left > 5 for pose in sent))
    stream.halt()                                             # a goto takes over (a dance, an emote)
    count = len(sent)
    time.sleep(0.02)
    assert len(sent) == count and not stream.moving           # nothing while the robot is elsewhere
    stream.placed()
    stream.set("speak")                                       # the next motion brings the posture back
    assert wait_for(lambda: len(sent) > count and sent[-1].left > 5)
    stream.set_affect(None)
    stream.set(None)
    assert wait_for(lambda: not stream.moving)
    with pytest.raises(ValueError):
        stream.set_affect("rage")


# ── the facade and the motion worker ──
def test_the_facade_hands_the_posture_to_the_gesture_stream_and_lists_both_libraries():
    from medcare_reachy.bridge.media import MOVE_LIBRARIES, ReachyRobot, VideoFileRobot

    class Mini:
        media, media_released = None, False

    robot = ReachyRobot.attach(Mini())
    robot.express("concern")
    assert robot._gestures._affect == "concern"
    robot.express(None)
    assert robot._gestures._affect is None
    assert set(MOVE_LIBRARIES) == {"dance", "emotion"}
    with pytest.raises(ValueError):
        robot._library("moods")
    video = VideoFileRobot.__new__(VideoFileRobot)
    video.actions = []
    video.express("joy")
    assert video.play_move("emotion", "cheerful1", lambda: False) == "completed"
    assert video.actions == [("express", "joy"), ("play_move", "emotion", "cheerful1")] and video.move_names("emotion") == {}


def test_play_emotion_runs_one_recorded_emotion_move_on_the_motion_worker():
    async def scenario():
        robot = FakeRobot()
        hw = ReachyMiniHardware(robot, run_blocking=inline)
        answer = await hw.play_emotion("downcast1")
        assert answer["status"] == "started" and answer["library"] == "emotion" and answer["move"] == "downcast1"
        assert "dance" not in answer and answer["available"] == ["cheerful1", "downcast1"]
        assert await hw.wait_motion_idle(5.0)
        assert robot.calls[-2:] == ["play_move:emotion:downcast1", "hold_head"]
        with pytest.raises(HardwareError, match="unknown_emotion:rage1; available: cheerful1, downcast1"):
            await hw.play_emotion("rage1")
        dance = await hw.start_dance("jackson_square")
        assert dance["dance"] == "jackson_square" and dance["library"] == "dance"
        await hw.shutdown()
    run(scenario)


# ── the live chat's `express` message ──
def test_the_servers_express_sets_the_posture_plays_the_move_and_answers_only_when_asked():
    async def scenario():
        chat = _chat_with_hardware(frame())
        task = chat.start()
        await chat.ready()
        chat.socket.serve({"type": "express", "affect": "concern", "reason": "patient_words"})   # no request_id
        await until(lambda: "express:concern" in chat.robot.calls)
        chat.socket.serve({"type": "express", "move": "downcast1", "request_id": "x1"})
        await until(lambda: any(m.get("type") == "express_result" for m in chat.sent()))
        answer = next(m for m in chat.sent() if m["type"] == "express_result")
        assert answer["ok"] and answer["request_id"] == "x1" and answer["result"]["move"]["move"] == "downcast1"
        assert answer["result"]["affect"] == "unchanged"
        assert await chat.live.hardware.wait_motion_idle(5.0)
        assert chat.robot.affects == ["concern"]                            # the move left the posture as it was
        chat.socket.serve({"type": "express", "affect": "neutral"})
        await until(lambda: chat.robot.affects[-1:] == [None])
        chat.socket.serve({"type": "end", "reason": "finished"})
        await task
        await chat.live.hardware.shutdown()
        return chat

    chat = run(scenario)
    assert chat.robot.affects[-1] is None and chat.robot.calls.count("express:None") == 2   # lifted at the end too


def test_express_is_refused_with_the_gestures_switch_off_and_without_motion():
    async def scenario():
        chat = _chat_with_hardware(frame())
        chat.live.gestures = False
        task = chat.start()
        await chat.ready()
        chat.socket.serve({"type": "express", "affect": "joy", "request_id": "x2"})
        await until(lambda: any(m.get("type") == "express_result" for m in chat.sent()))
        assert next(m for m in chat.sent() if m["type"] == "express_result")["error"] == "gestures_off"
        assert chat.robot.affects == []
        chat.socket.serve({"type": "end", "reason": "finished"})
        await task
        await chat.live.hardware.shutdown()

        chat = Chat()   # no hardware adapter: the posture still works, a move does not
        task = chat.start()
        await chat.ready()
        chat.socket.serve({"type": "express", "affect": "joy", "move": "cheerful1", "request_id": "x3"})
        await until(lambda: any(m.get("type") == "express_result" for m in chat.sent()))
        answer = next(m for m in chat.sent() if m["type"] == "express_result")
        assert answer["ok"] is False and answer["error"] == "no_motion" and chat.robot.affects == ["joy"]
        chat.socket.serve({"type": "end", "reason": "finished"})
        await task
    run(scenario)


# ── the settings API's test on the real robot ──
def test_the_express_test_runs_the_whole_show_on_an_idle_robot_and_is_refused_while_busy(monkeypatch):
    monkeypatch.setattr(service_module.time, "sleep", lambda seconds: None)
    robot = FakeRobot()
    runner = type("Runner", (), {"robot": robot, "slot": None})()
    svc = service_module.BridgeService.__new__(service_module.BridgeService)
    svc._lock, svc._runner = threading.Lock(), runner
    answer = svc.express("concern", "downcast1", 4)
    assert answer == {"status": "started", "affect": "concern", "move": "downcast1", "seconds": 4.0}
    assert wait_for(lambda: "sleep" in robot.calls)
    assert robot.calls == ["wake", "hold_head", "express:concern", "gesture:speak", "play_move:emotion:downcast1",
                           "hold_head", "gesture:speak", "gesture:None", "express:None", "sleep"]
    runner.slot = object()
    with pytest.raises(service_module.ServiceBusy, match="idle"):
        svc.express("joy", None)
    runner.slot = None
    with pytest.raises(service_module.ServiceBusy, match="unknown affect"):
        svc.express("rage", None)
    svc._runner = None
    with pytest.raises(service_module.ServiceBusy, match="not running"):
        svc.express(None, None)
