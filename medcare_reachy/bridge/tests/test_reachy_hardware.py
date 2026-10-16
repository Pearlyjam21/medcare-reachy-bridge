"""The GPT-Live agent's hardware adapter (bridge/reachy_hardware.py) on the fake robot: the camera worker, the
serialised emotes that end in the neutral pose, the audio conversions; and the live chat's answers to the server's
camera and emote tool requests (bridge/live.py)."""

import asyncio
import base64
import json
import time

import numpy as np
import pytest

from medcare_reachy.bridge import reachy_hardware
from medcare_reachy.bridge.reachy_hardware import EMOTE_PRESETS, HardwareError, ReachyMiniHardware
from medcare_reachy.bridge.tests.fakes import FakeRobot
from medcare_reachy.bridge.tests.test_live import Chat, inline, run, until


def frame():
    return np.full((480, 640, 3), 90, dtype=np.uint8)


def hardware(robot=None):
    return ReachyMiniHardware(robot or FakeRobot(), run_blocking=inline)


def test_capture_frame_copies_the_robots_frame_and_jpeg_encodes_it_off_the_loop():
    async def scenario():
        robot = FakeRobot()
        robot.frame = frame()
        hw = hardware(robot)
        captured = await hw.capture_frame()
        assert (captured.width, captured.height) == (640, 480) and captured.image is not robot.frame
        jpeg, meta = await hw.capture_jpeg()
        assert jpeg[:2] == b"\xff\xd8" and meta.width == 640
        await hw.shutdown()
    run(scenario)


def test_a_camera_that_gives_no_frame_is_a_hardware_error(monkeypatch):
    monkeypatch.setattr(reachy_hardware, "CAPTURE_RETRY_SECONDS", 0.0)

    async def scenario():
        hw = hardware()
        with pytest.raises(HardwareError, match="no_camera_frame"):
            await hw.capture_frame()
        await hw.shutdown()
    run(scenario)


def test_an_emote_is_queued_at_once_then_runs_its_gotos_and_ends_in_the_neutral_pose():
    async def scenario():
        robot = FakeRobot()
        hw = hardware(robot)
        answer = await hw.trigger_emote("applaud_cheer")
        assert answer == {"status": "queued", "preset": "applaud_cheer", "pending_motion_jobs": 1}
        assert await hw.wait_motion_idle(5.0)
        gotos = [call for call in robot.calls if call.startswith("goto:")]
        assert gotos == ["goto:-32,32", "goto:24,-24", "goto:-32,32", "goto:24,-24", "goto:-18,18"]
        assert robot.calls[-1] == "hold_head"
        await hw.shutdown()
    run(scenario)


@pytest.mark.parametrize("preset", EMOTE_PRESETS)
def test_every_preset_moves_within_the_limits_and_recentres(preset):
    async def scenario():
        robot = FakeRobot()
        hw = hardware(robot)
        await hw.trigger_emote(preset)
        assert await hw.wait_motion_idle(5.0)
        gotos = [call for call in robot.calls if call.startswith("goto:")]
        assert gotos and robot.calls[-1] == "hold_head"
        for call in gotos:
            right, left = (float(v) for v in call[5:].split(","))
            assert abs(right) <= reachy_hardware.ANTENNA_LIMIT_DEG and abs(left) <= reachy_hardware.ANTENNA_LIMIT_DEG
        await hw.shutdown()
    run(scenario)


class SlowRobot(FakeRobot):
    """A robot whose gotos take time, as the SDK's goto_target does."""

    def goto(self, pose, antennas_deg=(0.0, 0.0), duration=0.5):
        time.sleep(0.05)
        super().goto(pose, antennas_deg, duration)


def test_an_unknown_preset_and_a_busy_motion_worker_are_refused():
    async def scenario():
        hw = hardware(SlowRobot())
        with pytest.raises(HardwareError, match="unknown_emote"):
            await hw.trigger_emote("backflip")
        for _ in range(reachy_hardware.MAX_PENDING_EMOTES):
            await hw.trigger_emote("calming_sway")
        with pytest.raises(HardwareError, match="motion_busy"):
            await hw.trigger_emote("calming_sway")
        assert await hw.wait_motion_idle(20.0)
        await hw.shutdown()
    run(scenario)


def test_microphone_and_speaker_conversions_round_trip_pcm16():
    async def scenario():
        robot = FakeRobot()
        hw = hardware(robot)
        assert await hw.read_input_pcm16() == b""
        robot.mic.append((np.column_stack([np.full(4, 0.5, np.float32), np.zeros(4, np.float32)]), 16000))
        pcm = await hw.read_input_pcm16()
        assert np.frombuffer(pcm, dtype="<i2").tolist() == [16384] * 4
        assert await hw.enqueue_output_pcm16(pcm) and robot.blocks == 1
        assert np.allclose(robot.pushed[0], 0.5)
        await hw.clear_output_audio()
        assert "clear_live_audio" in robot.calls
        await hw.shutdown()
    run(scenario)


def _chat_with_hardware(robot_frame=None):
    chat = Chat()
    chat.robot.frame = robot_frame
    chat.live.hardware = ReachyMiniHardware(chat.robot, run_blocking=inline)
    return chat


def test_the_server_camera_request_is_answered_with_a_base64_jpeg():
    async def scenario():
        chat = _chat_with_hardware(frame())
        task = chat.start()
        await chat.ready()
        chat.socket.serve({"type": "camera", "request_id": "r1"})
        await until(lambda: any(m.get("type") == "frame" for m in chat.sent()))
        answer = next(m for m in chat.sent() if m["type"] == "frame")
        assert answer["request_id"] == "r1" and (answer["width"], answer["height"]) == (640, 480)
        assert base64.b64decode(answer["jpeg"])[:2] == b"\xff\xd8"
        chat.socket.serve({"type": "end", "reason": "finished"})
        await task
        await chat.live.hardware.shutdown()
    run(scenario)


def test_a_camera_request_without_a_frame_answers_an_error(monkeypatch):
    monkeypatch.setattr(reachy_hardware, "CAPTURE_RETRY_SECONDS", 0.0)

    async def scenario():
        chat = _chat_with_hardware(None)
        task = chat.start()
        await chat.ready()
        chat.socket.serve({"type": "camera", "request_id": "r2"})
        await until(lambda: any(m.get("type") == "frame" for m in chat.sent()))
        answer = next(m for m in chat.sent() if m["type"] == "frame")
        assert answer == {"type": "frame", "request_id": "r2", "error": "no_camera_frame"}
        chat.socket.serve({"type": "end", "reason": "finished"})
        await task
        await chat.live.hardware.shutdown()
    run(scenario)


def test_the_server_emote_request_queues_the_gesture_and_answers_ok():
    async def scenario():
        chat = _chat_with_hardware(frame())
        task = chat.start()
        await chat.ready()
        chat.socket.serve({"type": "emote", "preset": "sympathy_droop_warm", "request_id": "e1"})
        await until(lambda: any(m.get("type") == "emote_result" for m in chat.sent()))
        answer = next(m for m in chat.sent() if m["type"] == "emote_result")
        assert answer["ok"] and answer["result"]["preset"] == "sympathy_droop_warm"
        assert await chat.live.hardware.wait_motion_idle(5.0)
        assert "hold_head" in chat.robot.calls
        chat.socket.serve({"type": "end", "reason": "finished"})
        await task
        await chat.live.hardware.shutdown()
    run(scenario)


def test_emote_requests_are_refused_without_hardware_or_with_gestures_off():
    async def scenario():
        chat = Chat()   # no hardware adapter
        task = chat.start()
        await chat.ready()
        chat.socket.serve({"type": "emote", "preset": "curious_tilt", "request_id": "e2"})
        await until(lambda: any(m.get("type") == "emote_result" for m in chat.sent()))
        assert next(m for m in chat.sent() if m["type"] == "emote_result") == {
            "type": "emote_result", "request_id": "e2", "ok": False, "error": "no_motion"}
        chat.socket.serve({"type": "end", "reason": "finished"})
        await task

        chat = _chat_with_hardware(frame())
        chat.live.gestures = False
        task = chat.start()
        await chat.ready()
        chat.socket.serve({"type": "emote", "preset": "curious_tilt", "request_id": "e3"})
        await until(lambda: any(m.get("type") == "emote_result" for m in chat.sent()))
        assert next(m for m in chat.sent() if m["type"] == "emote_result")["error"] == "gestures_off"
        chat.socket.serve({"type": "end", "reason": "finished"})
        await task
        await chat.live.hardware.shutdown()
    run(scenario)


class Mood:
    """A multimodal emotion engine stand-in (multimodal/engine.py report())."""

    def __init__(self, error=None):
        self.error, self.calls = error, 0

    def report(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return {"available": True, "fused": {"top": "happy", "confidence": 0.8, "used": {"face": 1.0}},
                "fer": {"status": "scored"}, "ser": {"status": "silence"}, "ter": {"status": "no_transcript"},
                "branches": {"face": True, "speech": True, "text": False}, "active": False}


def test_the_server_emotion_request_is_answered_with_the_engines_report():
    async def scenario():
        chat = Chat()
        chat.live.mood = Mood()
        task = chat.start()
        await chat.ready()
        chat.socket.serve({"type": "emotion", "request_id": "m1"})
        await until(lambda: any(m.get("type") == "emotion_result" for m in chat.sent()))
        answer = next(m for m in chat.sent() if m["type"] == "emotion_result")
        assert answer["request_id"] == "m1" and answer["available"] is True
        assert answer["fused"]["top"] == "happy" and answer["ser"] == {"status": "silence"}
        assert chat.live.mood.calls == 1
        chat.socket.serve({"type": "end", "reason": "finished"})
        await task
    run(scenario)


def test_an_emotion_request_without_an_engine_or_with_a_failing_one_answers_unavailable():
    async def scenario():
        chat = Chat()   # no engine
        task = chat.start()
        await chat.ready()
        chat.socket.serve({"type": "emotion", "request_id": "m2"})
        await until(lambda: any(m.get("type") == "emotion_result" for m in chat.sent()))
        assert next(m for m in chat.sent() if m["type"] == "emotion_result") == {
            "type": "emotion_result", "request_id": "m2", "available": False, "error": "no_emotion_engine"}
        chat.socket.serve({"type": "end", "reason": "finished"})
        await task

        chat = Chat()
        chat.live.mood = Mood(error=RuntimeError("model gone"))
        task = chat.start()
        await chat.ready()
        chat.socket.serve({"type": "emotion", "request_id": "m3"})
        await until(lambda: any(m.get("type") == "emotion_result" for m in chat.sent()))
        answer = next(m for m in chat.sent() if m["type"] == "emotion_result")
        assert answer["available"] is False and answer["error"] == "model gone"
        chat.socket.serve({"type": "end", "reason": "finished"})
        await task
    run(scenario)


def test_the_message_shapes_are_json_serialisable():
    message = {"type": "frame", "request_id": "x", "jpeg": base64.b64encode(b"\xff\xd8").decode("ascii"),
               "width": 640, "height": 480, "captured_at": 1.5}
    assert json.loads(json.dumps(message)) == message


# ── dances (the SDK's recorded moves) ──
def test_a_dance_starts_on_the_motion_worker_and_answers_at_once_with_the_real_names():
    async def scenario():
        robot = FakeRobot()
        hw = hardware(robot)
        assert await hw.dances() == robot.dances
        answer = await hw.start_dance("jackson_square", repeat=2)
        assert answer["status"] == "started" and answer["dance"] == "jackson_square" and answer["repeat"] == 2
        assert answer["available"] == ["jackson_square", "side_to_side_sway"] and answer["replaced"] is False
        assert await hw.wait_motion_idle(5.0)
        assert robot.danced == ["jackson_square", "jackson_square"] and robot.calls[-1] == "hold_head"
        assert await hw.stop_dance() == {"status": "idle"}
        await hw.shutdown()
    run(scenario)


def test_without_a_name_a_dance_is_picked_from_the_library_and_the_next_one_differs():
    async def scenario():
        robot = FakeRobot()
        hw = hardware(robot)
        first = (await hw.start_dance(None))["dance"]
        assert await hw.wait_motion_idle(5.0)
        second = (await hw.start_dance(None, repeat=9))["dance"]
        assert await hw.wait_motion_idle(5.0)
        assert {first, second} == set(robot.dances) and first != second
        assert robot.danced.count(second) == reachy_hardware.MAX_DANCE_REPEAT
        await hw.shutdown()
    run(scenario)


def test_an_unknown_dance_and_a_missing_library_are_refused():
    async def scenario():
        robot = FakeRobot()
        hw = hardware(robot)
        with pytest.raises(HardwareError, match="unknown_dance:moonwalk; available: jackson_square, side_to_side_sway"):
            await hw.start_dance("moonwalk")
        robot.dances = {}
        with pytest.raises(HardwareError, match="no_dance_library"):
            await hw.start_dance(None)
        assert robot.danced == []
        await hw.shutdown()
    run(scenario)


def test_stop_dance_ends_a_running_dance_and_a_new_dance_replaces_one():
    async def scenario():
        robot = FakeRobot()
        robot.dance_seconds = 5.0
        hw = hardware(robot)
        await hw.start_dance("side_to_side_sway", repeat=3)
        await asyncio.sleep(0.05)
        assert await hw.stop_dance() == {"status": "cancelled", "dance": "side_to_side_sway"}
        assert await hw.wait_motion_idle(2.0)   # it stopped within a step, not after 15 s
        assert robot.danced == ["side_to_side_sway"] and robot.calls[-1] == "hold_head"
        await hw.start_dance("side_to_side_sway")
        await asyncio.sleep(0.05)
        answer = await hw.start_dance("jackson_square")
        assert answer["replaced"] is True
        await until(lambda: robot.danced[-1:] == ["jackson_square"])   # the first stopped, the second started
        assert await hw.stop_dance() == {"status": "cancelled", "dance": "jackson_square"}
        assert await hw.wait_motion_idle(2.0)
        await hw.start_dance("jackson_square")
        await asyncio.sleep(0.05)
        await hw.shutdown()                      # the chat's end stops a dance too
        assert robot.calls[-1] == "hold_head"
    run(scenario)


def test_the_server_dance_requests_are_answered_and_refused_like_emotes():
    async def scenario():
        chat = _chat_with_hardware(frame())
        task = chat.start()
        await chat.ready()
        chat.socket.serve({"type": "dance", "dance": "jackson_square", "repeat": 1, "request_id": "d1"})
        await until(lambda: any(m.get("type") == "dance_result" for m in chat.sent()))
        answer = next(m for m in chat.sent() if m["type"] == "dance_result")
        assert answer["ok"] and answer["request_id"] == "d1" and answer["result"]["dance"] == "jackson_square"
        assert await chat.live.hardware.wait_motion_idle(5.0)
        chat.socket.serve({"type": "stop_dance", "request_id": "d2"})
        await until(lambda: sum(m.get("type") == "dance_result" for m in chat.sent()) == 2)
        assert [m for m in chat.sent() if m["type"] == "dance_result"][1] == {
            "type": "dance_result", "request_id": "d2", "ok": True, "result": {"status": "idle"}}
        chat.socket.serve({"type": "dance", "dance": "moonwalk", "request_id": "d3"})
        await until(lambda: sum(m.get("type") == "dance_result" for m in chat.sent()) == 3)
        assert [m for m in chat.sent() if m["type"] == "dance_result"][2]["error"].startswith("unknown_dance:moonwalk")
        chat.socket.serve({"type": "end", "reason": "finished"})
        await task
        await chat.live.hardware.shutdown()

        chat = Chat()   # no hardware adapter
        task = chat.start()
        await chat.ready()
        chat.socket.serve({"type": "dance", "request_id": "d4"})
        await until(lambda: any(m.get("type") == "dance_result" for m in chat.sent()))
        assert next(m for m in chat.sent() if m["type"] == "dance_result") == {
            "type": "dance_result", "request_id": "d4", "ok": False, "error": "no_motion"}
        chat.socket.serve({"type": "end", "reason": "finished"})
        await task
    run(scenario)
