"""MonitorStream + multimodal emotion: scored on the robot, handed to the laptop when the robot can't keep up."""

import asyncio

from medcare_reachy.bridge.multimodal import ComputeGovernor
from medcare_reachy.bridge.runner import MonitorStream
from medcare_reachy.bridge.tests.test_runner import (
    Camera, Clock, Engine, FakeEmotion, FrameApp, HeartbeatApp, StreamApp, heartbeat_runner, inline)


class FrameAppWithEmotion(FrameApp):
    """A server that takes the robot's emotion report along with a streamed frame (0.7.0 servers)."""

    async def monitor_frame(self, session_id, generation, frame_seq, timestamp, jpeg, emotion=None):
        self.frames.append((session_id, generation, frame_seq, timestamp, jpeg, emotion))
        if self.gate:
            await self.gate.wait()
        return {"session_id": session_id, "frame_seq": frame_seq, "identity_status": "verified",
                "target_box": [0.1, 0.1, 0.4, 0.4]}


class StreamCamera(Camera):
    def get_camera_frame(self):
        return self.get_frame()


def server_stream(emotion, governor, clock):
    app = FrameAppWithEmotion()
    stream = MonitorStream(app, StreamCamera(), None, clock=clock, run_blocking=inline,
                           stream_encode=lambda frame: b"jpg", emotion=emotion, governor=governor)
    return stream, app


async def run(stream, clock, steps):
    for _ in range(steps):
        clock.t += 1 / 15
        await stream.step()
        await stream.drain()


def test_streamed_frames_are_scored_on_the_robot_and_the_report_rides_on_the_next_frame():
    clock = Clock()
    emotion = FakeEmotion({"face_index": None, "probabilities": {"sad": 1.0}})
    stream, app = server_stream(emotion, ComputeGovernor("auto", clock=clock), clock)

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        await stream.step()                     # first frame: nobody verified yet
        await stream.drain()
        assert emotion.calls == []
        await run(stream, clock, 12)

    asyncio.run(scenario())
    reports = [frame[5] for frame in app.frames if frame[5] is not None]
    assert emotion.calls == [[0.1, 0.1, 0.4, 0.4]] * 2          # at most every 0.5 s, the verified face only
    assert len(reports) == 2 and reports[0]["probabilities"] == {"sad": 1.0}
    assert reports[0]["target_box"] == [0.1, 0.1, 0.4, 0.4] and "timestamp" in reports[0]
    assert stream.emotion_compute() == "robot" and stream.emotion_fps() > 0
    assert stream.emotion_status()["compute"] == "robot" and stream.emotion_status()["fallbacks"] == 0


def test_repeated_failures_hand_emotion_to_the_laptop_and_frames_keep_flowing():
    clock = Clock()

    class Broken:
        calls = 0

        def score(self, *args):
            Broken.calls += 1
            raise RuntimeError("model crashed")

    stream, app = server_stream(Broken(), ComputeGovernor("auto", max_failures=3, clock=clock), clock)

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        await run(stream, clock, 40)

    asyncio.run(scenario())
    assert Broken.calls == 3 and stream.emotion_compute() == "laptop"
    assert stream.governor.fallbacks == 1 and "failed" in stream.governor.reason
    assert len(app.frames) >= 30 and all(frame[5] is None for frame in app.frames[-5:])


def test_on_robot_landmarks_switch_to_streamed_frames_when_emotion_falls_back():
    """With landmarks on the robot the laptop sees no frames: the fallback has to start streaming them."""
    clock = Clock()

    class BothApps(StreamApp, FrameAppWithEmotion):
        def __init__(self):
            StreamApp.__init__(self)
            FrameAppWithEmotion.__init__(self)
            self.target_box = [0.1, 0.1, 0.4, 0.4]

    app = BothApps()
    slow = {"latency": 0.0}

    class SlowEmotion:
        def score(self, frame, packet, target):
            clock.t += slow["latency"]
            return {"face_index": 0, "probabilities": {"sad": 1.0}}

    governor = ComputeGovernor("auto", target_fps=2.0, max_slow=3, recover_after=5.0, clock=clock)
    stream = MonitorStream(app, StreamCamera(), Engine(), clock=clock, run_blocking=inline,
                           encode=lambda frame: b"jpg", stream_encode=lambda frame: b"jpg",
                           emotion=SlowEmotion(), governor=governor)

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        await run(stream, clock, 12)
        assert app.landmarks and not app.frames and stream.emotion_compute() == "robot"
        assert any("emotion" in packet for packet in app.landmarks)
        slow["latency"] = 0.8                   # each score now takes longer than a frame interval
        await run(stream, clock, 40)
        assert stream.emotion_compute() == "laptop" and "took over" in governor.reason
        landmarks_before, frames_before = len(app.landmarks), len(app.frames)
        await run(stream, clock, 10)
        assert len(app.frames) > frames_before and len(app.landmarks) == landmarks_before
        slow["latency"] = 0.0
        clock.t += 5.0                          # recover_after: the robot gets another go
        await run(stream, clock, 10)
        assert stream.emotion_compute() == "robot" and governor.recoveries == 1
        assert len(app.landmarks) > landmarks_before

    asyncio.run(scenario())


def test_pinned_laptop_mode_never_scores_on_the_robot():
    clock = Clock()
    emotion = FakeEmotion()
    stream, app = server_stream(emotion, ComputeGovernor("laptop", clock=clock), clock)

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        await run(stream, clock, 12)

    asyncio.run(scenario())
    assert emotion.calls == [] and stream.emotion_compute() == "laptop" and len(app.frames) >= 10


def test_without_emotion_the_stream_reports_none_and_old_servers_get_the_old_payload():
    clock = Clock()
    stream, app = server_stream(None, None, clock)
    assert stream.emotion_compute() is None and stream.emotion_status() is None

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        await run(stream, clock, 3)

    asyncio.run(scenario())
    assert app.frames and all(frame[5] is None for frame in app.frames)


def test_heartbeat_reports_where_emotion_runs():
    app = HeartbeatApp({"stop_all": False})
    runner = heartbeat_runner(app)
    runner.stream.emotion_compute = lambda: "robot"
    runner.stream.emotion_fps = lambda: 1.9
    asyncio.run(runner.heartbeat_once())
    assert app.payloads[0]["emotion_compute"] == "robot" and app.payloads[0]["emotion_fps"] == 1.9
