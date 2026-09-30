import asyncio

import numpy as np

from reachy_bridge import __version__
from reachy_bridge.app_client import AppUnreachable, NotAuthorised, SessionLost
from reachy_bridge.runner import MonitorStream, Runner
from reachy_bridge.tests.fakes import FakeClips, FakeRobot, FakeStream, dose, make_task


class Clock:
    def __init__(self):
        self.t = 50.0

    def __call__(self):
        return self.t


async def inline(fn, *args):
    return fn(*args)


class Camera:
    def __init__(self):
        self.grabs = 0
        self.frame = np.zeros((480, 640, 3), np.uint8)

    def get_frame(self):
        self.grabs += 1
        return self.frame


class Engine:
    def process(self, frame, seq, timestamp_ms):
        return {"frame_seq": seq, "timestamp": timestamp_ms / 1000, "width": 640, "height": 480,
                "faces": [], "hands": [], "poses": []}


class StreamApp:
    def __init__(self):
        self.landmarks, self.vision = [], []
        self.gate = None
        self.fail_landmarks = None
        self.unreachable_since = None

    async def monitor_landmarks(self, packet):
        self.landmarks.append(packet)
        if self.gate:
            await self.gate.wait()
        if self.fail_landmarks:
            raise self.fail_landmarks
        return {"session_id": packet["session_id"], "frame_seq": packet["frame_seq"], "identity_status": "verified"}

    async def monitor_vision(self, session_id, generation, frame_seq, jpeg):
        self.vision.append((session_id, generation, frame_seq, jpeg))
        return {"session_id": session_id, "frame_seq": frame_seq, "identity_status": "verified"}


def make_stream():
    clock, app, camera = Clock(), StreamApp(), Camera()
    stream = MonitorStream(app, camera, Engine(), clock=clock, run_blocking=inline, encode=lambda frame: b"jpg")
    return stream, app, camera, clock


def test_no_session_means_no_capture():
    stream, app, camera, _ = make_stream()

    async def scenario():
        await stream.step()

    asyncio.run(scenario())
    assert camera.grabs == 0 and app.landmarks == []


def test_landmarks_carry_session_ids_monotonic_seq_and_capture_timestamps():
    stream, app, camera, clock = make_stream()

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        for _ in range(3):
            await stream.step()
            await stream.drain()
            clock.t += 1 / 15

    asyncio.run(scenario())
    assert [p["frame_seq"] for p in app.landmarks] == [1, 2, 3]
    assert all(p["session_id"] == "s1" and p["generation"] == "g1" for p in app.landmarks)
    assert app.landmarks[0]["timestamp"] == 50.0
    assert abs(app.landmarks[1]["timestamp"] - (50 + 1 / 15)) < 1e-9
    assert list(app.landmarks[0])[:3] == ["session_id", "generation", "frame_seq"]
    assert stream.latest["frame_seq"] == 3


def test_one_landmark_request_in_flight_and_stale_frames_dropped():
    stream, app, camera, clock = make_stream()

    async def scenario():
        app.gate = asyncio.Event()
        stream.attach({"session_id": "s1", "generation": "g1"})
        await stream.step()
        await asyncio.sleep(0)
        await stream.step()          # previous request still in flight -> frame not even grabbed
        await stream.step()
        assert camera.grabs == 1 and len(app.landmarks) == 1
        app.gate.set()
        await stream.drain()
        await stream.step()
        await stream.drain()

    asyncio.run(scenario())
    assert [p["frame_seq"] for p in app.landmarks] == [1, 2]


def test_vision_at_most_5_fps_and_after_its_landmark_packet():
    stream, app, camera, clock = make_stream()

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        for _ in range(8):
            await stream.step()
            await stream.drain()
            clock.t += 0.07

    asyncio.run(scenario())
    assert [seq for _, _, seq, _ in app.vision] == [1, 4, 7]
    assert app.vision[0] == ("s1", "g1", 1, b"jpg")


def test_attach_resets_seq_and_old_session_results_are_ignored():
    stream, app, camera, clock = make_stream()

    async def scenario():
        app.gate = asyncio.Event()
        stream.attach({"session_id": "s1", "generation": "g1"})
        await stream.step()
        await asyncio.sleep(0)
        stream.detach()
        stream.attach({"session_id": "s2", "generation": "g2", "identity_status": "searching"})
        app.fail_landmarks = SessionLost(status=409)
        app.gate.set()
        await stream.drain()
        assert stream.take_error() is None                     # error belonged to s1
        assert stream.latest["identity_status"] == "searching"  # s1's response was not accepted
        app.fail_landmarks = None
        app.gate = None
        await stream.step()
        await stream.drain()

    asyncio.run(scenario())
    assert app.landmarks[-1]["session_id"] == "s2" and app.landmarks[-1]["frame_seq"] == 1


def test_errors_of_the_attached_session_are_reported_once():
    stream, app, camera, clock = make_stream()

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        app.fail_landmarks = AppUnreachable("down")
        await stream.step()
        await stream.drain()

    asyncio.run(scenario())
    assert isinstance(stream.take_error(), AppUnreachable)
    assert stream.take_error() is None


def test_accept_keeps_newest_frame_and_does_not_hide_a_record():
    stream, *_ = make_stream()
    stream.attach({"session_id": "s1", "generation": "g1"})
    session = stream.session
    stream._accept(session, {"frame_seq": 5, "recorded": {"status": "taken"}})
    stream._accept(session, {"frame_seq": 5, "recorded": None})
    stream._accept(session, {"frame_seq": 4, "recorded": None})
    assert stream.latest["recorded"] == {"status": "taken"}
    stream._accept(session, {"frame_seq": 6, "recorded": None, "extra_events": []})
    assert stream.latest["frame_seq"] == 6


def test_fps_counters_use_a_10_second_window():
    stream, app, camera, clock = make_stream()

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        for _ in range(150):
            await stream.step()
            await stream.drain()
            clock.t += 1 / 15

    asyncio.run(scenario())
    assert 14.5 <= stream.landmark_fps() <= 15.0
    assert 4.5 <= stream.vision_fps() <= 5.0
    clock.t += 11
    assert stream.landmark_fps() == 0


class HeartbeatApp:
    def __init__(self, response=None, error=None):
        self.payloads, self.response, self.error = [], response, error
        self.unreachable_since = None

    async def heartbeat(self, payload):
        self.payloads.append(payload)
        if self.error:
            raise self.error
        return self.response


class StubStream:
    def landmark_fps(self):
        return 14.8

    def vision_fps(self):
        return 4.9


class StubSlot:
    def __init__(self):
        self.stopped = None

    def stop(self, reason, abort=True):
        self.stopped = (reason, abort)


def heartbeat_runner(app):
    clips = FakeClips()
    clips.missing_count = 2
    return Runner(app=app, robot=FakeRobot(), clips=clips, stream=StubStream(), run_blocking=inline)


def test_heartbeat_payload_and_stop_all():
    app = HeartbeatApp({"stop_all": True, "server_time": "t"})
    runner = heartbeat_runner(app)
    runner.slot = StubSlot()
    asyncio.run(runner.heartbeat_once())
    assert app.payloads == [{"robot_reachable": True, "landmark_fps": 14.8, "vision_fps": 4.9,
                             "bridge_version": __version__, "missing_clips": 2}]
    assert runner.slot.stopped == ("stop_all", True)


def test_heartbeat_not_authorised_stops_and_unreachable_does_not():
    runner = heartbeat_runner(HeartbeatApp(error=NotAuthorised(status=403)))
    runner.slot = StubSlot()
    asyncio.run(runner.heartbeat_once())
    assert runner.slot.stopped == ("stop_all", True)
    runner = heartbeat_runner(HeartbeatApp(error=AppUnreachable("x")))
    runner.slot = StubSlot()
    asyncio.run(runner.heartbeat_once())
    assert runner.slot.stopped is None


class TaskApp:
    """tasks/current returns a held task once (restart recovery), then tasks/next stops the runner."""

    def __init__(self, runner_ref):
        self.runner_ref = runner_ref
        self.calls = []
        self.unreachable_since = None

    async def tasks_current(self):
        self.calls.append("tasks_current")
        return make_task([dose(1)], status="in_progress")

    async def tasks_next(self, wait):
        self.calls.append(f"tasks_next:{wait}")
        self.runner_ref[0].stopping.set()
        return None

    async def task_status(self, task_id, status, detail=None):
        self.calls.append(f"status:{status}")


def test_task_loop_resumes_the_held_task_first_then_polls():
    ref = []
    app = TaskApp(ref)
    runner = Runner(app=app, robot=FakeRobot(reachable=False), clips=FakeClips(), stream=FakeStream(),
                    tick_seconds=0, run_blocking=inline)
    ref.append(runner)
    asyncio.run(runner._task_loop())
    assert app.calls == ["tasks_current", "status:aborted", "tasks_next:25"]


def test_shutdown_during_a_slot_keeps_the_lease():
    ref = []
    app = TaskApp(ref)
    robot = FakeRobot()
    runner = Runner(app=app, robot=robot, clips=FakeClips(), stream=FakeStream(), tick_seconds=0, run_blocking=inline)
    runner.request_shutdown()
    slot = asyncio.run(runner.run_slot(make_task([dose(1)])))
    assert slot.result == "bridge_shutdown" and robot.calls == ["sleep"]
    assert not [call for call in app.calls if call.startswith("status:")]


def test_after_failing_closed_on_app_loss_the_held_task_is_resumed_via_tasks_current():
    class LoopApp:
        def __init__(self):
            self.calls = []
            self.unreachable_since = None

        async def tasks_current(self):
            self.calls.append("tasks_current")
            if self.calls.count("tasks_current") == 2:
                raise AppUnreachable("still down")   # retried until it answers
            return make_task([dose(1)], status="in_progress")

        async def tasks_next(self, wait):
            self.calls.append("tasks_next")
            runner.stopping.set()

    class Result:
        def __init__(self, result):
            self.result = result

    app = LoopApp()
    runner = Runner(app=app, robot=FakeRobot(), clips=FakeClips(), stream=FakeStream(), run_blocking=inline)
    results = iter(["app_unreachable", "completed"])

    async def fake_run_slot(task):
        app.calls.append("slot")
        return Result(next(results))

    async def no_pause(seconds):
        return None

    runner.run_slot = fake_run_slot
    runner._pause = no_pause
    asyncio.run(runner._task_loop())
    assert app.calls == ["tasks_current", "slot", "tasks_current", "tasks_current", "slot", "tasks_next"]
