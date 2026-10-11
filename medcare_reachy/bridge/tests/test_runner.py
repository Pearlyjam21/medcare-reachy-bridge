import asyncio

import numpy as np

from medcare_reachy.bridge import __version__
from medcare_reachy.bridge import runner as runner_module
from medcare_reachy.bridge.app_client import AppUnreachable, NotAuthorised, SessionLost
from medcare_reachy.bridge.camera_watch import CameraWatchdog
from medcare_reachy.bridge.runner import MonitorStream, Runner
from medcare_reachy.bridge.tests.fakes import FakeClips, FakeRobot, FakeStream, dose, make_task


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

    def get_camera_frame(self):
        return self.get_frame()


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
        self.target_box = None   # set once the fake server "verifies" the patient

    async def monitor_landmarks(self, packet):
        self.landmarks.append(packet)
        if self.gate:
            await self.gate.wait()
        if self.fail_landmarks:
            raise self.fail_landmarks
        return {"session_id": packet["session_id"], "frame_seq": packet["frame_seq"], "identity_status": "verified",
                "target_box": self.target_box}

    async def monitor_vision(self, session_id, generation, frame_seq, jpeg):
        self.vision.append((session_id, generation, frame_seq, jpeg))
        return {"session_id": session_id, "frame_seq": frame_seq, "identity_status": "verified"}


def make_stream():
    clock, app, camera = Clock(), StreamApp(), Camera()
    stream = MonitorStream(app, camera, Engine(), clock=clock, run_blocking=inline, encode=lambda frame: b"jpg")
    return stream, app, camera, clock


class FrameApp:
    """The server-vision endpoint: the robot sends JPEGs, the server answers with the session state."""

    def __init__(self):
        self.frames = []
        self.gate = None
        self.unreachable_since = None

    async def monitor_frame(self, session_id, generation, frame_seq, timestamp, jpeg):
        self.frames.append((session_id, generation, frame_seq, timestamp, jpeg))
        if self.gate:
            await self.gate.wait()
        return {"session_id": session_id, "frame_seq": frame_seq, "identity_status": "verified"}


def make_server_stream(fps=15.0):
    clock, app, camera = Clock(), FrameApp(), Camera()
    stream = MonitorStream(app, camera, None, clock=clock, run_blocking=inline, encode=lambda frame: b"big",
                           stream_encode=lambda frame: b"jpg", fps=fps)
    return stream, app, camera, clock


def test_server_vision_streams_jpegs_with_capture_timestamps():
    stream, app, camera, clock = make_server_stream()

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        for _ in range(3):
            await stream.step()
            await stream.drain()
            clock.t += 1 / 15

    asyncio.run(scenario())
    assert [frame[2] for frame in app.frames] == [1, 2, 3]
    assert app.frames[0][:2] == ("s1", "g1") and app.frames[0][4] == b"jpg"
    assert app.frames[0][3] == 50.0 and abs(app.frames[1][3] - (50 + 1 / 15)) < 1e-9
    assert stream.latest["frame_seq"] == 3 and stream.landmark_fps() > 0


def test_server_vision_keeps_a_few_frames_in_flight_then_drops_rather_than_queues():
    stream, app, camera, clock = make_server_stream()

    async def scenario():
        app.gate = asyncio.Event()
        stream.attach({"session_id": "s1", "generation": "g1"})
        for _ in range(stream.max_in_flight):
            await stream.step()       # slow network: these stay in flight together
            clock.t += 1 / 15
        await asyncio.sleep(0)        # let the send tasks start
        assert [frame[2] for frame in app.frames] == [1, 2, 3]
        await stream.step()           # every slot busy for the whole wait: dropped
        assert len(app.frames) == 3 and camera.grabs == 4
        app.gate.set()
        await stream.drain()
        await stream.step()
        await stream.drain()

    asyncio.run(scenario())
    assert [frame[2] for frame in app.frames] == [1, 2, 3, 4]   # sequence numbers stay contiguous
    assert stream._frames_in_flight == 0


def stream_camera(stream, clock, frame_times):
    """Hand the stream one camera frame at each time, as the loop would when the camera delivers it."""
    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        for at in frame_times:
            clock.t = at
            await stream.step()
            await stream.drain()

    asyncio.run(scenario())


def test_server_stream_takes_every_frame_of_a_15_fps_camera_despite_jitter():
    stream, app, camera, clock = make_server_stream()
    jitter = [0.004, -0.004, 0.003, -0.002, 0.0]
    times = [50 + i / 15 + jitter[i % len(jitter)] for i in range(45)]
    stream_camera(stream, clock, times)
    assert len(app.frames) == 45    # a strict 1/15 s gate would drop each frame that came a few ms early
    assert [frame[3] for frame in app.frames] == times


def test_server_stream_sends_at_most_15_fps_from_a_faster_camera():
    stream, app, camera, clock = make_server_stream()
    stream_camera(stream, clock, [50 + i / 30 for i in range(90)])    # 3 s at 30 fps
    assert len(app.frames) == 45 and camera.grabs == 90
    gaps = {round(b[3] - a[3], 3) for a, b in zip(app.frames, app.frames[1:])}
    assert gaps == {round(2 / 30, 3)}


def test_a_lower_stream_rate_keeps_its_schedule_on_a_15_fps_camera():
    stream, app, camera, clock = make_server_stream(fps=10.0)
    stream_camera(stream, clock, [50 + i / 15 for i in range(45)])    # 3 s
    assert len(app.frames) == 30                                       # 2 frames in 3, not every other one


def test_after_the_camera_stalls_the_stream_does_not_burst_to_catch_up():
    stream, app, camera, clock = make_server_stream()
    times = [50 + i / 15 for i in range(15)] + [52 + i / 30 for i in range(30)]   # 1 s stall, then 30 fps
    stream_camera(stream, clock, times)
    later = [frame[3] for frame in app.frames if frame[3] >= 52]
    assert all(b - a > 0.06 for a, b in zip(later, later[1:]))


def test_camera_rate_is_measured_and_a_slow_camera_is_logged_once(caplog):
    stream, app, camera, clock = make_server_stream()
    with caplog.at_level("WARNING", logger="medcare_reachy.bridge.runner"):
        stream_camera(stream, clock, [50 + i / 10 for i in range(150)])    # 15 s at 10 fps, the daemon's cap
    assert len(app.frames) == 150
    assert 9.8 <= stream.camera_fps() <= 10.2 and 9.8 <= stream.landmark_fps() <= 10.2
    warnings = [r.getMessage() for r in caplog.records if "camera hands this app only" in r.getMessage()]
    assert len(warnings) == 1 and "IPC_FPS" in warnings[0]


def test_a_15_fps_camera_is_not_reported_as_slow(caplog):
    stream, app, camera, clock = make_server_stream()
    with caplog.at_level("WARNING", logger="medcare_reachy.bridge.runner"):
        stream_camera(stream, clock, [50 + i / 15 for i in range(225)])
    assert stream.camera_fps() >= 14.5
    assert not [r for r in caplog.records if "camera hands this app only" in r.getMessage()]


def stall_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("server stream:")]


def test_a_steady_stream_logs_no_stall(caplog):
    stream, app, camera, clock = make_server_stream()
    with caplog.at_level("WARNING", logger="medcare_reachy.bridge.runner"):
        stream_camera(stream, clock, [50 + i / 15 for i in range(45)])
    assert stall_lines(caplog) == []


def test_a_camera_stall_is_logged_with_how_long_the_camera_gave_nothing(caplog):
    stream, app, camera, clock = make_server_stream()

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        await stream.step()                  # frame 1 at 50.0
        await stream.drain()
        frame, camera.frame = camera.frame, None
        for i in range(1, 21):               # 1 s of polls with no new camera frame
            clock.t = 50 + i * 0.05
            await stream.step()
        camera.frame = frame
        clock.t = 51.1
        await stream.step()                  # frame 2
        await stream.drain()

    with caplog.at_level("WARNING", logger="medcare_reachy.bridge.runner"):
        asyncio.run(scenario())
    [line] = stall_lines(caplog)
    assert "1.10 s between frames before frame 2" in line
    assert "longest wait for a camera frame 1.05 s" in line and "free request slot: 0 " in line


def test_a_server_or_wifi_stall_is_logged_with_the_dropped_frames_and_the_oldest_request(caplog, monkeypatch):
    monkeypatch.setattr(runner_module, "FRAME_WAIT_STEPS", 1)
    stream, app, camera, clock = make_server_stream()

    async def scenario():
        app.gate = asyncio.Event()           # no answers: the requests stay in flight
        stream.attach({"session_id": "s1", "generation": "g1"})
        for i in range(15):                  # 1 s at 15 fps: 3 go out, 12 find no free slot
            clock.t = 50 + i / 15
            await stream.step()
        app.gate.set()
        await stream.drain()
        clock.t = 51.0
        await stream.step()                  # frame 4, 0.87 s after frame 3
        await stream.drain()

    with caplog.at_level("WARNING", logger="medcare_reachy.bridge.runner"):
        asyncio.run(scenario())
    assert [frame[2] for frame in app.frames] == [1, 2, 3, 4]
    [line] = stall_lines(caplog)
    assert "0.87 s between frames before frame 4" in line
    assert "free request slot: 12 (oldest unanswered request 0.93 s)" in line


def test_stall_lines_are_rate_limited_and_count_the_gaps_between_them(caplog):
    stream, app, camera, clock = make_server_stream()
    times = [50 + i * 0.5 for i in range(12)]    # a 2 fps camera: every gap is a stall, for 5.5 s
    with caplog.at_level("WARNING", logger="medcare_reachy.bridge.runner"):
        stream_camera(stream, clock, times)
    lines = stall_lines(caplog)
    assert len(lines) == 6                         # at 50.5, 51.5, ... 55.5
    assert "(1 gap(s)" in lines[0] and all("(2 gap(s)" in line for line in lines[1:])


def test_loop_waits_for_the_next_frame_only_while_streaming():
    stream, app, camera, clock = make_server_stream()
    assert abs(stream._pause(clock.t) - 1 / 15) < 1e-9          # idle: one interval, never a busy loop
    stream_camera(stream, clock, [50.0])
    assert abs(stream._pause(clock.t) - 0.8 / 15) < 1e-9        # just sent: sleep until the next may go out
    clock.t += 1
    assert stream._pause(clock.t) == 0.005                       # overdue: poll the camera again soon


def test_streamed_frames_are_smaller_than_the_camera_frame():
    import io

    import numpy as np
    from PIL import Image

    from medcare_reachy.bridge.runner import STREAM_SIZE, encode_jpeg, encode_stream_jpeg

    frame = (np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8) // 16 * 16)
    small = encode_stream_jpeg(frame)
    assert Image.open(io.BytesIO(small)).size == STREAM_SIZE
    assert len(small) < len(encode_jpeg(frame)) * 0.7


def test_stream_jpeg_from_the_wireless_camera_frame_is_its_4_3_centre_in_rgb():
    import io

    from PIL import Image

    from medcare_reachy.bridge.runner import STREAM_SIZE, encode_stream_jpeg

    frame = np.zeros((720, 1280, 3), np.uint8)    # BGR, as the SDK hands it over
    frame[:, :160] = (0, 255, 0)                   # outside the 4:3 centre: must not show
    frame[:, 1120:] = (0, 255, 0)
    frame[:, 160:640] = (255, 0, 0)                # left half of the centre: blue
    frame[:, 640:1120] = (0, 0, 255)               # right half: red
    image = Image.open(io.BytesIO(encode_stream_jpeg(frame))).convert("RGB")
    assert image.size == STREAM_SIZE
    pixels = np.asarray(image).astype(int)
    left, right = pixels[:, 4:236].mean(axis=(0, 1)), pixels[:, 244:476].mean(axis=(0, 1))
    assert left[2] > 200 and left[0] < 40 and left[1] < 40      # blue stays blue
    assert right[0] > 200 and right[2] < 40 and right[1] < 40    # red stays red
    assert pixels[:, [0, -1], 1].mean() < 40                      # no green bars: the sides were cropped


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


def test_vision_at_most_2_fps_and_after_its_landmark_packet():
    stream, app, camera, clock = make_stream()

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        for _ in range(16):
            await stream.step()
            await stream.drain()
            clock.t += 0.07

    asyncio.run(scenario())
    assert [seq for _, _, seq, _ in app.vision] == [1, 9]
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
    assert 1.8 <= stream.vision_fps() <= 2.0
    clock.t += 11
    assert stream.landmark_fps() == 0


class HeartbeatApp:
    def __init__(self, response=None, error=None):
        self.payloads, self.cameras, self.response, self.error = [], [], response, error
        self.unreachable_since = None

    async def heartbeat(self, payload, camera=None):
        self.payloads.append(payload)
        self.cameras.append(camera)
        if self.error:
            raise self.error
        return self.response


class StubCamera:
    SNAPSHOT = {"state": "ok", "fps": 10.0, "since": 1760000000.0, "media": None}

    def snapshot(self):
        return dict(self.SNAPSHOT)


class StubStream:
    def __init__(self):
        self.camera = StubCamera()

    def landmark_fps(self):
        return 14.8

    def vision_fps(self):
        return 4.9


class StubSlot:
    def __init__(self):
        self.stopped = None
        self.task = {"microphone": True}

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
    assert app.cameras == [StubCamera.SNAPSHOT]
    assert runner.slot.stopped == ("stop_all", True)


def test_every_heartbeat_asks_for_media_a_refused_recovery_left_released():
    runner = heartbeat_runner(HeartbeatApp({"stop_all": False}))
    asyncio.run(runner.heartbeat_once())
    asyncio.run(runner.heartbeat_once())
    assert runner.robot.calls == ["is_reachable", "restore_media"] * 2
    offline = heartbeat_runner(HeartbeatApp({"stop_all": False}))
    offline.robot = FakeRobot(reachable=False)
    asyncio.run(offline.heartbeat_once())
    assert offline.robot.calls == ["is_reachable"]     # no robot: nothing to ask


def test_heartbeat_not_authorised_stops_and_unreachable_does_not():
    runner = heartbeat_runner(HeartbeatApp(error=NotAuthorised(status=403)))
    runner.slot = StubSlot()
    asyncio.run(runner.heartbeat_once())
    assert runner.slot.stopped == ("stop_all", True)
    assert runner.slot.task["microphone"] is False
    runner = heartbeat_runner(HeartbeatApp(error=AppUnreachable("x")))
    runner.slot = StubSlot()
    asyncio.run(runner.heartbeat_once())
    assert runner.slot.stopped is None
    assert runner.slot.task["microphone"] is True   # no answer: keep what the task said


def test_heartbeat_microphone_consent_reaches_the_running_slot():
    runner = heartbeat_runner(HeartbeatApp({"stop_all": False, "microphone": False}))
    runner.slot = StubSlot()
    asyncio.run(runner.heartbeat_once())
    assert runner.slot.task["microphone"] is False and runner.slot.stopped is None


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


def test_shutdown_does_not_wait_out_the_task_long_poll():
    class PollingApp:
        unreachable_since = None

        async def tasks_current(self):
            return None

        async def tasks_next(self, wait):
            await asyncio.sleep(wait)   # the server's 25 s long poll

        async def heartbeat(self, payload, camera=None):
            return {}

    stream = StubStream()
    stream.run = lambda stop: stop.wait()
    runner = Runner(app=PollingApp(), robot=FakeRobot(), clips=FakeClips(), stream=stream, run_blocking=inline)

    async def scenario():
        started = asyncio.get_running_loop().time()
        run = asyncio.create_task(runner.run())
        await asyncio.sleep(0.05)
        runner.request_shutdown()
        await asyncio.wait_for(run, 2)
        return asyncio.get_running_loop().time() - started

    assert asyncio.run(scenario()) < 1


def test_shutdown_during_a_slot_keeps_the_lease():
    ref = []
    app = TaskApp(ref)
    robot = FakeRobot()
    runner = Runner(app=app, robot=robot, clips=FakeClips(), stream=FakeStream(), tick_seconds=0, run_blocking=inline)
    runner.request_shutdown()
    slot = asyncio.run(runner.run_slot(make_task([dose(1)])))
    assert slot.result == "bridge_shutdown" and robot.calls == ["sleep"]
    assert not [call for call in app.calls if call.startswith("status:")]


def test_the_settings_switch_the_check_in_mm_and_gestures_for_every_slot():
    runner = Runner(app=TaskApp([]), robot=FakeRobot(), clips=FakeClips(), stream=FakeStream(), tick_seconds=0,
                    run_blocking=inline, ack=False, gestures=False)
    runner.request_shutdown()
    slot = asyncio.run(runner.run_slot(make_task([dose(1)])))
    assert slot.ack is False and slot.gestures is False


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


# ── emotion scored on the robot ──
class FakeEmotion:
    def __init__(self, report=None):
        self.calls = []
        self.report = report if report is not None else {"face_index": 0, "probabilities": {"sad": 1.0}}

    def score(self, frame, packet, target_box):
        self.calls.append(target_box)
        return self.report


def test_emotion_rides_on_the_landmark_packet_only_once_the_server_names_the_face():
    clock, app, camera = Clock(), StreamApp(), Camera()
    emotion = FakeEmotion()
    stream = MonitorStream(app, camera, Engine(), clock=clock, run_blocking=inline, encode=lambda frame: b"jpg",
                           emotion=emotion)

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        await stream.step()
        await stream.drain()
        assert "emotion" not in app.landmarks[0] and emotion.calls == []    # no verified face yet
        app.target_box = [0.1, 0.1, 0.4, 0.4]
        stream.latest = {**stream.latest, "target_box": app.target_box}
        for _ in range(12):
            clock.t += 1 / 15
            await stream.step()
            await stream.drain()

    asyncio.run(scenario())
    with_emotion = [p["frame_seq"] for p in app.landmarks if "emotion" in p]
    assert len(with_emotion) == 2 and with_emotion[0] == 2 and with_emotion[1] - 2 >= 8   # at most every 0.5 s
    assert app.landmarks[1]["emotion"] == {"face_index": 0, "probabilities": {"sad": 1.0}}
    assert emotion.calls == [[0.1, 0.1, 0.4, 0.4]] * 2


def test_emotion_failure_never_drops_the_landmark_frame():
    clock, app, camera = Clock(), StreamApp(), Camera()

    class Broken:
        def score(self, *args):
            raise RuntimeError("model crashed")

    stream = MonitorStream(app, camera, Engine(), clock=clock, run_blocking=inline, encode=lambda frame: b"jpg",
                           emotion=Broken())

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1", "target_box": [0.1, 0.1, 0.4, 0.4]})
        await stream.step()
        await stream.drain()

    asyncio.run(scenario())
    assert len(app.landmarks) == 1 and "emotion" not in app.landmarks[0]


# ── camera watchdog (spec 2026-10-11 A2, A3) ──
RELEASED = {"available": False, "released": True, "no_media": False}


class DyingCamera(Camera):
    """A camera whose feed dies (`frame = None`) until reopen_camera() brings it back (`revives`)."""

    def __init__(self, revives=True):
        super().__init__()
        self.alive_frame, self.revives, self.reopened = self.frame, revives, 0

    def reopen_camera(self):
        self.reopened += 1
        if not self.revives:
            return "failed: the daemon did not give the media back"
        self.frame = self.alive_frame
        return "acquired"

    def media_status(self):
        return RELEASED


def make_dying_stream(engine=None, revives=True):
    clock, app, camera = Clock(), FrameApp(), DyingCamera(revives)
    stream = MonitorStream(app, camera, engine, clock=clock, run_blocking=inline, encode=lambda frame: b"big",
                           stream_encode=lambda frame: b"jpg")
    return stream, app, camera, clock


def test_the_stream_notices_a_dead_camera_and_gets_it_back(caplog):
    stream, app, camera, clock = make_dying_stream()

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        await stream.step()                       # frame 1 at 50.0
        await stream.drain()
        camera.frame = None                       # the daemon's camera was released
        states = []
        for i in range(1, 9):                     # reads every 0.25 s
            clock.t = 50 + i * 0.25
            await stream.step()
            states.append(stream.camera.state)
        clock.t = 52.25
        await stream.step()                       # the reopened camera hands over a frame again
        await stream.drain()
        first = stream.camera.state, [frame[2] for frame in app.frames], list(caplog.records)
        for i in range(1, 21):                    # steady frames for 5 s more: the stall is over
            clock.t = 52.25 + i * 0.25
            await stream.step()
        return states, first

    with caplog.at_level("INFO", logger="medcare_reachy.bridge.camera_watch"):
        states, (state_at_first_frame, sent_at_first_frame, records_at_first_frame) = asyncio.run(scenario())
    assert states == ["ok"] * 7 + ["recovering"] and camera.reopened == 1
    assert state_at_first_frame == "ok" and sent_at_first_frame == [1, 2] and stream.camera.state == "ok"
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("camera stalled: no new frame for 2.0 s") and "'released': True" in m for m in messages)
    assert not any(r.getMessage().startswith("camera recovered") for r in records_at_first_frame)   # not yet
    assert any(m.startswith("camera recovered: new frames for 5.0 s without a gap; the stall began 5.2 s ago")
               for m in messages)


def test_a_camera_that_stays_dead_is_unavailable_after_15_s_of_stall():
    stream, app, camera, clock = make_dying_stream(revives=False)

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        await stream.step()
        await stream.drain()
        camera.frame = None
        for i in range(1, 69):                    # 17 s of reads that find nothing
            clock.t = 50 + i * 0.25
            await stream.step()

    asyncio.run(scenario())
    assert stream.camera.state == "unavailable" and camera.reopened == 3   # at 52, 57 and 62
    stream.detach()                                                        # the search ended: still reported
    assert stream.camera.snapshot()["state"] == "unavailable"


def test_on_robot_vision_watches_the_camera_too():
    stream, app, camera, clock = make_dying_stream(engine=Engine())

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        camera.frame = None
        for i in range(1, 9):
            clock.t = 50 + i * 0.25
            await stream.step()

    asyncio.run(scenario())
    assert camera.reopened == 1 and stream.camera.state == "recovering"


def test_the_slow_camera_warning_can_fire_again_after_a_recovery(caplog):
    stream, app, camera, clock = make_dying_stream()

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        for i in range(101):                      # 10 s at 10 fps, the daemon's cap: warned
            clock.t = 50 + i / 10
            await stream.step()
            await stream.drain()
        camera.frame = None
        for i in range(1, 13):                    # 3 s without a frame: stall, recovery
            clock.t = 60 + i * 0.25
            await stream.step()
        for i in range(101):                      # 10 s at 10 fps again: warned again
            clock.t = 63.25 + i / 10
            await stream.step()
            await stream.drain()

    with caplog.at_level("WARNING", logger="medcare_reachy.bridge.runner"):
        asyncio.run(scenario())
    assert camera.reopened == 1
    assert len([r for r in caplog.records if "camera hands this app only" in r.getMessage()]) == 2


def test_the_heartbeat_reports_the_camera_in_each_state():
    stream, app, camera, clock = make_dying_stream(revives=False)
    beats = HeartbeatApp({"stop_all": False})
    clips = FakeClips()
    clips.missing_count = 0
    runner = Runner(app=beats, robot=FakeRobot(), clips=clips, stream=stream, run_blocking=inline)

    async def scenario():
        await runner.heartbeat_once()                         # no session: idle
        stream.attach({"session_id": "s1", "generation": "g1"})
        await stream.step()
        await stream.drain()
        await runner.heartbeat_once()                         # streaming: ok
        camera.frame = None
        for i in range(1, 9):
            clock.t = 50 + i * 0.25
            await stream.step()
        await runner.heartbeat_once()                         # the recovery failed: stalled
        for i in range(9, 69):
            clock.t = 50 + i * 0.25
            await stream.step()
        await runner.heartbeat_once()                         # given up: unavailable
        stream.detach()
        await runner.heartbeat_once()                         # still, while no session runs

    asyncio.run(scenario())
    assert [beat["state"] for beat in beats.cameras] == ["idle", "ok", "stalled", "unavailable", "unavailable"]
    assert beats.cameras[1]["fps"] == 0.1 and beats.cameras[2]["media"] == RELEASED
    assert all(set(beat) == {"state", "fps", "since", "media"} for beat in beats.cameras)


def test_a_robot_that_cannot_reopen_its_camera_only_reports_it():
    stream, app, camera, clock = make_server_stream()
    assert isinstance(stream.camera, CameraWatchdog)

    async def scenario():
        stream.attach({"session_id": "s1", "generation": "g1"})
        await stream.step()
        await stream.drain()
        camera.frame = None
        for i in range(1, 9):
            clock.t = 50 + i * 0.25
            await stream.step()

    asyncio.run(scenario())
    assert stream.camera.state == "stalled"
