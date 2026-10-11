"""ReachyRobot.reopen_camera() and media_status(): the camera watchdog's recovery (spec 2026-10-11 A2) through the
SDK's public release_media() and acquire_media() (reachy_mini 1.11.0, reachy_mini.py:203-240)."""

import asyncio
import threading
import time
import wave

import httpx
import numpy as np
import pytest

from medcare_reachy.bridge import media as media_module
from medcare_reachy.bridge.media import RESTORE_RETRY_SECONDS, ReachyRobot
from medcare_reachy.bridge.runner import MonitorStream, Runner
from medcare_reachy.bridge.tests.fakes import FakeClips

RELEASED = {"available": False, "released": True, "no_media": False}
AVAILABLE = {"available": True, "released": False, "no_media": False}


class Media:
    """One SDK media manager: hands out frames and microphone samples until it is closed."""

    def __init__(self, name):
        self.name, self.closed, self.recording, self.pushed = name, False, False, 0

    def get_frame(self):
        return None if self.closed else np.zeros((720, 1280, 3), np.uint8)

    def get_audio_sample(self):
        return None if self.closed else np.zeros((160, 2), np.float32)

    def get_input_audio_samplerate(self):
        return 16000

    def get_output_audio_samplerate(self):
        return 16000

    def start_playing(self):
        pass

    def push_audio_sample(self, samples):
        self.pushed += 1

    def start_recording(self):
        self.recording = True

    def close(self):
        self.closed = True


class MediaMini:
    """ReachyMini's media calls as SDK 1.11.0 makes them: release_media() closes the manager and marks it released
    (once); acquire_media() does nothing unless released, else asks the daemon and builds a fresh manager, or logs and
    stays released when the daemon refuses (`gives_back=False`)."""

    def __init__(self, gives_back=True, error=None):
        self.calls, self.gives_back, self.error = [], gives_back, error
        self.media_released = False
        self.managers = [Media("first")]

    @property
    def media(self):
        return self.managers[-1]

    def release_media(self):
        self.calls.append("release_media")
        if self.error is not None:
            raise self.error
        if self.media_released:
            return
        self.media.close()
        self.media_released = True

    def acquire_media(self):
        self.calls.append("acquire_media")
        if not self.media_released or not self.gives_back:
            return
        self.media.close()
        self.managers.append(Media(f"fresh {len(self.managers)}"))
        self.media_released = False


def robot_with(mini, status=RELEASED):
    robot = ReachyRobot.attach(mini)
    robot.media_status = lambda: status
    return robot


def test_a_camera_the_daemon_released_is_acquired_back_with_a_fresh_media_manager():
    mini = MediaMini()
    robot = robot_with(mini, RELEASED)
    assert robot.reopen_camera() == "acquired"
    assert mini.calls == ["release_media", "acquire_media"]
    assert mini.managers[0].closed and mini.media.name == "fresh 1"
    assert mini.media.recording                       # the microphone's pipeline runs again
    assert robot.get_camera_frame() is not None       # read from the fresh manager


def test_a_stale_reader_of_a_daemon_that_has_its_media_is_reopened():
    mini = MediaMini()
    robot = robot_with(mini, AVAILABLE)
    assert robot.reopen_camera() == "reopened"
    assert mini.calls == ["release_media", "acquire_media"] and mini.media.name == "fresh 1"


def test_a_refused_acquire_fails_and_the_next_attempt_only_asks_again():
    mini = MediaMini(gives_back=False)
    robot = robot_with(mini, RELEASED)
    assert robot.reopen_camera() == "failed: the daemon did not give the media back"
    assert mini.media_released
    # Meanwhile nothing reads the closed manager (the SDK would log a warning on every call).
    assert robot.get_camera_frame() is None and robot.get_audio() is None
    mini.gives_back = True
    assert robot.reopen_camera() == "acquired"
    assert mini.calls == ["release_media", "acquire_media", "acquire_media"]
    assert robot.get_camera_frame() is not None and robot.get_audio() is not None


def test_a_daemon_without_media_is_not_cycled():
    mini = MediaMini()
    robot = robot_with(mini, {"available": False, "released": False, "no_media": True})
    assert robot.reopen_camera() == "failed: the daemon runs without media"
    assert mini.calls == []


def test_an_sdk_error_is_a_failed_attempt_never_an_exception(caplog):
    mini = MediaMini(error=RuntimeError("daemon unreachable"))
    robot = robot_with(mini, None)
    with caplog.at_level("ERROR", logger="medcare_reachy.bridge.media"):
        assert robot.reopen_camera() == "failed: RuntimeError('daemon unreachable')"
    assert any("camera recovery failed" in r.getMessage() for r in caplog.records)


def test_no_connection_cannot_reopen():
    robot = ReachyRobot("192.168.1.50")
    assert robot.reopen_camera() == "failed: no robot connection"


def test_a_recovery_waits_for_the_clip_still_playing():
    mini = MediaMini()
    robot = robot_with(mini, AVAILABLE)
    robot._quiet_at = time.monotonic() + 0.3      # a clip still playing for 0.3 s
    began = time.perf_counter()                   # monotonic() ticks every 15.6 ms on Windows
    assert robot.reopen_camera() == "reopened"
    assert time.perf_counter() - began >= 0.25


def test_a_clip_after_a_recovery_plays_on_the_fresh_manager(tmp_path):
    import wave

    path = tmp_path / "thanks.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.zeros(1600, "<i2").tobytes())
    mini = MediaMini()
    robot = robot_with(mini, RELEASED)
    robot.reopen_camera()
    assert robot.play_clip(path, wait=False) is True
    assert mini.media.pushed > 0 and mini.managers[0].pushed == 0
    mini.media_released = True                                    # a recovery between its release and its acquire
    assert robot.play_clip(path, wait=False) is False


class Response:
    def __init__(self, status=200, body=None):
        self.status_code, self.body = status, body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)

    def json(self):
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


@pytest.mark.parametrize("answer, expected", [
    (Response(200, RELEASED), RELEASED),
    (Response(503, {"detail": "Backend not running"}), None),
    (Response(200, ValueError("not JSON")), None),
    (Response(200, ["not", "a", "dict"]), None),
    (httpx.ConnectError("refused"), None),
])
def test_media_status_reads_the_local_daemon_within_2_s(monkeypatch, answer, expected):
    seen = []

    def get(url, timeout):
        seen.append((url, timeout))
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(media_module.httpx, "get", get)
    assert ReachyRobot.attach(MediaMini()).media_status() == expected
    assert seen == [("http://127.0.0.1:8000/api/media/status", 2.0)]


# ── restore_media(): media a refused recovery left released (final review of 11 Oct, R1) ──────────────────────
# release_media() closes the whole media manager: camera, speaker and microphone. When the daemon then refuses to
# give the media back, only an attached, stalled session's watchdog used to ask again, so after a give-up the app
# stayed mute and deaf until it restarted.

def _released(mini):
    """This app's media released by a recovery, as the SDK leaves it; the test's own call is not counted."""
    mini.release_media()
    mini.calls.clear()


def test_restore_media_is_a_no_op_while_the_app_has_its_media():
    mini = MediaMini()
    robot = robot_with(mini)
    assert robot.restore_media() is True
    assert mini.calls == [] and mini.media.name == "first"


def test_restore_media_only_asks_for_released_media_back_and_never_releases():
    mini = MediaMini()
    robot = robot_with(mini)
    _released(mini)
    assert robot.get_audio() is None
    assert robot.restore_media() is True
    assert mini.calls == ["acquire_media"] and mini.media.name == "fresh 1"
    assert mini.media.recording                       # the microphone's pipeline runs again
    assert robot.get_audio() is not None and robot.get_camera_frame() is not None
    assert robot.restore_media() is True and mini.calls == ["acquire_media"]   # back: nothing more to do


def test_a_refused_restore_warns_once_per_streak_and_never_raises(caplog):
    mini = MediaMini(gives_back=False)
    robot = robot_with(mini)
    _released(mini)
    clock = [100.0]
    robot._now = lambda: clock[0]
    with caplog.at_level("WARNING", logger="medcare_reachy.bridge.media"):
        assert robot.restore_media() is False
        clock[0] += RESTORE_RETRY_SECONDS
        assert robot.restore_media() is False
        assert len(caplog.records) == 1
        assert "the speaker and microphone stay off" in caplog.records[0].getMessage()
        mini.gives_back = True
        clock[0] += RESTORE_RETRY_SECONDS
        assert robot.restore_media() is True
        mini.gives_back = False
        _released(mini)
        assert robot.restore_media() is False          # a new streak warns again

        def broken():
            raise RuntimeError("daemon unreachable")

        mini.acquire_media = broken
        assert robot.restore_media() is False          # an SDK error is a failure, never an exception
    assert [r.levelname for r in caplog.records] == ["WARNING", "WARNING"]
    assert "release_media" not in mini.calls


def test_restore_media_holds_the_playback_lock():
    mini = MediaMini()
    robot = robot_with(mini)
    _released(mini)
    with robot._lock:                                  # a clip being handed to the speaker, or a recovery
        worker = threading.Thread(target=robot.restore_media)
        worker.start()
        worker.join(0.2)
        assert worker.is_alive() and mini.calls == []
    worker.join(2.0)
    assert not worker.is_alive() and mini.calls == ["acquire_media"]


def test_no_connection_has_no_media_to_restore():
    assert ReachyRobot("192.168.1.50").restore_media() is False


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class _App:
    unreachable_since = None

    async def heartbeat(self, payload, camera=None):
        return {"stop_all": False}


async def _inline(fn, *args):
    return fn(*args)


def test_after_a_refused_recovery_gives_up_the_next_heartbeat_gets_the_speaker_and_microphone_back(tmp_path, caplog):
    path = tmp_path / "thanks.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.zeros(1600, "<i2").tobytes())
    mini = MediaMini(gives_back=False)                 # the daemon refuses to give the media back
    robot = robot_with(mini, RELEASED)
    mini.managers[0].get_frame = lambda: None          # the camera feed is dead; its audio still works
    clock = _Clock()
    robot._now = lambda: clock.t                       # restore_media() paces its retries on the same clock
    stream = MonitorStream(_App(), robot, None, clock=clock, run_blocking=_inline, stream_encode=lambda frame: b"j")
    clips = FakeClips()
    clips.missing_count = 0
    runner = Runner(app=_App(), robot=robot, clips=clips, stream=stream, run_blocking=_inline)
    assert robot.play_clip(path, wait=False) is True

    async def search_then_heartbeats():
        stream.attach({"session_id": "s1", "generation": "g1"})
        for i in range(1, 80):                         # 20 s of reads that find nothing
            clock.t = 1000 + i * 0.25
            await stream.step()
        assert stream.camera.state == "unavailable"
        stream.detach()                                # the search ends with camera_unavailable
        for _ in range(3):                             # heartbeats between tasks: the daemon still refuses
            clock.t += 10
            await runner.heartbeat_once()
        assert robot.play_clip(path, wait=False) is False and robot.get_audio() is None
        mini.gives_back = True                         # the daemon can give the media back now
        clock.t += RESTORE_RETRY_SECONDS               # the first heartbeat after the retry pause asks again
        await runner.heartbeat_once()

    with caplog.at_level("INFO"):
        asyncio.run(search_then_heartbeats())
    assert not mini.media_released and robot.play_clip(path, wait=False) is True
    assert robot.get_audio() is not None and mini.media.recording
    assert mini.calls.count("release_media") == 1      # the first recovery's; nothing after it releases
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert len([m for m in errors if "the speaker and microphone are off" in m]) == 1


def test_a_refused_restore_waits_a_minute_before_asking_the_daemon_again():
    # The SDK logs a WARNING and an ERROR for every refused acquire, and each request can take up to 10 s: asking at
    # every 10 s heartbeat filled the robot's log and delayed the heartbeat while the daemon kept refusing.
    mini = MediaMini(gives_back=False)
    robot = robot_with(mini)
    _released(mini)
    clock = [100.0]
    robot._now = lambda: clock[0]
    assert robot.restore_media() is False and mini.calls == ["acquire_media"]
    clock[0] += 10                                     # the next heartbeat
    assert robot.restore_media() is False and mini.calls == ["acquire_media"]   # not asked again yet
    clock[0] += RESTORE_RETRY_SECONDS - 10
    mini.gives_back = True
    assert robot.restore_media() is True and mini.calls == ["acquire_media", "acquire_media"]
    mini.gives_back = False
    _released(mini)
    assert robot.restore_media() is False and mini.calls == ["acquire_media"]   # a new streak asks at once
