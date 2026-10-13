import asyncio
import hashlib
import json
import logging
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from medcare_reachy import models
from medcare_reachy.service import BridgeService
from medcare_reachy.settings_store import SettingsError, SettingsStore, public_view
from medcare_reachy.web import register_routes

GOOD = {"app_url": "http://192.168.1.20:8001", "device_token": "rdv1.secret", "language": "zh-TW", "capture_fps": 10}


# ── settings ──
def test_settings_default_to_not_configured(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    assert not SettingsStore.configured(store.load())


def test_save_validates_and_persists(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    saved = store.save(GOOD)
    assert SettingsStore.configured(saved)
    assert json.loads((tmp_path / "settings.json").read_text())["device_token"] == "rdv1.secret"


def test_empty_key_on_the_form_keeps_the_saved_key(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    store.save(GOOD)
    saved = store.save({**GOOD, "device_token": "", "language": "en"})
    assert saved["device_token"] == "rdv1.secret" and saved["language"] == "en"


@pytest.mark.parametrize("change", [
    {"app_url": "ftp://x"}, {"app_url": "http://host:8001/api/device"}, {"device_token": "not-a-key"},
    {"language": "fr"}, {"capture_fps": 0}, {"capture_fps": 30}, {"capture_fps": "fast"},
])
def test_invalid_settings_are_rejected(tmp_path, change):
    with pytest.raises(SettingsError):
        SettingsStore(tmp_path / "s.json").save({**GOOD, **change})


def test_public_view_never_contains_the_key():
    view = public_view({**GOOD, "capture_fps": 10.0})
    assert "rdv1" not in json.dumps(view) and view["device_token_set"] is True


def test_the_check_in_mm_and_gestures_are_on_until_switched_off(tmp_path):
    store = SettingsStore(tmp_path / "settings.json")
    saved = store.save(GOOD)
    assert saved["checkin_ack"] is True and saved["checkin_gestures"] is True
    saved = store.save({"checkin_gestures": False})               # e.g. the settings page's checkbox
    assert saved["checkin_ack"] is True and saved["checkin_gestures"] is False
    assert public_view(store.load())["checkin_gestures"] is False
    assert public_view(GOOD)["checkin_ack"] is True                 # settings saved before 0.5.2
    for bad in ({"checkin_ack": "no"}, {"checkin_gestures": 0}):
        with pytest.raises(SettingsError):
            store.save(bad)


# ── models ──
def test_bundled_models_match_their_pinned_hashes():
    assert models.verify_models() == models.DEFAULT_DIR
    landmark_models = __import__("medcare_reachy.bridge.vision", fromlist=["x"]).MODEL_FILES
    assert set(landmark_models) | {"emotion_seed43.onnx", "emotion_seed43_metadata.json"} == set(models.MODELS)


def test_bundled_emotion_model_loads_and_scores_seven_emotions():
    import numpy as np
    from medcare_reachy.bridge.emotion import LABELS, EmotionEngine

    engine = EmotionEngine(models.DEFAULT_DIR / models.EMOTION_MODEL)
    scores = engine.predict(np.full((480, 640, 3), 128, np.uint8), [0.3, 0.2, 0.3, 0.4])
    assert tuple(scores) == LABELS and abs(sum(scores.values()) - 1) < 1e-6


def test_missing_or_altered_model_is_refused(tmp_path):
    expected = {"m.onnx": hashlib.sha256(b"model").hexdigest()}
    with pytest.raises(models.ModelError, match="missing"):
        models.verify_models(tmp_path, expected)
    (tmp_path / "m.onnx").write_bytes(b"evil")
    with pytest.raises(models.ModelError, match="integrity"):
        models.verify_models(tmp_path, expected)
    (tmp_path / "m.onnx").write_bytes(b"model")
    assert models.verify_models(tmp_path, expected) == tmp_path


def test_nothing_imports_mediapipe_or_opencv():
    """The robot's Raspberry Pi 4 can't load MediaPipe (SIGILL), and OpenCV isn't installed there."""
    import re
    package = Path(models.__file__).parent
    for source in package.rglob("*.py"):
        if "tests" in source.parts:
            continue
        text = source.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+mediapipe\b", text, re.M), source
        if source.name != "media.py":   # VideoFileRobot (hardware-free testing) is the only OpenCV user
            assert not re.search(r"^\s*import cv2\b", text, re.M), source


# ── service ──
class FakeRunner:
    def __init__(self):
        self.stopping = asyncio.Event()
        self.slot = None
        self.robot_reachable = True
        self.clips = type("C", (), {"missing_count": 11})()
        self.stream = type("S", (), {"landmark_fps": lambda self: 9.84, "vision_fps": lambda self: 4.0,
                                     "camera_fps": lambda self: 10.04})()

    def request_shutdown(self):
        self.stopping.set()


def fake_bridge(record):
    async def bridge(reachy_mini, settings, models_dir, on_runner):
        record.append(settings)
        runner = FakeRunner()
        on_runner(runner)
        await runner.stopping.wait()
    return bridge


def wait_for(predicate, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_service_waits_for_configuration(tmp_path):
    service = BridgeService(object(), SettingsStore(tmp_path / "s.json"), bridge=fake_bridge([]))
    service.start()
    assert service.status()["state"] == "not_configured"


def test_service_runs_reports_and_stops(tmp_path):
    store = SettingsStore(tmp_path / "s.json")
    store.save(GOOD)
    runs = []
    service = BridgeService(object(), store, bridge=fake_bridge(runs))
    service.start()
    assert wait_for(lambda: service.status()["state"] == "running")
    status = service.status()
    assert status["landmark_fps"] == 9.8 and status["missing_clips"] == 11 and status["robot_reachable"] is True
    assert status["camera_fps"] == 10.0   # what the camera delivers: the ceiling for the landmark rate
    service.stop()
    assert service.status()["state"] == "stopped" and runs[0]["capture_fps"] == 10.0


def test_bridge_failure_is_reported_not_raised(tmp_path):
    store = SettingsStore(tmp_path / "s.json")
    store.save(GOOD)

    async def broken(*args):
        raise models.ModelError("Vision model hand_detector.onnx failed its integrity check")

    service = BridgeService(object(), store, bridge=broken)
    service.start()
    assert wait_for(lambda: service.status()["state"] == "error")
    assert "integrity" in service.status()["detail"]


# ── one bridge at a time ──
def tracked_bridge(alive, runs, start_seconds=0.1, stop_seconds=0.2):
    """Shaped like default_bridge: a moment to start before it has a runner (robot, clips, voice, server client),
    and a moment to shut down. `alive` holds the threads whose bridge is running right now."""
    async def bridge(reachy_mini, settings, models_dir, on_runner):
        me = threading.current_thread()
        alive.add(me)
        runs.append(settings)
        try:
            await asyncio.sleep(start_seconds)
            runner = FakeRunner()
            on_runner(runner)
            await runner.stopping.wait()
            await asyncio.sleep(stop_seconds)   # monitor_end, robot.sleep(), voice.stop()
        finally:
            alive.discard(me)
    return bridge


def test_two_restarts_at_once_leave_one_bridge_and_it_is_the_tracked_one(tmp_path):
    """A double click on Save or Restart sends two requests, and the settings server runs them at the same time
    (its endpoints are plain functions, run on a thread pool). Both bridges used to drive the same robot."""
    store = SettingsStore(tmp_path / "s.json")
    store.save(GOOD)
    alive, runs = set(), []
    service = BridgeService(object(), store, bridge=tracked_bridge(alive, runs))
    service.start()
    assert wait_for(lambda: service.status()["state"] == "running")
    began = time.monotonic()
    callers = [threading.Thread(target=service.restart) for _ in range(2)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(20)
    assert time.monotonic() - began < 5   # the second restart doesn't sit out stop()'s 15 s timeout
    assert wait_for(lambda: service.status()["state"] == "running")
    assert alive == {service._thread}     # exactly one bridge, and stop() can reach it
    assert len(runs) == 3                 # each restart started a fresh bridge with the saved settings
    service.stop()
    assert not alive and service.status()["state"] == "stopped"


def test_a_bridge_that_outlives_stop_stays_tracked_and_no_second_one_starts(tmp_path, caplog):
    store = SettingsStore(tmp_path / "s.json")
    store.save(GOOD)
    alive, runs = set(), []
    release = threading.Event()

    async def slow_to_stop(reachy_mini, settings, models_dir, on_runner):
        me = threading.current_thread()
        alive.add(me)
        runs.append(settings)
        try:
            runner = FakeRunner()
            on_runner(runner)
            await runner.stopping.wait()
            release.wait(5)   # shutdown work that takes longer than stop() waits (a blocking robot call)
        finally:
            alive.discard(me)

    service = BridgeService(object(), store, bridge=slow_to_stop)
    service.start()
    assert wait_for(lambda: service.status()["state"] == "running")
    first = service._thread
    with caplog.at_level(logging.WARNING, logger="medcare_reachy"):
        service.stop(timeout=0.2)
        assert service.status()["state"] == "stopping"
        service.start()
    assert service._thread is first and alive == {first} and len(runs) == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("did not stop" in m for m in messages) and any("not starting a second" in m for m in messages)
    release.set()
    first.join(2)
    assert service.status()["state"] == "stopped"   # its own exit reports it
    service.start()   # once it has ended, a new bridge may start
    assert wait_for(lambda: service.status()["state"] == "running") and service._thread is not first
    service.stop()
    assert not alive


def test_a_stop_while_the_bridge_is_still_starting_ends_it_once_it_has_a_runner(tmp_path):
    store = SettingsStore(tmp_path / "s.json")
    store.save(GOOD)
    alive, runs = set(), []
    service = BridgeService(object(), store, bridge=tracked_bridge(alive, runs, start_seconds=0.3))
    service.start()
    began = time.monotonic()
    service.stop(timeout=5)
    assert time.monotonic() - began < 2   # stopped as soon as it had a runner, not after the timeout
    assert not alive and service.status()["state"] == "stopped"


# ── settings page API ──
def test_settings_api_saves_restarts_and_masks_key(tmp_path):
    store = SettingsStore(tmp_path / "s.json")
    runs = []
    service = BridgeService(object(), store, bridge=fake_bridge(runs))
    api = FastAPI()
    register_routes(api, service)
    client = TestClient(api)
    assert client.get("/api/config").json()["device_token_set"] is False
    response = client.post("/api/config", json=GOOD)
    assert response.status_code == 200 and "rdv1" not in response.text
    assert wait_for(lambda: service.status()["state"] == "running")
    assert client.post("/api/config", json={**GOOD, "device_token": "bad"}).status_code == 422
    service.stop()


# ── found in the simulator smoke test ──
def test_status_is_available_before_the_robot_is_attached(tmp_path):
    """The settings page loads while the runtime is still connecting to the robot; its API must answer."""
    store = SettingsStore(tmp_path / "s.json")
    store.save(GOOD)
    service = BridgeService(None, store, bridge=fake_bridge([]))
    api = FastAPI()
    register_routes(api, service)
    client = TestClient(api)
    assert client.get("/api/status").json()["state"] == "connecting_robot"
    assert client.post("/api/config", json=GOOD).status_code == 200   # saving works; it starts once attached
    assert service.status()["state"] == "connecting_robot"
    service.attach(object())
    service.start()
    assert wait_for(lambda: service.status()["state"] == "running")
    service.stop()


def test_status_describes_the_current_step_during_a_task(tmp_path):
    store = SettingsStore(tmp_path / "s.json")
    store.save(GOOD)

    def bridge_with_slot(record):
        async def bridge(reachy_mini, settings, models_dir, on_runner):
            runner = FakeRunner()
            runner.slot = type("Slot", (), {"state": "SEARCHING", "task": {"task_id": "t-1"}})()
            on_runner(runner)
            await runner.stopping.wait()
        return bridge

    service = BridgeService(object(), store, bridge=bridge_with_slot([]))
    service.start()
    assert wait_for(lambda: service.status().get("slot_state") == "SEARCHING")
    assert "Looking for the patient" in service.status()["detail"]
    service.stop()


# ── the real default bridge ──
def test_damaged_install_stops_with_a_clear_status_instead_of_crashing(tmp_path):
    store = SettingsStore(tmp_path / "s.json")
    store.save({**GOOD, "vision_on_server": False})   # on-robot vision needs the bundled models
    service = BridgeService(object(), store, models_dir=tmp_path / "empty")   # real default_bridge
    service.start()
    assert wait_for(lambda: service.status()["state"] == "error")
    assert "reinstall the app" in service.status()["detail"]


def test_server_vision_runs_without_the_vision_models(tmp_path):
    store = SettingsStore(tmp_path / "s.json")
    store.save(GOOD)
    assert store.load()["vision_on_server"] is True   # the default
    service = BridgeService(object(), store, models_dir=tmp_path / "empty")   # real default_bridge
    service.start()
    try:
        assert wait_for(lambda: service.status()["state"] == "running")
    finally:
        service.stop()


# ── version ──
def test_the_package_and_the_heartbeat_report_the_same_version():
    """The app store installs pyproject's version; heartbeats report bridge.__version__ (spec 2026-10-12 F3)."""
    import tomllib

    from medcare_reachy.bridge import __version__

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"] == __version__ == "0.10.0"
