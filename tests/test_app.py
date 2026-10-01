import asyncio
import hashlib
import json
import time

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


# ── models ──
def test_models_are_downloaded_once_and_verified(tmp_path, monkeypatch):
    payload = {name: name.encode() * 10 for name in models.MODELS}
    monkeypatch.setattr(models, "MODELS", {n: (f"https://x/{n}", hashlib.sha256(b).hexdigest()) for n, b in payload.items()})
    calls = []

    def download(url, destination):
        calls.append(url)
        destination.write_bytes(payload[url.rsplit("/", 1)[1]])

    models.ensure_models(tmp_path, download=download)
    models.ensure_models(tmp_path, download=download)
    assert len(calls) == 3   # second call finds valid files


def test_tampered_model_is_rejected_and_not_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "MODELS", {"m.task": ("https://x/m.task", "0" * 64)})
    with pytest.raises(models.ModelError):
        models.ensure_models(tmp_path, download=lambda url, dest: dest.write_bytes(b"evil"))
    assert not (tmp_path / "m.task").exists() and not (tmp_path / "m.part").exists()


def test_pinned_hashes_match_the_repo_models():
    repo = __import__("pathlib").Path(r"D:\medcareai2_20260920\app\frontend_source\public\models")
    if not repo.exists():
        pytest.skip("needs the full app repository")
    for name, (_, expected) in models.MODELS.items():
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == expected


# ── service ──
class FakeRunner:
    def __init__(self):
        self.stopping = asyncio.Event()
        self.slot = None
        self.robot_reachable = True
        self.clips = type("C", (), {"missing_count": 11})()
        self.stream = type("S", (), {"landmark_fps": lambda self: 9.84, "vision_fps": lambda self: 4.0})()

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
    service.stop()
    assert service.status()["state"] == "stopped" and runs[0]["capture_fps"] == 10.0


def test_bridge_failure_is_reported_not_raised(tmp_path):
    store = SettingsStore(tmp_path / "s.json")
    store.save(GOOD)

    async def broken(*args):
        raise models.ModelError("face_landmarker.task failed its integrity check")

    service = BridgeService(object(), store, bridge=broken)
    service.start()
    assert wait_for(lambda: service.status()["state"] == "error")
    assert "integrity" in service.status()["detail"]


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


# ── robot CPU without AES (Raspberry Pi 4): MediaPipe 1.x SIGILLs, which can't be caught ──
PI4_CPUINFO = "processor : 0\nFeatures\t: fp asimd evtstrm crc32 cpuid\nCPU part\t: 0xd08\n"
X86_CPUINFO = "processor : 0\nflags\t\t: fpu sse2 aes avx2\n"


def test_vision_is_refused_on_arm_without_aes(monkeypatch):
    from medcare_reachy import platform_check
    ok, reason = platform_check.vision_supported(machine="aarch64", cpuinfo=PI4_CPUINFO, mediapipe_present=True)
    assert not ok and "AES" in reason


def test_vision_is_allowed_on_x86_and_arm_with_aes():
    from medcare_reachy import platform_check
    assert platform_check.vision_supported(machine="x86_64", cpuinfo=X86_CPUINFO, mediapipe_present=True)[0]
    arm_aes = PI4_CPUINFO.replace("crc32", "aes pmull sha1 sha2 crc32")
    assert platform_check.vision_supported(machine="aarch64", cpuinfo=arm_aes, mediapipe_present=True)[0]


def test_missing_mediapipe_is_reported():
    from medcare_reachy import platform_check
    ok, reason = platform_check.vision_supported(machine="x86_64", cpuinfo=X86_CPUINFO, mediapipe_present=False)
    assert not ok and "MediaPipe" in reason


def test_unsupported_platform_stops_the_bridge_with_a_clear_status_instead_of_crashing(tmp_path, monkeypatch):
    import medcare_reachy.service as service_module
    store = SettingsStore(tmp_path / "s.json")
    store.save(GOOD)
    monkeypatch.setattr(service_module.platform_check, "vision_supported",
                        lambda: (False, "This robot's processor has no AES instructions."))
    imported = []
    monkeypatch.setattr(service_module, "_import_vision", lambda: imported.append(True))
    service = BridgeService(object(), store)          # real default_bridge
    service.start()
    assert wait_for(lambda: service.status()["state"] == "error")
    assert "AES" in service.status()["detail"]
    assert imported == []                             # MediaPipe was never loaded
