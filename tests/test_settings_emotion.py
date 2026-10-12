"""The multimodal emotion settings: on by default, computed on the robot with the laptop as fallback."""

import pytest

from medcare_reachy.settings_store import SettingsError, SettingsStore, public_view

GOOD = {"app_url": "http://192.168.1.20:8001", "device_token": "rdv1.abc", "language": "zh-TW", "capture_fps": 15.0}


def test_defaults_and_choices(tmp_path):
    store = SettingsStore(tmp_path / "s.json")
    saved = store.save(GOOD)
    assert saved["multimodal_emotion"] is True and saved["emotion_compute"] == "auto"
    assert public_view(store.load())["emotion_compute"] == "auto"
    saved = store.save({"emotion_compute": "Laptop", "multimodal_emotion": False})
    assert saved["emotion_compute"] == "laptop" and saved["multimodal_emotion"] is False
    assert public_view(store.load())["multimodal_emotion"] is False
    assert public_view(GOOD)["emotion_compute"] == "auto"     # settings saved before 0.7.0


@pytest.mark.parametrize("change", [{"emotion_compute": "cloud"}, {"emotion_compute": ""},
                                    {"multimodal_emotion": "yes"}])
def test_bad_values_are_rejected(tmp_path, change):
    with pytest.raises(SettingsError):
        SettingsStore(tmp_path / "s.json").save({**GOOD, **change})
