"""App settings entered on the robot's settings page, kept in a private JSON file on the robot."""

import json
import os
from pathlib import Path
from urllib.parse import urlparse

from medcare_reachy.bridge.config import LANGUAGES

DEFAULT_PATH = Path.home() / ".medcare_reachy" / "settings.json"
DEFAULTS = {"app_url": "", "device_token": "", "language": "zh-TW", "capture_fps": 10.0}
MAX_FPS = 15.0   # the intake detector is calibrated at ~15 fps; the robot's local camera feed tops out near 10


class SettingsError(ValueError):
    pass


class SettingsStore:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)

    def load(self) -> dict:
        try:
            stored = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            stored = {}
        return {**DEFAULTS, **{k: v for k, v in stored.items() if k in DEFAULTS}}

    def save(self, update: dict) -> dict:
        settings = self.load()
        for key in ("app_url", "language", "capture_fps"):
            if key in update and update[key] is not None:
                settings[key] = update[key]
        # The key is write-only: an empty field on the form means "keep the current key".
        token = (update.get("device_token") or "").strip()
        if token:
            settings["device_token"] = token
        settings = validate(settings)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(settings), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)   # the device key acts as this robot; keep it private to the app user
        except OSError:
            pass
        tmp.replace(self.path)
        return settings

    @staticmethod
    def configured(settings: dict) -> bool:
        return bool(settings["app_url"]) and settings["device_token"].startswith("rdv1.")


def validate(settings: dict) -> dict:
    url = str(settings["app_url"]).strip().rstrip("/")
    if url:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.path not in ("", "/"):
            raise SettingsError("Server address must look like http://192.168.1.20:8001")
    token = str(settings["device_token"]).strip()
    if token and not token.startswith("rdv1."):
        raise SettingsError("The robot key starts with rdv1. (Settings → Reachy robot → Pair)")
    if settings["language"] not in LANGUAGES:
        raise SettingsError(f"Language must be one of {', '.join(LANGUAGES)}")
    try:
        fps = float(settings["capture_fps"])
    except (TypeError, ValueError) as exc:
        raise SettingsError("Camera rate must be a number") from exc
    if not 1.0 <= fps <= MAX_FPS:
        raise SettingsError(f"Camera rate must be between 1 and {MAX_FPS:g} fps")
    return {"app_url": url, "device_token": token, "language": settings["language"], "capture_fps": fps}


def public_view(settings: dict) -> dict:
    """Never send the key back to the browser; only whether one is set."""
    return {"app_url": settings["app_url"], "language": settings["language"],
            "capture_fps": settings["capture_fps"], "device_token_set": bool(settings["device_token"])}
