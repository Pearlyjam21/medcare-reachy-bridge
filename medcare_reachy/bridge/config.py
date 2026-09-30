"""Bridge settings from the environment."""

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_MODELS = PACKAGE_DIR.parent / "frontend_source" / "public" / "models"

LANGUAGES = ("zh-TW", "en")
BACKENDS = ("reachy", "video")


@dataclass(frozen=True)
class Settings:
    app_url: str
    device_token: str
    robot_host: str
    robot_backend: str
    video_path: str
    language: str
    models_dir: Path
    clips_dir: Path

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        env = os.environ if env is None else env
        settings = cls(
            app_url=env.get("APP_INTERNAL_URL", "http://localhost:8001").rstrip("/"),
            device_token=env.get("REACHY_DEVICE_TOKEN", "").strip(),
            robot_host=env.get("REACHY_ROBOT_HOST", "").strip(),
            robot_backend=env.get("ROBOT_BACKEND", "reachy").strip().lower(),
            video_path=env.get("VIDEO_PATH", "").strip(),
            language=env.get("LANGUAGE", "zh-TW").strip(),
            models_dir=Path(env.get("MODELS_DIR") or _REPO_MODELS),
            clips_dir=Path(env.get("CLIPS_DIR") or PACKAGE_DIR / "clips"),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.device_token.startswith("rdv1."):
            raise ValueError("REACHY_DEVICE_TOKEN must be the rdv1.… token shown once at pairing")
        if self.robot_backend not in BACKENDS:
            raise ValueError(f"ROBOT_BACKEND must be one of {BACKENDS}")
        if self.robot_backend == "reachy" and not self.robot_host:
            raise ValueError("REACHY_ROBOT_HOST is required with ROBOT_BACKEND=reachy")
        if self.robot_backend == "video" and not self.video_path:
            raise ValueError("VIDEO_PATH is required with ROBOT_BACKEND=video")
        if self.language not in LANGUAGES:
            raise ValueError(f"LANGUAGE must be one of {LANGUAGES}")
