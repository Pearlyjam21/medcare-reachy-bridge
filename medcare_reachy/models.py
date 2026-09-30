"""MediaPipe model files, downloaded once from Google's official storage and verified by SHA-256.

The hashes are the exact files the web app's browser worker and the bridge tests use, so the robot runs
byte-identical models.
"""

import hashlib
import urllib.request
from pathlib import Path

_BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "face_landmarker.task": (f"{_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                             "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
    "hand_landmarker.task": (f"{_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                             "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"),
    "pose_landmarker_lite.task": (f"{_BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                                  "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"),
}
DEFAULT_DIR = Path.home() / ".medcare_reachy" / "models"


class ModelError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, open(destination, "wb") as out:
        while block := response.read(1 << 20):
            out.write(block)


def ensure_models(directory: Path = DEFAULT_DIR, download=_download) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in MODELS.items():
        target = directory / name
        if target.exists() and _sha256(target) == expected:
            continue
        partial = target.with_suffix(".part")
        try:
            download(url, partial)
        except OSError as exc:
            partial.unlink(missing_ok=True)
            raise ModelError(f"Could not download {name}: {exc}") from exc
        actual = _sha256(partial)
        if actual != expected:
            partial.unlink(missing_ok=True)
            raise ModelError(f"{name} failed its integrity check (got {actual[:12]}…)")
        partial.replace(target)
    return directory
