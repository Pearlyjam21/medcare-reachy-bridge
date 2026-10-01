"""The vision models, shipped inside this app and checked by SHA-256 before use.

Six landmark models: the browser worker's MediaPipe Tasks models (face_landmarker, hand_landmarker,
pose_landmarker_lite; Apache-2.0) converted to ONNX, plus OpenCV Zoo's ONNX export of the same MediaPipe pose
detector (its TFLite original uses sparse weights the converter can't read). And the MedAiCarePlus server's seed-43
emotion model, so emotion is scored on the robot. tools/MODELS.md records how they were made. Nothing is downloaded
at run time; a damaged or altered install stops with a clear message.
"""

import hashlib
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent / "vision_models"
MODELS = {
    "face_detector.onnx": "d97a63b82d1751e6dd219b842f55acc28c9656865611caeb88d7bc48f49919c4",
    "face_landmarks_detector.onnx": "1d97e6326845df6c239b5c2662c9bec62eb761839def949e1aa0fccb8ed1b47a",
    "hand_detector.onnx": "61c78d31e5c6c37a0ebde4c2f382db50d5116bfae0bd4be629f8a7d2aafc0f40",
    "hand_landmarks_detector.onnx": "5ed334609f59b97a43eed7d993ce305bb49134dc4c8918ecb7471099fc51071a",
    "pose_detector.onnx": "47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f",
    "pose_landmarks_detector.onnx": "cd7c2b67e1275239b80b17eab9cd861422ef6ac291370baf512c3022409a248d",
    "emotion_seed43.onnx": "0caaedf04b60d1c95d89ee2162c8bf207ccd669b88865f155b17987cc15ffbad",
    "emotion_seed43_metadata.json": "a5eb931d1cbb9ed862642bf73713954d318791bd0982020eeb04c051f96be739",
}
EMOTION_MODEL = "emotion_seed43.onnx"


class ModelError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_models(directory: Path = DEFAULT_DIR, expected: dict[str, str] = MODELS) -> Path:
    directory = Path(directory)
    for name, digest in expected.items():
        path = directory / name
        if not path.is_file():
            raise ModelError(f"Vision model {name} is missing; reinstall the app from the dashboard.")
        if _sha256(path) != digest:
            raise ModelError(f"Vision model {name} failed its integrity check; reinstall the app from the dashboard.")
    return directory
