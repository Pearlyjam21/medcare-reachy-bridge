"""The bundled ONNX models + bridge graph code reproduce real MediaPipe Tasks landmarks.

reference.json was produced by the real MediaPipe Tasks (x86) on MediaPipe's own sample images
(spike/onnxconv/mp_reference.py); this test needs that folder, so it is skipped elsewhere.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from medcare_reachy import models
from medcare_reachy.bridge.vision import VisionEngine

FIXTURES = Path(r"D:\medcareai2_20260920\spike\onnxconv")
pytestmark = pytest.mark.skipif(not (FIXTURES / "reference.json").exists(), reason="needs the MediaPipe reference set")


def landmarks(image_name):
    from PIL import Image

    image = Image.open(FIXTURES / "img" / image_name).convert("RGB")
    engine = VisionEngine(models.DEFAULT_DIR, detect_every=1, pose_every=1)
    try:
        return {name: tracker.step(image) for name, tracker in engine.trackers.items()}, image.size
    finally:
        engine.close()


def best_error(ours, reference, size, points):
    ref = np.array(reference)[:, :2] * size
    return min(np.linalg.norm(o[list(points), :2] - ref[list(points)], axis=1).mean() for o in ours)


REFERENCE = json.loads((FIXTURES / "reference.json").read_text()) if (FIXTURES / "reference.json").exists() else {}


@pytest.mark.parametrize("image_name,kind,ref_key,points,limit_px", [
    ("portrait.jpg", "face", "faces", (1, 10, 13, 14, 33, 61, 152, 263, 291), 3.0),   # the server's 9 face points
    ("pointing_up.jpg", "hand", "hands", tuple(range(21)), 3.0),
    ("thumb_up.jpg", "hand", "hands", tuple(range(21)), 8.0),
    ("pose.jpg", "pose", "poses", (0, 11, 12, 15, 16), 15.0),                         # nose, shoulders, wrists
])
def test_landmarks_match_mediapipe(image_name, kind, ref_key, points, limit_px):
    got, size = landmarks(image_name)
    reference = REFERENCE[image_name][ref_key]
    assert got[kind] and len(reference) == 1
    assert best_error(got[kind], reference[0], size, points) < limit_px
