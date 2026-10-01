"""Time the landmark models on this machine and estimate the landmark frame rate it can sustain.

    python -m medcare_reachy.bridge.vision_bench [models_dir]

The server only auto-records from a session that keeps >= 12 landmark frames per second (monitor_service
FPS_MIN), so this tells you, before a real reminder, whether this computer is fast enough.
"""

import statistics
import sys
import time
from pathlib import Path

import numpy as np

from medcare_reachy.bridge import vision

SERVER_FPS_MIN = 12.0


def time_model(model: "vision.OnnxModel", runs: int) -> float:
    shape = [d if isinstance(d, int) else 1 for d in model.session.get_inputs()[0].shape]
    if model.channels_first:
        shape = [shape[0], shape[2], shape[3], shape[1]]
    tensor = np.random.default_rng(0).random(shape, dtype=np.float32)
    for _ in range(3):
        model.run(tensor)
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        model.run(tensor)
        samples.append(time.perf_counter() - start)
    return 1000 * statistics.median(samples)


def main(models_dir: Path, runs: int = 20) -> dict:
    ms = {name.removesuffix(".onnx"): time_model(vision.OnnxModel(Path(models_dir) / name), runs)
          for name in vision.MODEL_FILES}
    for name, value in ms.items():
        print(f"{name:26s} {value:7.1f} ms")
    every = vision.DETECT_EVERY
    # One person, one hand in view; the three landmarkers run on their own threads, so the slowest bounds the rate.
    per_frame = {
        "face": ms["face_landmarks_detector"] + ms["face_detector"] / every,
        "hand": ms["hand_landmarks_detector"] + ms["hand_detector"] / every,
        "pose": (ms["pose_landmarks_detector"] + ms["pose_detector"] / every) / vision.POSE_EVERY,
    }
    slowest = max(per_frame, key=lambda name: per_frame[name])
    fps = 1000 / per_frame[slowest]
    print(f"\nper frame, one person + one hand: " + ", ".join(f"{k} {v:.1f} ms" for k, v in per_frame.items()))
    print(f"expected landmark rate: about {fps:.1f} fps (limited by {slowest}; camera capture is 15 fps)")
    if fps >= SERVER_FPS_MIN:
        print(f"OK: at or above the server's {SERVER_FPS_MIN:.0f} fps minimum for automatic recording.")
    else:
        print(f"BELOW the server's {SERVER_FPS_MIN:.0f} fps minimum: sessions will be marked degraded and every dose "
              "will go to a caregiver to confirm on LINE.")
    return {"ms": ms, "fps": fps}


if __name__ == "__main__":
    from medcare_reachy.bridge.config import _REPO_MODELS

    main(Path(sys.argv[1]) if len(sys.argv) > 1 else _REPO_MODELS)
