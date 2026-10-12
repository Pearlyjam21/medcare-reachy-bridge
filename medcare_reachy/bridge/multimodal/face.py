"""Face branch on the server-verified face, with the face model this app already ships.

`emotion_seed43.onnx` is byte-for-byte the Space's `face/fer_mobilenetv3_fp32.onnx` (same FER+ MobileNetV3,
same `ferplus-gray-rgb-imagenet-v1` preprocessing), so the bundled, SHA-256-checked copy is used and nothing is
downloaded for this branch. The Space detects the largest face with YuNet; here the face is the one the server
confirmed is the patient (`target_box`), and, as before, it is never scored while a hand covers the mouth.

Two views of one inference: `raw` (softmax at temperature 1, what the server's own model reports and what the
server's reliability thresholds were set on) and the calibrated probabilities used for fusion (temperature from
the model repo's config.json).
"""

from __future__ import annotations

import time

import numpy as np

from medcare_reachy.bridge.emotion import EmotionEngine, mouth_covered, target_face

from .fusion import OK, BranchResult
from .labels import softmax, to_canonical


class FaceBranch:
    def __init__(self, engine: EmotionEngine, temperature: float = 1.0) -> None:
        self.engine, self.temperature = engine, temperature

    def analyze(self, frame_bgr: np.ndarray, packet: dict | None, target_box) -> BranchResult:
        """Score the verified face. `packet` (the robot's own landmarks) gates on a covered mouth; without one
        (frames are streamed, landmarks live on the server) the server applies that gate itself."""
        if frame_bgr is None:
            return BranchResult(status="no_camera")
        if not target_box:
            return BranchResult(status="no_face")
        index = None
        if packet is not None:
            index = target_face(packet, target_box)
            if index is None:
                return BranchResult(status="no_face")
            if mouth_covered(packet["faces"][index], packet["hands"], target_box):
                return BranchResult(status="mouth_covered", detail={"face_index": index})
        t0 = time.perf_counter()
        logits = to_canonical("face", self.engine.logits(frame_bgr, target_box))
        return BranchResult(softmax(logits, self.temperature), OK,
                            {"face_index": index, "raw": softmax(logits),
                             "latency_ms": (time.perf_counter() - t0) * 1000.0})
