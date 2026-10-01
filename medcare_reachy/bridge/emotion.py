"""The server's seed-43 emotion model, run on the robot so its camera frames never need scoring on the server.

Same model file and the same preprocessing as app/services/emotion_service.py (grayscale replicated to RGB,
112x112 bilinear stretch, ImageNet normalisation, a 10% padded face crop), without OpenCV.
"""

import json
from pathlib import Path

import numpy as np

LABELS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")   # = emotion_service.LABELS
PREPROCESS = "ferplus-gray-rgb-imagenet-v1"
MODEL_FILE = "emotion_seed43.onnx"
MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
MOUTH = (5, 8)        # FACE_INDICES positions of the mouth corners (61, 291)
MIN_TARGET_OVERLAP = 0.20


def overlap(a, b) -> float:
    """IoU of two normalised [x, y, w, h] boxes, as monitor_service.overlap."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union > 0 else 0.0


def target_face(packet: dict, target_box) -> int | None:
    """Index of the packet face the server verified as the patient, or None."""
    if not target_box:
        return None
    scored = sorted(((overlap(face["box"], target_box), i) for i, face in enumerate(packet["faces"])), reverse=True)
    return scored[0][1] if scored and scored[0][0] >= MIN_TARGET_OVERLAP else None


def mouth_covered(face: dict, hands: list, target_box) -> bool:
    """Same rule as the server: any hand's wrist near the mouth (scaled by the face width) hides the expression."""
    points = [face["points"][i] for i in MOUTH if len(face["points"]) > i]
    if not points:
        return True
    mouth = (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
    reach = max(.04, target_box[2] * .6)
    return any(((hand[0][0] - mouth[0]) ** 2 + (hand[0][1] - mouth[1]) ** 2) ** .5 < reach for hand in hands if hand)


class EmotionEngine:
    def __init__(self, model_path: Path):
        import onnxruntime as ort

        model_path = Path(model_path)
        metadata = json.loads(model_path.with_name(model_path.stem + "_metadata.json").read_text(encoding="utf-8"))
        if tuple(metadata["labels"]) != LABELS or metadata["preprocess_version"] != PREPROCESS:
            raise ValueError("Unexpected emotion model metadata")
        self.size = int(metadata["input_size"])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def prepare(self, frame_bgr: np.ndarray, box) -> np.ndarray:
        """Normalised [x, y, w, h] face box -> (1, 3, size, size) tensor."""
        from PIL import Image

        h, w = frame_bgr.shape[:2]
        x, y, bw, bh = box[0] * w, box[1] * h, box[2] * w, box[3] * h
        left, top = max(0, int(x - bw * .10)), max(0, int(y - bh * .10))
        right, bottom = min(w, int(x + bw * 1.10)), min(h, int(y + bh * 1.10))
        face = frame_bgr[top:bottom, left:right]
        if face.size == 0:
            raise ValueError("Face crop is empty")
        gray = Image.fromarray(np.ascontiguousarray(face[:, :, ::-1])).convert("L")
        image = gray.convert("RGB").resize((self.size, self.size), resample=Image.Resampling.BILINEAR)
        values = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
        return ((values - MEAN) / STD)[None, ...]

    def predict(self, frame_bgr: np.ndarray, box) -> dict[str, float]:
        logits = np.asarray(self.session.run(None, {self.input_name: self.prepare(frame_bgr, box)})[0][0],
                            dtype=np.float64)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return {name: float(probabilities[i]) for i, name in enumerate(LABELS)}

    def score(self, frame_bgr: np.ndarray, packet: dict, target_box) -> dict | None:
        """The emotion report for this packet's verified face, or None when there's nothing fair to score.

        Like the server, the crop is the identity box the server verified (target_box), not the landmark box.
        """
        index = target_face(packet, target_box)
        if index is None or mouth_covered(packet["faces"][index], packet["hands"], target_box):
            return None
        return {"face_index": index, "probabilities": self.predict(frame_bgr, target_box)}
