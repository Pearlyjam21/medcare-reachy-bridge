"""One label order for fusion, and the index maps from each branch's own training order.

Each branch keeps the taxonomy it was trained with; logits are reordered into EMOTIONS before
calibration and fusion. EMOTIONS[i] == BRANCH_LABELS[b][TO_CANONICAL[b][i]].
"""

from __future__ import annotations

import numpy as np

EMOTIONS = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]

BRANCH_LABELS = {
    # SER ECAPA student (speech/v3/distill_config_v3.yaml)
    "speech": ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"],
    # newEmotion FER+ MobileNetV3 (model_fp32_metadata.json)
    "face": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
    # Traditional Chinese BERT (config.json id2label)
    "text": ["happy", "sad", "angry", "fear", "disgust", "neutral", "surprise"],
}

TO_CANONICAL = {b: np.array([labels.index(e) for e in EMOTIONS]) for b, labels in BRANCH_LABELS.items()}


def to_canonical(branch: str, logits: np.ndarray) -> np.ndarray:
    """Reorder a branch's logits (last axis) into EMOTIONS order."""
    return np.asarray(logits, dtype=np.float64)[..., TO_CANONICAL[branch]]


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
