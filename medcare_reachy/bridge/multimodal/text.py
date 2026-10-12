"""Traditional Chinese text-emotion branch (bert-base-chinese fine-tune, dynamic int8 ONNX).

Uses the `tokenizers` fast tokenizer (no transformers/torch). ASR output in Simplified Chinese is
converted to Traditional with chinese-converter, the converter used for the training fear data.
The model is Chinese-only: English or other transcripts are reported as unsupported, not scored.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np

from .fusion import OK, BranchResult
from .labels import softmax, to_canonical
from .runtime import session_options

MAX_LENGTH = 128
CHINESE_LANGS = {"zh", "yue"}
_HAN = re.compile(r"[\u3400-\u9fff]")  # CJK Unified Ideographs + Extension A


class TextBranch:
    def __init__(self, model_path: str | Path, tokenizer_path: str | Path, temperature: float = 1.0,
                 threads: int = 1) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.session = ort.InferenceSession(str(model_path), session_options(threads), providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(MAX_LENGTH)
        self.temperature = temperature

    def logits(self, text: str) -> np.ndarray:
        enc = self.tokenizer.encode(text)
        feeds = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([enc.type_ids], dtype=np.int64),
        }
        return self.session.run(None, feeds)[0][0]

    def classify(self, text: str, lang: str | None) -> tuple[BranchResult, str]:
        """Score one transcribed sentence. Returns (result, text in Traditional characters)."""
        text = text.strip()
        if not text:
            return BranchResult(status="no_transcript"), text
        if (lang is not None and lang not in CHINESE_LANGS) or not _HAN.search(text):
            return BranchResult(status="unsupported_language", detail={"lang": lang}), text
        import chinese_converter

        text = chinese_converter.to_traditional(text)
        t0 = time.perf_counter()
        probs = softmax(to_canonical("text", self.logits(text)), self.temperature)
        return BranchResult(probs, OK, {"latency_ms": (time.perf_counter() - t0) * 1000.0}), text
