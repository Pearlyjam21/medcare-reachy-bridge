"""One fused emotion for the dose being watched, from the face, the voice and the words.

Face scores come from the camera loop (`score`, the same call MonitorStream made to the old face-only
EmotionEngine). Voice and words come from the microphone thread (`hear`): each speech segment the "I finished"
listener cuts and decodes is scored by the acoustic model, and its transcript by the Chinese text model. A voice
or text result stays current for VOICE_TTL seconds and is fused with every face score meanwhile; a branch that
has no fresh evidence is left out and the weights are renormalised (fusion.py), never counted as "neutral".

What goes to the server keeps its contract (`emotion_seed43` probabilities in the server's label order): the
fused probabilities when at least two modalities are present, otherwise the face model's own softmax, so a
dose watched in silence is reported exactly as before and the server's reliability thresholds still apply.

Between `begin` and `end` every fused score is kept as a sample for `summary()`, which the slot session files
with the dose's confirmation evidence.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from medcare_reachy.bridge.emotion import LABELS as SERVER_LABELS

from .face import FaceBranch
from .fusion import OK, BranchResult, fuse
from .labels import EMOTIONS

log = logging.getLogger(__name__)

VOICE_TTL = 10.0            # a voice/text score stays current this long after the sentence ended
MAX_BRANCH_FAILURES = 3     # a branch that raises this often in a row is switched off (logged once)
MAX_SAMPLES = 1000
# The Space's defaults (config.json "fusion"): face 0.5 / voice 0.3 / text 0.9, face temperature fitted on CREMA-D.
DEFAULT_WEIGHTS = {"speech": 0.3, "face": 0.5, "text": 0.9}
DEFAULT_TEMPERATURES = {"speech": 1.0, "face": 6.461479787306789, "text": 1.0}
CHINESE = ("zh-TW",)
_TO_SERVER = [EMOTIONS.index(name) for name in SERVER_LABELS]


def to_server(probs: np.ndarray) -> dict[str, float]:
    """EMOTIONS-ordered probabilities -> the server's {label: p} (emotion_service.LABELS order)."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return {name: float(probs[i]) for name, i in zip(SERVER_LABELS, _TO_SERVER)}


class MultimodalEmotion:
    def __init__(self, face: FaceBranch | None, speech=None, text=None, *, weights: dict | None = None,
                 language: str = "zh-TW", voice_ttl: float = VOICE_TTL, clock=time.monotonic) -> None:
        self.face, self.speech, self.text = face, speech, text
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.language, self.voice_ttl, self.clock = language, voice_ttl, clock
        self.lock = threading.Lock()
        self.active = False
        self.intk_id: int | None = None
        self._speech = BranchResult(status="silence")
        self._text = BranchResult(status="no_transcript")
        self._voice_at = float("-inf")
        self._samples: list = []            # (fused probs, used) while a dose is watched
        self._counts = {"face": 0, "speech": 0, "text": 0}
        self._failures = {"speech": 0, "text": 0}
        self._off: set = set()
        self.last: dict | None = None       # the last report, for the status page

    @property
    def branches(self) -> dict[str, bool]:
        return {"face": self.face is not None, "speech": self.speech is not None and "speech" not in self._off,
                "text": self.text is not None and "text" not in self._off}

    # ── the dose window (slot session) ──
    def begin(self, intk_id: int | None) -> None:
        with self.lock:
            self.active, self.intk_id = True, intk_id
            self._samples = []
            self._counts = {"face": 0, "speech": 0, "text": 0}

    def end(self) -> None:
        with self.lock:
            self.active = False

    def summary(self) -> dict:
        """What was heard and seen while the dose was watched: the average fused emotion and the sample counts."""
        with self.lock:
            samples, counts, intk_id = list(self._samples), dict(self._counts), self.intk_id
        out = {"intk_id": intk_id, "samples": len(samples), "modalities": counts, "branches": self.branches}
        if samples:
            mean = np.mean([probs for probs, _ in samples], axis=0)
            out["top"] = EMOTIONS[int(mean.argmax())]
            out["confidence"] = round(float(mean.max()), 3)
            out["probabilities"] = {name: round(float(p), 3) for name, p in zip(EMOTIONS, mean)}
            out["fused_samples"] = sum(1 for _, used in samples if len(used) > 1)
        return out

    # ── the microphone (DoneListener thread) ──
    def hear(self, segment: np.ndarray, began: float, ended: float, text: str | None) -> None:
        if not self.active:
            return
        speech = self._run("speech", lambda: self.speech.score(segment)[0]) if self.speech else None
        words = None
        if self.text is not None and text:
            lang = "zh" if self.language in CHINESE else self.language
            words = self._run("text", lambda: self.text.classify(text, lang)[0])
        with self.lock:
            if speech is not None:
                self._speech = speech
                self._counts["speech"] += speech.present
            if words is not None:
                self._text = words
                self._counts["text"] += words.present
            self._voice_at = self.clock()

    def _run(self, branch: str, call):
        if branch in self._off:
            return None
        try:
            result = call()
        except Exception as exc:
            self._failures[branch] += 1
            if self._failures[branch] >= MAX_BRANCH_FAILURES:
                self._off.add(branch)
                log.exception("%s emotion failed %d times in a row; that branch is off", branch, self._failures[branch])
            else:
                log.warning("%s emotion failed: %r", branch, exc)
            return None
        self._failures[branch] = 0
        return result

    # ── the camera (MonitorStream) ──
    def score(self, frame_bgr, packet: dict | None, target_box, now: float | None = None) -> dict | None:
        """The server's emotion report for this frame, or None when there is nothing fair to score."""
        if self.face is None:
            return None
        face = self.face.analyze(frame_bgr, packet, target_box)
        if not face.present:
            return None
        now = self.clock() if now is None else now
        with self.lock:
            fresh = now - self._voice_at <= self.voice_ttl
            branches = {"face": face,
                        "speech": self._speech if fresh else BranchResult(status="stale"),
                        "text": self._text if fresh else BranchResult(status="stale")}
            fused = fuse(branches, self.weights)
            probs = fused.probs if len(fused.used) > 1 else face.detail["raw"]
            report = {"face_index": face.detail["face_index"], "probabilities": to_server(probs)}
            if self.active:
                self._counts["face"] += 1
                self._samples.append((np.asarray(probs, dtype=np.float64), fused.used))
                del self._samples[:-MAX_SAMPLES]
            self.last = {**report, "used": dict(fused.used), "top": EMOTIONS[int(np.argmax(probs))]}
        return report


def load_multimodal(models_dir: Path, language: str = "zh-TW", *, download: bool = True,
                    repo_root: Path | None = None) -> MultimodalEmotion:
    """The bundled face model always; the voice and (for Chinese) text models from the Hub when they can be had.

    Fetching uses the Hugging Face login already on the machine (`hf auth login`); a missing login, no network or
    a missing package leaves those branches out with one log line, never stops the app.
    """
    from medcare_reachy import models
    from medcare_reachy.bridge.emotion import EmotionEngine

    models.verify_models(models_dir, {name: models.MODELS[name] for name in models.EMOTION_FILES})
    face = FaceBranch(EmotionEngine(Path(models_dir) / models.EMOTION_MODEL), DEFAULT_TEMPERATURES["face"])
    speech = text = None
    weights = dict(DEFAULT_WEIGHTS)
    try:
        from .models import fetch_models, load_config

        root = repo_root or (fetch_models() if download else None)
        if root is not None:
            cfg = load_config(root)
            weights = dict(cfg.get("fusion", {}).get("weights", weights))
            temperatures = {**DEFAULT_TEMPERATURES, **cfg.get("fusion", {}).get("temperatures", {})}
            face.temperature = float(temperatures["face"])
            files = cfg["files"]
            from .speech import SpeechBranch

            speech = SpeechBranch(root / files["speech"], float(temperatures["speech"]))
            if language in CHINESE:
                from .text import TextBranch

                text = TextBranch(root / files["text"], root / files["text_tokenizer"], float(temperatures["text"]))
    except Exception as exc:   # voice and words are extras; the face branch carries on
        log.warning("voice/text emotion models unavailable (%r); emotion runs on the face alone", exc)
    log.info("multimodal emotion: face%s%s", ", voice" if speech else "", ", words" if text else "")
    return MultimodalEmotion(face, speech, text, weights=weights, language=language)
