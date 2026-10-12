"""Reachy's voice for text that isn't a prerecorded clip (check-in replies), with Matcha-TTS on the robot.

The server sends replies already converted for this voice (`speech_text`, Simplified characters for the Mandarin
zh-baker voice). Audio is synthesised into a temporary WAV, played, and deleted.
"""

import logging
import queue
import os
import re
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

TTS_DIR = Path.home() / ".medcare_reachy" / "models" / "tts"
MATCHA = "matcha-icefall-zh-baker"
TTS_FILES = (f"{MATCHA}/model-steps-3.onnx", f"{MATCHA}/lexicon.txt", f"{MATCHA}/tokens.txt", "vocos-22khz-univ.onnx")
SPEED = 0.85   # a little slower than default for older listeners

_engine = None
_engine_lock = threading.Lock()


def _matcha(models_dir: Path):
    import sherpa_onnx

    matcha = models_dir / MATCHA
    fsts = ",".join(str(matcha / f) for f in ("phone.fst", "date.fst", "number.fst") if (matcha / f).is_file())
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=str(matcha / "model-steps-3.onnx"), vocoder=str(models_dir / "vocos-22khz-univ.onnx"),
                lexicon=str(matcha / "lexicon.txt"), tokens=str(matcha / "tokens.txt")),
            num_threads=2),
        rule_fsts=fsts, max_num_sentences=2)
    return sherpa_onnx.OfflineTts(config)


MAX_CHUNK = 16       # comma clauses are joined up to this many characters, so the first chunk is ready in ~2 s
MIN_FIRST_CJK = 6    # an opening comma clause with this many Chinese characters is spoken on its own

_TERMINALS = "\u3002\uff01\uff1f!?\uff1b;"
_CLOSERS = "\u300d\u300f\uff09)\u300b\"'\u201d\u2019"
# After \uff0c\u3001 and after a comma, except inside a number such as 1,000.
_CLAUSE_BREAK = re.compile(r"(?<=[\uff0c\u3001])|(?<=\D,)|(?<=,)(?!\d)")
_CJK = re.compile(r"[\u3400-\u9fff]")


def _sentences(text: str) -> list[str]:
    """Split after \u3002\uff01\uff1f!?\uff1b; and after a full stop followed by a space or the end.

    A run such as \uff01\uff01 stays together with any closing quote or bracket after it, and a full stop inside 25.5 or
    ... is not an end.
    """
    out: list[str] = []
    start = i = 0
    while i < len(text):
        char = text[i]
        i += 1
        if char in _TERMINALS or (char == "." and (i == len(text) or text[i].isspace() or text[i] in _CLOSERS)):
            while i < len(text) and (text[i] in _TERMINALS or text[i] in _CLOSERS):
                i += 1
            out.append(text[start:i])
            start = i
    out.append(text[start:])
    return [sentence.strip() for sentence in out if sentence.strip()]


def chunks(text: str | None) -> list[str]:
    """Split a reply into sentences, and long sentences again at commas, keeping the punctuation.

    A clause without a comma is never cut, even when it is longer than MAX_CHUNK. Each chunk is spoken with its own
    falling tone and a pause, and a cut inside a clause splits words and numbers: the spoken dose refusals would
    become \u660e\u5929\u65e9|\u4e0a6\u70b9 and \u5403\u6ee1 4|\u6b21\u4e86, and \u8981\u4e0d\u8981 would become \u8981|\u4e0d\u8981 ("don't").
    The first comma clause, when it has at least MIN_FIRST_CJK Chinese characters, is a chunk of its own, so
    Reachy starts speaking while the rest is synthesised.
    """
    out: list[str] = []
    for sentence in _sentences((text or "").strip()):
        clauses = [clause for clause in _CLAUSE_BREAK.split(sentence) if clause]
        if len(clauses) > 1 and len(_CJK.findall(clauses[0])) >= MIN_FIRST_CJK:
            out.append(clauses[0].strip())
            clauses = clauses[1:]
        elif len(sentence) <= MAX_CHUNK:
            out.append(sentence)
            continue
        current = ""
        for clause in clauses:
            if current and len(current) + len(clause) > MAX_CHUNK:
                out.append(current.strip())
                current = clause
            else:
                current += clause
        if current.strip():
            out.append(current.strip())
    return out


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())


def _ms(seconds: float) -> int:
    return max(0, int(round(seconds * 1000)))


class Speaker:
    def __init__(self, robot, models_dir: Path = TTS_DIR, *, load=_matcha, clock=time.monotonic):
        self.robot, self.models_dir, self._load, self.clock = robot, Path(models_dir), load, clock
        missing = [name for name in TTS_FILES if not (self.models_dir / name).is_file()]
        self.available = not missing
        if missing:
            log.warning("check-in voice is off: missing %s in %s", ", ".join(missing), self.models_dir)

    def _tts(self):
        global _engine
        with _engine_lock:   # one engine per process (~12 s to load on the Pi); reused across bridge restarts
            if _engine is None:
                _engine = self._load(self.models_dir)
            return _engine

    def preload(self) -> None:
        """Load the voice in the background so the first reply isn't delayed."""
        if self.available:
            threading.Thread(target=self._safe_load, name="medcare-tts-load", daemon=True).start()

    def _safe_load(self) -> None:
        try:
            self._tts()
        except Exception:
            log.exception("check-in voice failed to load")
            self.available = False

    def say(self, text: str, stats: dict | None = None, on_audio=None) -> bool:
        """Speak `text` (blocking until playback ends). False when there's no voice or nothing to say.

        Synthesis on the Pi runs at roughly real time, so a whole reply would keep the patient waiting as long
        as it lasts. Instead each chunk is played while the next one is being synthesised.

        `stats`, when given, receives the timings in milliseconds from the start of this call: until the first
        chunk is handed to the speaker (tts_first_audio_ms) and until playback ends (tts_total_ms), plus the
        number of chunks played (tts_chunks) and the total synthesis time (tts_synth_ms).
        `on_audio()` is called once, just before the first chunk is handed to the speaker.
        """
        pieces = chunks(text)
        if not pieces or not self.available:
            return False
        started = self.clock()
        tts = self._tts()
        ready: queue.Queue = queue.Queue(maxsize=2)
        synth: list[float] = []

        def produce() -> None:
            try:
                for piece in pieces:
                    began = self.clock()
                    audio = tts.generate(piece, sid=0, speed=SPEED)
                    synth.append(self.clock() - began)
                    ready.put(audio)
            except Exception:
                log.exception("speech synthesis failed")
            finally:
                ready.put(None)

        threading.Thread(target=produce, name="medcare-tts", daemon=True).start()
        played = False
        first_audio: float | None = None
        handed = 0
        while (audio := ready.get()) is not None:
            if len(audio.samples) == 0:
                continue
            handle, name = tempfile.mkstemp(prefix="medcare-say-", suffix=".wav")
            os.close(handle)
            path = Path(name)
            try:
                write_wav(path, audio.samples, audio.sample_rate)
                if first_audio is None:
                    first_audio = self.clock() - started
                    if on_audio is not None:
                        on_audio()
                handed += 1
                played = bool(self.robot.play_clip(path)) or played
            finally:
                path.unlink(missing_ok=True)
        if stats is not None:
            stats.update({"tts_total_ms": _ms(self.clock() - started), "tts_chunks": handed,
                          "tts_synth_ms": _ms(sum(synth))})   # the producer is done: it sent the final None
            if first_audio is not None:
                stats["tts_first_audio_ms"] = _ms(first_audio)
        return played

    def synth(self, text: str) -> tuple[np.ndarray, int] | None:
        """`text` in Reachy's voice without playing it: (float32 mono samples, their rate). The live chat
        (bridge/live.py) owns the speaker and queues the samples itself; it calls this for one chunk at a time, so
        its first sound waits for one chunk only. Blocks for the synthesis (about real time on the Pi). None when there
        is no voice, nothing to say, or no sound came out."""
        pieces = chunks(text)
        if not pieces or not self.available:
            return None
        tts = self._tts()
        parts, rate = [], 0
        for piece in pieces:
            audio = tts.generate(piece, sid=0, speed=SPEED)
            if len(audio.samples):
                parts.append(np.asarray(audio.samples, dtype=np.float32))
                rate = int(audio.sample_rate)
        if not parts:
            return None
        return np.concatenate(parts), rate
