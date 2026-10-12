"""Torch-free acoustic emotion branch (compact ECAPA-TDNN student, SER v3).

The log-mel front end is the NumPy port verified against torchaudio in SER_Reachy_merli_v2
(MelSpectrogram n_fft=512, win=400 Hann, hop=160, 80 HTK mel bands 20-8000 Hz, power 2,
log clamp 1e-5, per-utterance mean/std normalisation). tests/test_speech.py re-checks it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .fusion import OK, BranchResult
from .labels import softmax, to_canonical
from .runtime import session_options

SAMPLE_RATE = 16000
WINDOW_S = 3.0      # sentences longer than MAX_SINGLE_S are scored as overlapping windows
HOP_S = 1.5
MAX_SINGLE_S = 4.0  # the student was trained on 1.5-4 s crops
MIN_S = 0.5


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


class LogMel:
    """NumPy port of utils.audio.LogMelExtractor from the SER training repo."""

    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=512, win_length=400, hop_length=160,
                 n_mels=80, f_min=20.0, f_max=8000.0) -> None:
        self.n_fft, self.hop = n_fft, hop_length
        # Periodic Hann window, zero-padded to n_fft and centred (as torch.stft does)
        win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win_length) / win_length)
        left = (n_fft - win_length) // 2
        self.window = np.zeros(n_fft)
        self.window[left : left + win_length] = win
        # HTK mel filterbank without normalisation (torchaudio.functional.melscale_fbanks)
        all_freqs = np.linspace(0, sample_rate // 2, n_fft // 2 + 1)
        f_pts = _mel_to_hz(np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2))
        f_diff = f_pts[1:] - f_pts[:-1]
        slopes = f_pts[None, :] - all_freqs[:, None]
        down = -slopes[:, :-2] / f_diff[:-1]
        up = slopes[:, 2:] / f_diff[1:]
        self.fbank = np.maximum(0.0, np.minimum(down, up))  # [n_freqs, n_mels]

    def __call__(self, wave: np.ndarray) -> np.ndarray:
        """wave: [T] float -> normalised log-mel [80, frames] float32."""
        x = np.pad(np.asarray(wave, dtype=np.float64), self.n_fft // 2, mode="reflect")
        n_frames = 1 + (len(x) - self.n_fft) // self.hop
        idx = np.arange(self.n_fft)[None, :] + self.hop * np.arange(n_frames)[:, None]
        spec = np.abs(np.fft.rfft(x[idx] * self.window, axis=1)) ** 2  # [frames, n_freqs]
        log_mel = np.log(np.maximum(spec @ self.fbank, 1e-5)).T  # [n_mels, frames]
        mean = log_mel.mean(axis=1, keepdims=True)
        std = log_mel.std(axis=1, ddof=1, keepdims=True) + 1e-5
        return ((log_mel - mean) / std).astype(np.float32)


class SpeechBranch:
    """Scores one whole sentence: log-mel + ONNX, averaged over windows for long sentences.

    Also returns the student's valence/arousal/dominance head (0-1 each).
    """

    def __init__(self, model_path: str | Path, temperature: float = 1.0, threads: int = 1) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), session_options(threads), providers=["CPUExecutionProvider"])
        self.temperature = temperature
        self.mel = LogMel()

    def logits(self, wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw branch-order emotion logits [7] and VAD [3] for one mono 16 kHz window."""
        logits, vad = self.session.run(None, {"log_mel": self.mel(wave)[None]})
        return logits[0], vad[0]

    def score(self, samples: np.ndarray) -> tuple[BranchResult, np.ndarray | None]:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(samples) < MIN_S * SAMPLE_RATE:
            return BranchResult(status="too_short"), None
        t0 = time.perf_counter()
        if len(samples) <= MAX_SINGLE_S * SAMPLE_RATE:
            windows = [samples]
        else:
            win, hop = int(WINDOW_S * SAMPLE_RATE), int(HOP_S * SAMPLE_RATE)
            starts = list(range(0, len(samples) - win + 1, hop))
            if starts[-1] + win < len(samples):
                starts.append(len(samples) - win)
            windows = [samples[i:i + win] for i in starts]
        probs, vads = [], []
        for w in windows:
            logits, vad = self.logits(w)
            probs.append(softmax(to_canonical("speech", logits), self.temperature))
            vads.append(vad)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return (BranchResult(np.mean(probs, axis=0), OK, {"latency_ms": latency_ms, "windows": len(windows)}),
                np.mean(vads, axis=0))
