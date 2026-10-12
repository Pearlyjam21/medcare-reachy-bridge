"""Calibrated late fusion: weighted log-linear pooling over the branches that are present.

A branch that has no evidence (no face, silence, no transcript, unsupported language, stale
result) is left out and the remaining weights are renormalised. Absence is never encoded as an
observed "neutral".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .labels import EMOTIONS

OK = "ok"


@dataclass
class BranchResult:
    """Calibrated probabilities in EMOTIONS order, or None with a status explaining why."""

    probs: np.ndarray | None = None
    status: str = "waiting"
    detail: dict = field(default_factory=dict)

    @property
    def present(self) -> bool:
        return self.status == OK and self.probs is not None


@dataclass
class FusedResult:
    probs: np.ndarray | None
    used: dict[str, float]  # branch -> normalised weight actually applied

    @property
    def top(self) -> str | None:
        return None if self.probs is None else EMOTIONS[int(self.probs.argmax())]

    @property
    def confidence(self) -> float:
        return 0.0 if self.probs is None else float(self.probs.max())


def fuse(results: dict[str, BranchResult], weights: dict[str, float], eps: float = 1e-6) -> FusedResult:
    present = {b: r.probs for b, r in results.items()
               if r.present and r.probs is not None and weights.get(b, 0.0) > 0.0}
    if not present:
        return FusedResult(None, {})
    total = sum(weights[b] for b in present)
    used = {b: weights[b] / total for b in present}
    log_p = np.zeros(len(EMOTIONS))
    for b, w in used.items():
        log_p += w * np.log(np.clip(present[b], eps, 1.0))
    p = np.exp(log_p - log_p.max())
    return FusedResult(p / p.sum(), used)
