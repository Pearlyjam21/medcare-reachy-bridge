"""Prerecorded prompt clips: manifest lookup and playback through the robot."""

import json
import logging
import random
from pathlib import Path

from medcare_reachy.bridge.media import wav_seconds

log = logging.getLogger(__name__)

MANIFEST = Path(__file__).resolve().parent / "clips" / "manifest.json"
GENERIC_MED_PROMPT = "med_prompt_generic"
THINKING = "thinking"   # the check-in's thinking phrases (manifest "variants"), said once the patient's turn is in


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class ClipPlayer:
    """Plays `<clips_dir>/<language>/<clip_id>.wav`; a missing file is logged once and counted, never fatal.

    A group of variants in the manifest (`"variants": {"thinking": [...]}`: the check-in's thinking phrases) plays
    one of its clips each time, so Reachy doesn't sound canned: every variant once in a shuffled order, then again,
    never the same one twice in a row. Variants without a WAV are skipped; with none, nothing plays. `played` names
    the clips played.
    """

    def __init__(self, robot, clips_dir: Path, language: str, manifest: dict | None = None,
                 rng: random.Random | None = None):
        self.robot = robot
        self.dir = Path(clips_dir) / language
        self.manifest = manifest if manifest is not None else load_manifest()
        self.missing: set[str] = set()
        self.played: list[str] = []
        self._random = rng or random.Random()
        self._round: dict[str, list[str]] = {}   # per clip: the variants still to play in this round
        self._last: dict[str, str] = {}          # per clip: the variant played last

    def path(self, clip_id: str) -> Path:
        return self.dir / f"{clip_id}.wav"

    def _file(self, clip_id: str) -> Path | None:
        if clip_id not in self.manifest["clips"] and not clip_id.startswith("med_"):
            raise KeyError(f"unknown clip {clip_id}")
        path = self.path(clip_id)
        if not path.is_file():
            if clip_id not in self.missing:
                log.warning("clip %s missing at %s", clip_id, path)
            self.missing.add(clip_id)
            return None
        return path

    def variant(self, clip_id: str) -> str | None:
        """The clip to play for `clip_id`: a group's next variant with a WAV (None when none has one), or the clip."""
        variants = self.manifest.get("variants", {}).get(clip_id)
        if variants is None:
            return clip_id
        present = [variant for variant in variants if self._file(variant) is not None]
        if not present:
            return None
        turn = [variant for variant in self._round.get(clip_id, []) if variant in present]
        if not turn:
            turn = list(present)
            self._random.shuffle(turn)
            if len(turn) > 1 and turn[0] == self._last.get(clip_id):
                turn.append(turn.pop(0))
        choice = turn.pop(0)
        self._round[clip_id], self._last[clip_id] = turn, choice
        return choice

    def play(self, clip_id: str) -> bool:
        clip_id = self.variant(clip_id)
        path = None if clip_id is None else self._file(clip_id)
        if path is None:
            return False
        self.played.append(clip_id)
        return bool(self.robot.play_clip(path))

    def start(self, clip_id: str) -> float | None:
        """Hand a clip to the speaker without waiting for it to end: its length in seconds, or None if not played."""
        clip_id = self.variant(clip_id)
        path = None if clip_id is None else self._file(clip_id)
        if path is None:
            return None
        seconds = wav_seconds(path)
        self.played.append(clip_id)
        return seconds if self.robot.play_clip(path, wait=False) else None

    def think_aloud(self) -> float | None:
        """The check-in's thinking phrase (「我再想一下喔。」 and its like), handed to the speaker once the patient's
        turn is in (voice.DoneListener's think_aloud): its length in seconds, or None without those clips."""
        return self.start(THINKING)

    def play_med_prompt(self, med_id) -> bool:
        clip_id = f"med_{med_id}"
        if self.path(clip_id).is_file():
            return self.play(clip_id)
        return self.play(GENERIC_MED_PROMPT)

    def audit(self) -> set[str]:
        """Record every manifest clip without a WAV, so the first heartbeat already reports them."""
        for clip_id in self.manifest["clips"]:
            if not self.path(clip_id).is_file():
                self.missing.add(clip_id)
        if self.missing:
            log.warning("missing clips for %s: %s", self.dir.name, ", ".join(sorted(self.missing)))
        return set(self.missing)

    @property
    def missing_count(self) -> int:
        return len(self.missing)
