"""Prerecorded prompt clips: manifest lookup and playback through the robot."""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

MANIFEST = Path(__file__).resolve().parent / "clips" / "manifest.json"
GENERIC_MED_PROMPT = "med_prompt_generic"


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class ClipPlayer:
    """Plays `<clips_dir>/<language>/<clip_id>.wav`; a missing file is logged once and counted, never fatal."""

    def __init__(self, robot, clips_dir: Path, language: str, manifest: dict | None = None):
        self.robot = robot
        self.dir = Path(clips_dir) / language
        self.manifest = manifest if manifest is not None else load_manifest()
        self.missing: set[str] = set()
        self.played: list[str] = []

    def path(self, clip_id: str) -> Path:
        return self.dir / f"{clip_id}.wav"

    def play(self, clip_id: str) -> bool:
        if clip_id not in self.manifest["clips"] and not clip_id.startswith("med_"):
            raise KeyError(f"unknown clip {clip_id}")
        path = self.path(clip_id)
        if not path.is_file():
            if clip_id not in self.missing:
                log.warning("clip %s missing at %s", clip_id, path)
            self.missing.add(clip_id)
            return False
        self.played.append(clip_id)
        return bool(self.robot.play_clip(path))

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
