"""Synthesise prompt WAVs from clips/manifest.json with a pluggable TTS engine (not run in tests).

    python -m reachy_bridge.tools.make_clips --engine mypkg.tts:synthesise --language zh-TW
    python -m reachy_bridge.tools.make_clips --engine mypkg.tts:synthesise --language en --med 12="Metformin"

The engine is `callable(text: str, language: str) -> (samples, sample_rate)`, samples being mono
float32 in [-1, 1]. Every generated clip is still `pending_clinician_review` until the text and the
audio have been reviewed; do not ship unreviewed clips.
"""

import argparse
import importlib
import wave
from pathlib import Path

import numpy as np

from reachy_bridge.clips import load_manifest
from reachy_bridge.config import LANGUAGES, PACKAGE_DIR


def load_engine(spec: str):
    module, _, name = spec.partition(":")
    if not name:
        raise SystemExit("--engine must look like package.module:function")
    return getattr(importlib.import_module(module), name)


def write_wav(path: Path, samples, rate: int) -> None:
    pcm = (np.clip(np.asarray(samples, np.float32), -1, 1) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(rate))
        wav.writeframes(pcm.tobytes())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--language", choices=LANGUAGES, required=True)
    parser.add_argument("--out", type=Path, default=PACKAGE_DIR / "clips")
    parser.add_argument("--med", action="append", default=[], metavar="MED_ID=NAME",
                        help="also write med_<MED_ID>.wav from the per-medication template")
    parser.add_argument("--force", action="store_true", help="overwrite existing WAVs")
    args = parser.parse_args(argv)

    synthesise = load_engine(args.engine)
    manifest = load_manifest()
    texts = {clip_id: entry[args.language] for clip_id, entry in manifest["clips"].items()}
    template = manifest["per_medication"]["template"][args.language]
    for item in args.med:
        med_id, _, name = item.partition("=")
        texts[f"med_{int(med_id)}"] = template.format(med_name=name.strip())

    for clip_id, text in texts.items():
        path = args.out / args.language / f"{clip_id}.wav"
        if path.exists() and not args.force:
            print(f"skip {path} (exists)")
            continue
        samples, rate = synthesise(text, args.language)
        write_wav(path, samples, rate)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
