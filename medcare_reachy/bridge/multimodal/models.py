"""The voice and text emotion models, from the Hugging Face Hub, with the login already on the machine.

`huggingface_hub` finds the token `hf auth login` stored (or HF_TOKEN) by itself; nothing is configured here.
Files land in the Hub cache and are reused on every later start; a later start without network still finds them.
MULTIMODAL_MODELS_DIR points at a folder with the repo's layout instead (no download at all).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

MODEL_REPO = os.environ.get("MULTIMODAL_MODEL_REPO", "pearlyjam21/multimodal-emotion-zh")
PATTERNS = ["config.json", "speech/*.onnx", "text/*.onnx", "text/tokenizer.json"]   # no face files: bundled


def fetch_models(repo_id: str = MODEL_REPO, local_dir: str | None = None) -> Path:
    local = local_dir or os.environ.get("MULTIMODAL_MODELS_DIR")
    if local:
        return Path(local)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, allow_patterns=PATTERNS))


def load_config(root: Path) -> dict:
    return json.loads((Path(root) / "config.json").read_text(encoding="utf-8"))
