"""tools/deploy_to_robot.py and the check-in's 「嗯」 and thinking phrases: rendered with their own speeds, kept only
when the app recognises their echo, and a deploy that leaves any clip without a WAV fails."""

import importlib.util
import io
import json
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from medcare_reachy.bridge.clips import load_manifest
from medcare_reachy.bridge.voice import without_filler

TOOL = Path(__file__).resolve().parent.parent / "tools" / "deploy_to_robot.py"
spec = importlib.util.spec_from_file_location("deploy_to_robot_fillers", TOOL)
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)

THINKING = ["thinking_1", "thinking_2", "thinking_3"]


def test_the_thinking_phrases_are_acknowledgements_rendered_from_simplified_words():
    manifest = load_manifest()
    assert deploy.fillers(manifest) == {"ack", *THINKING}
    assert deploy.SPOKEN["thinking_1"] == ("我再想一下喔。", "我再想一下哦。")
    assert deploy.SPOKEN["thinking_2"] == ("讓我想一想喔。", "让我想一想哦。")
    assert deploy.SPOKEN["thinking_3"] == ("我想想看喔。", "我想想看哦。")
    for clip_id in THINKING:
        job = deploy.render_job_clip(clip_id, manifest)
        assert job["filler"] is True and job["text"] == deploy.SPOKEN[clip_id][1]
        assert job["speeds"] == list(deploy.RENDER_SPEEDS[clip_id]) and job["speeds"][0] == 1.0   # ~1.1-1.3 s
    prompt = deploy.render_job_clip("say_when_done", manifest)
    assert prompt["filler"] is False and prompt["speeds"] == list(deploy.DEFAULT_SPEEDS)   # prompts stay slow
    assert deploy.render_job_clip("ack", manifest)["filler"] is True


def test_every_acknowledgement_s_own_words_are_recognised_as_its_echo():
    manifest = load_manifest()
    for clip_id in deploy.fillers(manifest):
        written_for, spoken = deploy.SPOKEN[clip_id]
        assert without_filler(spoken) == "" and without_filler(written_for) == "", clip_id


def test_a_new_robot_renders_the_thinking_phrases():
    manifest = load_manifest()
    rendered = {clip_id: entry["zh-TW"] for clip_id, entry in manifest["clips"].items() if clip_id not in THINKING}
    on_robot = set(rendered)
    assert deploy.plan_clips(manifest, rendered, None, on_robot)[1] == THINKING


def test_any_clip_left_without_a_wav_fails_the_deploy():
    manifest = load_manifest()
    assert deploy.absent_clips(manifest, set(manifest["clips"])) == []
    assert deploy.absent_clips(manifest, set(manifest["clips"]) - {"thinking_2"}) == ["thinking_2"]


def test_an_unrecognised_phrase_counts_as_not_written():
    job = {"clips": [{"id": "thinking_1"}, {"id": "thinking_2"}]}
    output = ("render thinking_2 speed=1.0 heard='很音响哦。' ok=False echo-not-recognised\n"
              "unrecognised thinking_2\nwrote thinking_1 heard-ok\nrender-exit=0")
    assert deploy.unrecognised(output) == ["thinking_2"]
    assert deploy.render_problems(output, job) == (["thinking_2"], [])


# ── the render script itself, on a fake voice and recogniser ─────────────

def run_render(tmp_path, monkeypatch, job_clips, heard):
    """Runs RENDER as the robot would, with Matcha and Whisper replaced: `heard[(text, speed)]` is what the
    recogniser writes for that render. Returns (printed output, clips folder, speeds tried per text)."""
    tried = []
    fake = types.ModuleType("sherpa_onnx")

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Tts:
        def __init__(self, config):
            pass

        def generate(self, text, sid=0, speed=1.0):
            tried.append((text, speed))
            samples = np.concatenate([np.zeros(1000), np.full(int(2205 * speed), 0.3)]).astype(np.float32)
            return types.SimpleNamespace(samples=samples, sample_rate=22050)

    class WhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, path, **kwargs):
            return iter([types.SimpleNamespace(text=heard[tried[-1]])]), None

    fake.OfflineTtsConfig = fake.OfflineTtsModelConfig = fake.OfflineTtsMatchaModelConfig = Config
    fake.OfflineTts = Tts
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)
    fake_whisper = types.ModuleType("faster_whisper")
    fake_whisper.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_whisper)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps({"rendered": {"thanks": "謝謝您！"}, "clips": job_clips}, ensure_ascii=False),
                        encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["render", str(job_path)])
    printed = io.StringIO()
    monkeypatch.setattr(sys, "stdout", printed)
    exec(compile(deploy.RENDER, "render_clips.py", "exec"), {"__name__": "__main__"})
    return printed.getvalue(), tmp_path / ".medcare_reachy" / "clips" / "zh-TW", tried


def test_the_render_script_keeps_only_acknowledgements_whose_echo_the_app_recognises(tmp_path, monkeypatch):
    manifest = load_manifest()
    job = [deploy.render_job_clip(clip_id, manifest) for clip_id in ("thinking_1", "thinking_2", "help")]
    one, two, help_text = (clip["text"] for clip in job)
    heard = {
        (one, 1.0): "我再想一下活。", (one, 1.1): "我再想一下哦。",        # the second speed is kept
        **{(two, speed): "很音响哦。" for speed in (1.0, 1.1, 0.9)},       # never recognisable: not written
        **{(help_text, speed): "需要帮忙吗？准备好的时候，请在我面前吃。" for speed in deploy.DEFAULT_SPEEDS},
    }
    output, clips, tried = run_render(tmp_path, monkeypatch, job, heard)
    assert [speed for text, speed in tried if text == one] == [1.0, 1.1]
    assert [speed for text, speed in tried if text == two] == [1.0, 1.1, 0.9]
    assert [speed for text, speed in tried if text == help_text] == list(deploy.DEFAULT_SPEEDS)
    lines = output.splitlines()
    assert "render thinking_1 speed=1.0 heard='我再想一下活。' ok=False echo-not-recognised" in lines
    assert "wrote thinking_1 heard-ok" in lines and "unrecognised thinking_2" in lines
    assert "wrote help not-heard-as-written" in lines                 # a prompt: kept with a warning, as before
    assert {path.name for path in clips.glob("*.wav")} == {"thinking_1.wav", "help.wav"}
    with wave.open(str(clips / "thinking_1.wav"), "rb") as wav:        # the 1.1 render, leading silence trimmed
        assert wav.getnframes() == pytest.approx(int(2205 * 1.1) + 220, abs=5)
    record = json.loads((clips / "rendered.json").read_text(encoding="utf-8"))
    assert record == {"thanks": "謝謝您！", "thinking_1": "我再想一下喔。",
                      "help": manifest["clips"]["help"]["zh-TW"]}
    missing, misheard = deploy.render_problems(output + "\nrender-exit=0", {"clips": job})
    assert missing == ["thinking_2"] and misheard == ["help"]
    assert deploy.unrecognised(output) == ["thinking_2"]
