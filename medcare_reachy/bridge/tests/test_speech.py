"""Speaker: replies are split into short chunks and each is played while the next is synthesised."""

import threading
import time
import types

import numpy as np

from medcare_reachy.bridge import speech
from medcare_reachy.bridge.speech import Speaker, chunks


def test_chunks_split_sentences_and_long_sentences_at_commas():
    assert chunks("好的。今天天氣很好！") == ["好的。", "今天天氣很好！"]
    long = "那走太久膝蓋會累，現在有沒有好一點，如果痛得比較久，可以問問醫生或藥師。"
    pieces = chunks(long)
    assert "".join(pieces) == long and all(len(p) <= speech.MAX_CHUNK for p in pieces) and len(pieces) >= 3
    assert chunks("Hello there. How are you?") == ["Hello there.", "How are you?"]
    assert chunks("") == [] and chunks(None) == []


class Robot:
    def __init__(self):
        self.played, self.lock = [], threading.Lock()

    def play_clip(self, path):
        with self.lock:
            self.played.append((time.monotonic(), path.read_bytes()[:4]))
        time.sleep(0.05)
        return True


class SlowTTS:
    """0.05 s per chunk; records when each synthesis started."""

    def __init__(self):
        self.started = []

    def generate(self, text, sid, speed):
        self.started.append((time.monotonic(), text))
        time.sleep(0.05)
        return types.SimpleNamespace(samples=np.zeros(100, np.float32), sample_rate=16000)


def test_say_plays_every_chunk_in_order_while_synthesising_ahead(tmp_path, monkeypatch):
    for name in speech.TTS_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    tts, robot = SlowTTS(), Robot()
    monkeypatch.setattr(speech, "_engine", None)
    speaker = Speaker(robot, tmp_path, load=lambda models_dir: tts)
    assert speaker.say("第一句。第二句。第三句。") is True
    assert [text for _, text in tts.started] == ["第一句。", "第二句。", "第三句。"]
    assert len(robot.played) == 3 and all(head == b"RIFF" for _, head in robot.played)
    # the second chunk was being synthesised before the first finished playing
    assert tts.started[1][0] < robot.played[0][0] + 0.05


def test_say_without_models_or_text_says_nothing(tmp_path):
    speaker = Speaker(Robot(), tmp_path)
    assert speaker.available is False and speaker.say("你好") is False


def test_say_reports_its_timings(tmp_path, monkeypatch):
    for name in speech.TTS_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    tts, robot = SlowTTS(), Robot()
    monkeypatch.setattr(speech, "_engine", None)
    speaker = Speaker(robot, tmp_path, load=lambda models_dir: tts)
    stats = {}
    assert speaker.say("第一句。第二句。第三句。", stats) is True
    assert set(stats) == {"tts_first_audio_ms", "tts_total_ms", "tts_chunks", "tts_synth_ms"}
    assert all(isinstance(value, int) for value in stats.values())
    assert stats["tts_chunks"] == 3
    assert stats["tts_synth_ms"] >= 120                                   # three syntheses of 50 ms
    assert 30 <= stats["tts_first_audio_ms"] < stats["tts_total_ms"]      # the first waits for one synthesis
    assert stats["tts_total_ms"] >= stats["tts_first_audio_ms"] + 120     # ... then three clips of 50 ms play


def test_say_tells_when_its_first_sound_is_about_to_play(tmp_path, monkeypatch):
    """The check-in switches Reachy from thinking to speaking at its first sound, not when synthesis starts."""
    for name in speech.TTS_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    tts, robot = SlowTTS(), Robot()
    monkeypatch.setattr(speech, "_engine", None)
    cues = []
    assert Speaker(robot, tmp_path, load=lambda models_dir: tts).say(
        "第一句。第二句。", on_audio=lambda: cues.append((time.monotonic(), len(robot.played)))) is True
    assert len(cues) == 1 and cues[0][1] == 0                             # once, before anything played
    assert tts.started[0][0] + 0.05 <= cues[0][0] <= robot.played[0][0]   # after the first chunk's synthesis


def test_say_with_nothing_to_say_reports_nothing(tmp_path):
    stats = {}
    assert Speaker(Robot(), tmp_path).say("你好", stats) is False and stats == {}
