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


# The server's spoken dose refusals (dose_safety, as speech_text) and replies like the model's.
REFUSALS = [
    "这次没有记录，因为还没到吃普拿疼的时间（明天早上6点以后才可以）。如果您刚刚已经吃了，下一次的药请先问过家人再吃。",
    "这次没有记录，因为普拿疼您凌晨12点05分已经吃过了。已经通知家人；如果觉得不舒服，请马上告诉家人。",
    "这次没有记录，因为今天的这个药已经吃满 4 次了。如果觉得不舒服，请马上告诉家人。",
    "这次没有记录，因为昨天晚上10点05分的药已经错过了。如果您刚刚已经吃了，下一次的药请先问过家人再吃。",
    "This wasn't recorded: you already took this medicine at 12:05 am. Your family has been told.",
]
REPLIES = [
    "听起来您昨天晚上睡觉睡得不太好要不要今天早一点休息呢？",
    "请您记得在晚上八点也就是20:00准时吃药喔。",
    "您今天大概走了一万两千步差不多是1,000公尺左右呢。",
    "我知道了，您说的是Hydrochlorothiazide这个药对吧？",
    "今天散步很久現在膝蓋有一點累想坐下來休息再喝一杯溫水" * 2,
]
BREAKS = "，、,。！？!?；;.」』）)》\"'”’"


def _spoken(pieces):
    return "".join(pieces).replace(" ", "")


def test_chunks_cut_only_at_punctuation_and_lose_nothing():
    for text in REFUSALS + REPLIES:
        pieces = chunks(text)
        assert _spoken(pieces) == text.replace(" ", ""), text
        assert all(piece and piece[-1] in BREAKS for piece in pieces[:-1]), pieces


def test_chunks_keep_words_and_numbers_whole():
    def whole(text, *words):
        pieces = chunks(text)
        for word in words:
            assert any(word in piece for piece in pieces), (word, pieces)

    whole(REFUSALS[0], "明天早上6点以后")
    whole(REFUSALS[1], "已经吃过了")
    whole(REFUSALS[2], "吃满 4 次了")
    whole(REFUSALS[3], "已经错过了")
    whole(REPLIES[0], "要不要今天")
    whole(REPLIES[1], "20:00")
    whole(REPLIES[2], "1,000公尺")
    whole(REPLIES[3], "Hydrochlorothiazide")
    whole("The reading is 25.5 degrees today.", "25.5")
    assert chunks(REPLIES[4]) == [REPLIES[4]]     # a long clause without a comma is spoken whole
    assert len(chunks(REFUSALS[4])) == 2
    assert chunks("Hydrochlorothiazide.") == ["Hydrochlorothiazide."]


def test_chunks_keep_repeated_marks_quotes_and_ellipses_together():
    assert chunks("好的！！今天也要加油喔！") == ["好的！！", "今天也要加油喔！"]
    assert chunks("「好的。」您說得對。") == ["「好的。」", "您說得對。"]
    assert chunks("Wait... ok. Fine.") == ["Wait...", "ok.", "Fine."]
    assert chunks("好，1, 2, 3。") == ["好，1, 2, 3。"]


def test_chunks_start_with_a_complete_natural_comma_clause():
    assert chunks("今天天氣真好，出門散步看看。") == ["今天天氣真好，", "出門散步看看。"]
    assert chunks("好的，我知道了。") == ["好的，我知道了。"]              # a short opening is not worth a pause
    assert chunks(REFUSALS[0])[:2] == ["这次没有记录，", "因为还没到吃普拿疼的时间（明天早上6点以后才可以）。"]


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


def test_a_long_reply_with_commas_starts_playback_before_all_chunks_are_synthesised(tmp_path, monkeypatch):
    for name in speech.TTS_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    tts, robot = SlowTTS(), Robot()
    monkeypatch.setattr(speech, "_engine", None)
    text = "那走太久膝蓋會累，現在有沒有好一點，如果痛得比較久，可以問問醫生或藥師。"
    assert Speaker(robot, tmp_path, load=lambda models_dir: tts).say(text) is True
    assert "".join(piece for _, piece in tts.started) == text
    assert len(tts.started) > 2 and robot.played[0][0] < tts.started[-1][0]


def test_natural_first_clause_reduces_first_audio_wait_without_losing_text(tmp_path, monkeypatch):
    for name in speech.TTS_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")

    class LengthTTS(SlowTTS):
        def generate(self, text, sid, speed):
            self.started.append((time.monotonic(), text))
            time.sleep(0.03 * len(text))
            return types.SimpleNamespace(samples=np.zeros(100, np.float32), sample_rate=16000)

    tts, robot = LengthTTS(), Robot()
    monkeypatch.setattr(speech, "_engine", None)
    text = "今天天氣真好，出門散步看看。"
    stats = {}
    assert Speaker(robot, tmp_path, load=lambda models_dir: tts).say(text, stats) is True
    assert "".join(piece for _, piece in tts.started) == text
    assert stats["tts_first_audio_ms"] < 350
    assert stats["tts_chunks"] == 2


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
