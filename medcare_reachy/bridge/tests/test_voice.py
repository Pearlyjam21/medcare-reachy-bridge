"""DoneListener: phrase matching, audio conversion, and the listening thread with fake speech models."""

import threading
import time
import types

import numpy as np
import pytest

from medcare_reachy.bridge.voice import (
    DONE_MUTE_TAIL_SECONDS, ECHO_WINDOW, MIN_SILENCE_SECONDS, MIN_SPEECH_SECONDS, MUTE_TAIL_SECONDS, SAMPLE_RATE,
    STT_FILES, VAD_WINDOW, DoneListener, done_phrase, to_mono_16k, without_ack)


@pytest.mark.parametrize("text,word", [
    ("我吃完了。", "吃完"), ("我吃完了", "吃完"), ("吃好了", "吃好"), ("藥吃了", "吃了"), ("已經吃過了", "吃過"),
    ("I'm finished.", "finished"), ("OK, I took it", "took it"),
    ("我還沒吃完", None), ("还没吃", None), ("I haven't finished", None), ("not yet", None),
    ("今天天氣很好", None), ("", None), (None, None),
])
def test_done_phrase(text, word):
    assert done_phrase(text) == word


@pytest.mark.parametrize("text,words", [
    ("嗯", ""), ("嗯。", ""), ("嗯嗯，", ""), ("恩", ""), ("呃，嗯", ""), ("Mm-hmm.", ""), ("Hmm", ""), ("mhm", ""),
    ("嗯，活了", "活了"), ("嗯活了", "活了"), ("嗯 我吃过了", "我吃过了"), ("呃，嗯，我去散步了", "我去散步了"),
    ("Mm, yes.", "yes."),
    ("我嗯不想", "我嗯不想"), ("活了嗯", "活了嗯"), ("恩人来了", "恩人来了"), ("额头疼", "额头疼"),
    ("哼歌", "哼歌"), ("mom", "mom"), ("hmmm, home", "home"),
])
def test_only_a_leading_mm_is_taken_off(text, words):
    assert without_ack(text) == words


def test_audio_is_mixed_to_mono_and_decimated_to_16k():
    stereo = np.ones((4800, 2), np.float32) * np.array([0.2, 0.4], np.float32)
    mono = to_mono_16k(stereo, 48000)
    assert mono.shape == (1600,) and np.allclose(mono, 0.3)
    assert to_mono_16k(np.zeros(160, np.float32), 16000).shape == (160,)
    assert len(to_mono_16k(np.zeros(441, np.float32), 44100)) == 160


class Mic:
    def __init__(self):
        self.chunks = []

    def get_audio(self):
        return (self.chunks.pop(0), 16000) if self.chunks else None


class FakeVad:
    """Emits one speech segment for every four windows it is fed: two of speech, then two of the silence that ended
    it (none while `emit` is False). `speech` is what is_speech_detected() answers; `audio` is what it was fed."""

    def __init__(self):
        self.fed, self.segments, self.resets, self.speech = 0, [], 0, False
        self.count = 0   # windows since the last reset: segments' start counts samples from there, like sherpa-onnx
        self.emit, self.audio = True, []

    def accept_waveform(self, samples):
        assert len(samples) == VAD_WINDOW
        self.audio.append(np.array(samples))
        self.fed += 1
        self.count += 1
        if self.emit and self.count % 4 == 0:
            self.segments.append(types.SimpleNamespace(samples=np.zeros(VAD_WINDOW * 2, np.float32),
                                                       start=(self.count - 4) * VAD_WINDOW))

    def is_speech_detected(self):
        return self.speech

    def empty(self):
        return not self.segments

    @property
    def front(self):
        return self.segments[0]

    def pop(self):
        self.segments.pop(0)

    def reset(self):
        self.resets += 1
        self.count = 0


def listener(tmp_path, text="我吃完了"):
    for name in STT_FILES:
        (tmp_path / name).write_bytes(b"x")
    mic, vad = Mic(), FakeVad()
    heard_texts = []

    def transcribe(segment):
        heard_texts.append(len(segment))
        return text

    voice = DoneListener(mic, "zh-TW", tmp_path, load=lambda models_dir, language: (vad, transcribe))
    return voice, mic, vad, heard_texts


def wait_for(predicate, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_hears_finished_only_while_active(tmp_path):
    voice, mic, vad, texts = listener(tmp_path)
    voice.start()
    try:
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]
        assert wait_for(lambda: not mic.chunks)
        time.sleep(0.05)
        assert texts == [] and voice.heard_since(0) is None   # inactive: audio is dropped unheard
        voice.set_active(True)
        start = time.monotonic()
        mic.chunks = [np.zeros(VAD_WINDOW * 2, np.float32), np.zeros(VAD_WINDOW * 2, np.float32)]
        assert wait_for(lambda: voice.heard_since(start) == "吃完")
        assert voice.heard_since(time.monotonic() + 10) is None
        voice.set_active(False)
        voice.set_active(True)
        assert voice.heard_since(start) is None   # a new listening period forgets the old claim
    finally:
        voice.stop()


def test_muted_while_the_robot_speaks(tmp_path):
    voice, mic, vad, texts = listener(tmp_path)
    voice.start()
    try:
        voice.set_active(True)
        voice.hold()
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]
        assert wait_for(lambda: not mic.chunks)
        time.sleep(0.05)
        assert texts == [] and voice.heard_since(0) is None
        voice.release()
        assert voice.heard_since(0) is None
    finally:
        voice.stop()


def test_negated_speech_is_not_a_claim(tmp_path):
    voice, mic, vad, texts = listener(tmp_path, text="我還沒吃完")
    voice.start()
    try:
        voice.set_active(True)
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]
        assert wait_for(lambda: texts)
        time.sleep(0.05)
        assert voice.heard_since(0) is None
    finally:
        voice.stop()


def chat_listener(tmp_path, words, decode_seconds=0.0, gates=()):
    """A chat listener on a fake clock (stopped at 100 s) whose SenseVoice takes `decode_seconds` of it per segment,
    and, for the n-th segment, first waits for gates[n] to be set."""
    for name in STT_FILES:
        (tmp_path / name).write_bytes(b"x")
    clock, mic, vad = [100.0], Mic(), FakeVad()
    words, gates, decoded = iter(words), list(gates), []

    def transcribe(segment):
        if len(decoded) < len(gates):
            assert gates[len(decoded)].wait(2)
        decoded.append(len(segment))
        clock[0] += decode_seconds
        return next(words)

    voice = DoneListener(mic, "zh-TW", tmp_path, clock=lambda: clock[0],
                         load=lambda models_dir, language: (vad, transcribe))
    return voice, mic, vad, clock


def take_when_ready(voice, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        heard = voice.take_utterance_with_metrics()
        if heard is not None:
            return heard
        time.sleep(0.01)
    return None


def test_chat_hands_over_as_soon_as_the_last_segment_is_decoded_with_its_timings(tmp_path):
    voice, mic, vad, clock = chat_listener(tmp_path, ["我早上去散步了"], decode_seconds=0.3)
    voice.start()
    try:
        voice.set_active(True, mode="chat")
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]
        # No pause is waited for once it is decoded: the fake clock only moved by the 0.3 s of decoding.
        text, metrics = take_when_ready(voice)
        assert text == "我早上去散步了" and clock[0] == pytest.approx(100.3)
        # Speech ended 2 windows (64 ms) before the last sample fed; decoding took 300 ms after that.
        assert metrics == {"speech_ms": 64, "segments": 1, "vad_release_ms": 64, "stt_ms": 300, "stt_last_ms": 300,
                           "handover_ms": 364, "echo_dropped": 0, "listen_mode": "chat"}
        assert voice.take_utterance() is None
        assert voice.heard_since(0) is None               # chat words are never treated as "finished"
    finally:
        voice.stop()


def test_chat_waits_for_speech_that_resumed_while_decoding_and_joins_it(tmp_path):
    first, second = threading.Event(), threading.Event()
    voice, mic, vad, clock = chat_listener(tmp_path, ["我早上", "去散步了"], decode_seconds=0.2,
                                           gates=(first, second))
    voice.start()
    try:
        voice.set_active(True, mode="chat")
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]
        assert wait_for(lambda: vad.fed == 4 and not vad.segments)    # the first segment is being decoded
        vad.speech = True                                              # ... while the patient goes on talking
        mic.chunks = [np.zeros(VAD_WINDOW * 2, np.float32)]
        first.set()
        end = time.time() + 0.15
        while time.time() < end:   # decoded, but the audio that came meanwhile has the patient speaking again
            assert voice.take_utterance() is None
            time.sleep(0.005)
        assert vad.fed == 6
        vad.speech = False
        mic.chunks = [np.zeros(VAD_WINDOW * 2, np.float32)]            # their pause ends the second segment
        assert wait_for(lambda: vad.fed == 8 and not vad.segments)
        time.sleep(0.05)
        assert voice.take_utterance() is None                          # the first is queued; the second decoding
        second.set()
        text, metrics = take_when_ready(voice)
        assert text == "我早上 去散步了"
        assert metrics["segments"] == 2 and metrics["speech_ms"] == 128
        assert metrics["stt_ms"] == 400 and metrics["stt_last_ms"] == 200
    finally:
        voice.stop()


def test_chat_drops_reachy_heard_back_as_its_own_voice(tmp_path):
    voice, mic, vad, clock = chat_listener(
        tmp_path, ["今天感觉怎么样。", "今天感觉怎么羊", "感觉", "我今天感觉怎么样呢"])
    voice.start()
    try:
        voice.set_active(True, mode="chat")
        voice.hold()                                   # Reachy speaks ...
        voice.note_spoken("今天感觉怎么样？")
        voice.release()
        clock[0] += MUTE_TAIL_SECONDS                  # ... and listening resumes
        mic.chunks = [np.zeros(VAD_WINDOW * 12, np.float32)]
        # The same words, then a near miss, are its echo; "感觉" is too short to tell and is kept.
        text, metrics = take_when_ready(voice)
        assert text == "感觉" and metrics["echo_dropped"] == 2
        # The patient repeating Reachy's words well after listening resumed is not an echo.
        clock[0] += ECHO_WINDOW + 0.5
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]
        text, metrics = take_when_ready(voice)
        assert text == "我今天感觉怎么样呢" and metrics["echo_dropped"] == 0
    finally:
        voice.stop()


def heard_back(tmp_path, mode, words, windows):
    """A listener in `mode` that just heard Reachy say "...我吃完了" and resumed listening, fed `windows` of audio."""
    voice, mic, vad, clock = chat_listener(tmp_path, words)
    voice.start()
    voice.set_active(True, mode=mode)
    voice.hold()
    voice.note_spoken("吃完以后，请跟我说「我吃完了」。")
    voice.release()
    clock[0] += DONE_MUTE_TAIL_SECONDS            # both modes listen again, well inside the echo window
    mic.chunks = [np.zeros(VAD_WINDOW * windows, np.float32)]
    return voice


def test_echo_guard_only_applies_to_chat(tmp_path):
    # chat: "我吃完了" heard right after Reachy said it is its own voice ("好的" is too short to tell, and kept).
    voice = heard_back(tmp_path, "chat", ["我吃完了", "好的"], 8)
    try:
        text, metrics = take_when_ready(voice)
        assert text == "好的" and metrics["echo_dropped"] == 1
    finally:
        voice.stop()
    # done: the same words, at the same moment, are the patient's claim (it repeats the prompt by design).
    voice = heard_back(tmp_path, "done", ["我吃完了"], 4)
    try:
        assert wait_for(lambda: voice.heard_since(0) == "吃完")
    finally:
        voice.stop()


def test_done_mode_stays_muted_longer_after_reachy_speaks(tmp_path):
    voice, mic, vad, clock = chat_listener(tmp_path, ["我吃完了"])
    voice.start()
    try:
        voice.set_active(True)                     # "done": watching a dose after "say when you're done"
        voice.hold()
        voice.release()
        clock[0] += MUTE_TAIL_SECONDS              # enough for chat, which has the echo guard ...
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]
        assert wait_for(lambda: not mic.chunks)
        time.sleep(0.05)
        assert voice.heard_since(0) is None and vad.fed == 0   # ... but "done" still drops the prompt's echo
        clock[0] = 100.0 + DONE_MUTE_TAIL_SECONDS
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]
        assert wait_for(lambda: voice.heard_since(0) == "吃完")
    finally:
        voice.stop()


def test_chat_words_said_while_reachy_answered_are_timed_from_when_it_finished_speaking(tmp_path):
    voice, mic, vad, clock = chat_listener(tmp_path, ["然后吃了早餐"])
    voice.start()
    try:
        voice.set_active(True, mode="chat")
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]   # said while the server is still thinking
        assert wait_for(lambda: voice._segments)
        clock[0] += 5.0                                       # the reply arrives ...
        voice.note_spoken("真好，您早上做了什么？")
        voice.hold()
        clock[0] += 3.0                                       # ... and Reachy speaks it
        voice.release()
        clock[0] += 0.2                                       # the session takes what was said meanwhile
        text, metrics = take_when_ready(voice)
        assert text == "然后吃了早餐"
        assert metrics["handover_ms"] == 200 and metrics["vad_release_ms"] == 64   # not 8000+ ms of waiting
    finally:
        voice.stop()


def test_chat_times_audio_that_piled_up_during_a_decode_by_when_it_was_heard(tmp_path):
    voice, mic, vad, clock = chat_listener(tmp_path, ["我早上", "去散步了"], decode_seconds=2.0)
    voice.start()
    try:
        voice.set_active(True, mode="chat")
        # The second part was already waiting while the first took 2 s to decode: it is read at 102 s, but the
        # patient stopped speaking at 100.064 s, and the release and hand-over are timed from then.
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32), np.zeros(VAD_WINDOW * 4, np.float32)]
        text, metrics = take_when_ready(voice)
        assert text == "我早上 去散步了" and metrics["segments"] == 2
        assert metrics["vad_release_ms"] == 1936 and metrics["handover_ms"] == 3936
    finally:
        voice.stop()


def test_chat_tells_the_session_the_patient_paused_before_their_words_are_decoded(tmp_path):
    decoding = threading.Event()
    voice, mic, vad, clock = chat_listener(tmp_path, ["我早上去散步了"], decode_seconds=0.3, gates=(decoding,))
    voice.start()
    try:
        voice.set_active(True, mode="chat")
        assert voice.heard_pause() is None
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]
        assert wait_for(lambda: voice.heard_pause() is not None)
        # Two windows of speech, then the two of silence that ended it: the patient stopped 64 ms before 100 s.
        assert voice.heard_pause() == pytest.approx(100.0 - 2 * VAD_WINDOW / SAMPLE_RATE)
        assert voice.take_utterance() is None                        # SenseVoice is still on it
        decoding.set()
        text, metrics = take_when_ready(voice)
        assert text == "我早上去散步了" and "ack_ms" not in metrics   # no 「嗯」, no 「嗯」 timings
        assert voice.heard_pause() is None
    finally:
        voice.stop()


def test_chat_forgets_a_pause_that_decoded_to_nothing(tmp_path):
    voice, mic, vad, clock = chat_listener(tmp_path, ["", "我早上去散步了"])
    voice.start()
    try:
        voice.set_active(True, mode="chat")
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]          # a cough
        assert wait_for(lambda: vad.fed == 4 and not vad.segments)
        time.sleep(0.05)
        assert voice.heard_pause() is None and voice.take_utterance() is None
        clock[0] = 101.0
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]
        assert take_when_ready(voice)[0] == "我早上去散步了"
    finally:
        voice.stop()


def acknowledged(tmp_path, words, decode_seconds=0.3):
    """A chat listener that heard the patient (from 99.872 s, pausing at 99.936 s) and is decoding that while Reachy
    says a 0.1 s 「嗯」 at 100 s. Speech starting from then until its echo can have ended (100.95 s) is checked."""
    decoding = threading.Event()
    voice, mic, vad, clock = chat_listener(tmp_path, words, decode_seconds=decode_seconds, gates=(decoding,))
    voice.start()
    voice.set_active(True, mode="chat")
    mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]
    assert wait_for(lambda: voice.heard_pause() is not None)
    voice.note_ack(0.1)
    return voice, mic, vad, clock, decoding


def test_chat_keeps_the_words_being_decoded_and_everything_heard_under_the_acknowledgement(tmp_path):
    voice, mic, vad, clock, decoding = acknowledged(tmp_path, ["我早上去散步了"])
    try:
        vad.emit = False
        mic.chunks = [np.full(VAD_WINDOW * 8, 0.1, np.float32)]    # heard under the 「嗯」 (100-100.256 s)
        decoding.set()
        # Unlike hold(), the words being decoded are kept, and handed over as soon as they are decoded: the 「嗯」
        # adds no wait.
        text, metrics = take_when_ready(voice)
        assert text == "我早上去散步了" and clock[0] == pytest.approx(100.3)
        assert metrics["ack_ms"] == 64 and metrics["ack_resumed"] is False and metrics["echo_dropped"] == 0
        assert vad.fed == 12
        heard = np.concatenate(vad.audio)
        assert (heard[VAD_WINDOW * 4:] == np.float32(0.1)).all()   # the VAD heard all of it: nothing is silenced
    finally:
        voice.stop()


def test_chat_waits_for_a_patient_going_on_after_the_acknowledgement(tmp_path):
    voice, mic, vad, clock, decoding = acknowledged(tmp_path, ["我早上", "去公园散步了"])
    try:
        vad.emit, vad.speech = False, True
        mic.chunks = [np.full(VAD_WINDOW * 4, 0.1, np.float32)]    # the patient goes on under the 「嗯」 ...
        decoding.set()
        assert wait_for(lambda: vad.fed == 8)
        end = time.time() + 0.15
        while time.time() < end:                                    # ... and the VAD hears them
            assert voice.hears_speech() and voice.take_utterance() is None
            time.sleep(0.005)
        vad.emit, vad.speech = True, False
        mic.chunks = [np.full(VAD_WINDOW * 4, 0.1, np.float32)]    # their next pause ends what they said
        text, metrics = take_when_ready(voice)
        assert text == "我早上 去公园散步了" and metrics["segments"] == 2   # one turn, not two
        assert metrics["ack_ms"] == 64 and metrics["ack_resumed"] is True
    finally:
        voice.stop()


def test_chat_drops_reachys_own_mm_heard_back_but_keeps_the_patients(tmp_path):
    # The patient answers 「嗯」 (yes); Reachy says 「嗯」, which leaks back through the microphone on its own, then at
    # the start of what the patient says next.
    voice, mic, vad, clock, decoding = acknowledged(tmp_path, ["嗯", "嗯。", "嗯，我吃过了", "嗯"], decode_seconds=0.0)
    try:
        clock[0] = 100.384
        mic.chunks = [np.zeros(VAD_WINDOW * 8, np.float32)]    # speech starting at 100.0 and 100.128 s
        decoding.set()
        text, metrics = take_when_ready(voice)
        assert text == "嗯 我吃过了" and metrics["echo_dropped"] == 2
        assert metrics["ack_resumed"] is True
        # Long after Reachy's 「嗯」 a 嗯 is the patient's again.
        clock[0] = 101.5
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]    # speech from 101.372 s
        text, metrics = take_when_ready(voice)
        assert text == "嗯" and metrics["echo_dropped"] == 0
    finally:
        voice.stop()


def test_chat_never_reports_its_own_mm_as_a_pause_to_answer(tmp_path):
    """Speech that may be Reachy's 「嗯」 heard back is reported only once decoded: otherwise Reachy, hearing its own
    「嗯」 after a cough, would say 「嗯」 to it, hear that, and so on."""
    gates = [threading.Event() for _ in range(3)]
    voice, mic, vad, clock = chat_listener(tmp_path, ["", "嗯", "我早上去散步了"], gates=gates)
    voice.start()
    try:
        voice.set_active(True, mode="chat")
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]    # a cough
        assert wait_for(lambda: voice.heard_pause() is not None)
        voice.note_ack(0.4)                                    # Reachy says 「嗯」 at 100 s
        gates[0].set()
        assert wait_for(lambda: voice.heard_pause() is None)   # the cough said nothing
        clock[0] = 100.5
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]    # speech from 100.372 s: the 「嗯」 heard back?
        assert wait_for(lambda: vad.fed == 8 and not vad.segments)
        time.sleep(0.05)
        assert voice.heard_pause() is None                     # not while it is being decoded
        gates[1].set()
        time.sleep(0.05)
        assert voice.heard_pause() is None and voice.take_utterance() is None   # it was: dropped
        clock[0] = 100.7
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]    # the patient, from 100.572 s
        assert wait_for(lambda: vad.fed == 12 and not vad.segments)
        time.sleep(0.05)
        assert voice.heard_pause() is None
        gates[2].set()
        assert wait_for(lambda: voice.heard_pause() is not None)   # their words: a pause the session may answer
        text, metrics = take_when_ready(voice)
        assert text == "我早上去散步了" and metrics["echo_dropped"] == 1   # the 「嗯」 heard back meanwhile
    finally:
        voice.stop()


# ── a sentence resumed after Reachy's 「嗯」, through an energy VAD and a decoder that reads the audio itself ──

class EnergyVad:
    """Like silero with the app's settings: MIN_SPEECH_SECONDS of sound starts a segment, MIN_SILENCE_SECONDS of
    quiet ends it."""

    def __init__(self):
        self.windows, self.segments = [], []
        self.count = self.run = self.quiet = self.first = 0
        self.speaking = False

    def accept_waveform(self, samples):
        self.windows.append(np.asarray(samples, np.float32))
        loud = float(np.abs(samples).max()) > 0.01
        if not self.speaking:
            self.run = self.run + 1 if loud else 0
            if self.run == 1:
                self.first = self.count
            if self.run * VAD_WINDOW >= MIN_SPEECH_SECONDS * SAMPLE_RATE:
                self.speaking, self.quiet = True, 0
        else:
            self.quiet = 0 if loud else self.quiet + 1
            if self.quiet * VAD_WINDOW >= MIN_SILENCE_SECONDS * SAMPLE_RATE:
                end = self.count - self.quiet + 1
                self.segments.append(types.SimpleNamespace(samples=np.concatenate(self.windows[self.first:end]),
                                                           start=self.first * VAD_WINDOW))
                self.speaking, self.run = False, 0
        self.count += 1

    def is_speech_detected(self):
        return self.speaking

    def empty(self):
        return not self.segments

    @property
    def front(self):
        return self.segments[0]

    def pop(self):
        self.segments.pop(0)

    def reset(self):
        self.__init__()


SYLLABLES = {0.11: "我", 0.22: "不", 0.33: "想", 0.44: "活", 0.55: "了"}   # each spoken 0.25 s at this level


def resumed_sentence(models_dir, pause: float, acknowledge: bool) -> str:
    """The patient says 「我不想」, pauses, then 「活了」: microphone audio arrives in 10 ms buffers in real time (and piles
    up while SenseVoice decodes, 0.6 s per s of speech), and the session says a 0.4 s 「嗯」 at its first 0.2 s tick
    after the VAD released 「我不想」. Returns everything heard."""
    levels = {word: level for level, word in SYLLABLES.items()}
    audio = np.concatenate([np.zeros(int(0.3 * SAMPLE_RATE), np.float32)]
                           + [np.full(SAMPLE_RATE // 4, levels[word], np.float32) for word in "我不想"]
                           + [np.zeros(int(pause * SAMPLE_RATE), np.float32)]
                           + [np.full(SAMPLE_RATE // 4, levels[word], np.float32) for word in "活了"]
                           + [np.zeros(int(1.5 * SAMPLE_RATE), np.float32)])
    clock, lock = [100.0], threading.Lock()
    state = {"read": 0, "tick": None, "acked": False, "done": False}
    voice = None

    def acknowledge_by(until: float):
        """The session's tick: 「嗯」 at the first tick after the pause was reported, if that comes by `until`."""
        if not acknowledge or state["acked"] or voice.heard_pause() is None:
            return
        state["tick"] = state["tick"] or (int(clock[0] / 0.2) + 1) * 0.2
        if state["tick"] <= until:
            clock[0] = max(clock[0], state["tick"])
            state["acked"] = True
            voice.note_ack(0.4)

    class Microphone:
        def get_audio(self):
            with lock:
                acknowledge_by(clock[0])
                if state["read"] >= len(audio):
                    state["done"] = True
                    return None
                buffer = audio[state["read"]:state["read"] + SAMPLE_RATE // 100]
                state["read"] += len(buffer)
                clock[0] = max(clock[0], 100.0 + state["read"] / SAMPLE_RATE)
                return buffer, SAMPLE_RATE

    def transcribe(segment):
        decoded = clock[0] + 0.6 * len(segment) / SAMPLE_RATE
        with lock:
            acknowledge_by(decoded)   # the session ticks on while SenseVoice decodes
        clock[0] = decoded
        heard = []
        for start in range(0, len(segment), SAMPLE_RATE // 20):
            peak = float(np.abs(segment[start:start + SAMPLE_RATE // 20]).max())
            word = SYLLABLES[min(SYLLABLES, key=lambda level: abs(level - peak))] if peak > 0.01 else ""
            if word and (not heard or heard[-1] != word):
                heard.append(word)
        return "".join(heard)

    for name in STT_FILES:
        (models_dir / name).write_bytes(b"x")
    voice = DoneListener(Microphone(), "zh-TW", models_dir, clock=lambda: clock[0],
                         load=lambda models_dir, language: (EnergyVad(), transcribe))
    voice.set_active(True, mode="chat")
    voice.start()
    try:
        assert wait_for(lambda: state["done"], timeout=10)
        heard = take_when_ready(voice)
    finally:
        voice.stop()
    assert state["acked"] == acknowledge
    return heard[0] if heard else ""


@pytest.mark.parametrize("pause", [0.6, 0.8, 1.0, 1.4])
def test_words_said_under_or_after_reachys_mm_always_reach_the_transcript(tmp_path, pause):
    """Silencing the microphone under the 「嗯」 once turned 「我不想…活了」 into 「我不想」, which no safety check flags."""
    assert resumed_sentence(tmp_path, pause, acknowledge=True) == "我不想 活了"
    assert resumed_sentence(tmp_path, pause, acknowledge=False) == "我不想 活了"


def test_switching_mode_forgets_what_was_heard(tmp_path):
    voice, mic, vad, texts = listener(tmp_path)
    voice.start()
    try:
        voice.set_active(True, mode="chat")
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]
        assert wait_for(lambda: texts)
        time.sleep(0.05)
        voice.set_active(True, mode="done")
        assert voice.take_utterance() is None
    finally:
        voice.stop()


def test_missing_models_turn_listening_off(tmp_path):
    voice = DoneListener(Mic(), "zh-TW", tmp_path)
    assert voice.available is False
    voice.start()
    voice.set_active(True)
    assert voice._thread is None and voice._listening() is False


def test_the_listener_lowers_its_own_thread_priority_before_loading_the_models(tmp_path, monkeypatch):
    """Decoding speech on the Pi must not starve the camera stream; the decoder threads inherit this priority."""
    from medcare_reachy.bridge import voice as voice_module

    calls, loaded = [], []
    monkeypatch.setattr(voice_module.os, "PRIO_PROCESS", 0, raising=False)
    monkeypatch.setattr(voice_module.os, "setpriority", lambda which, who, nice: calls.append((who, nice)),
                        raising=False)

    def load(models_dir, language):
        loaded.append((threading.get_native_id(), list(calls)))
        return FakeVad(), lambda segment: ""

    for name in STT_FILES:
        (tmp_path / name).write_bytes(b"x")
    voice = DoneListener(Mic(), "zh-TW", tmp_path, load=load)
    voice.start()
    try:
        assert wait_for(lambda: loaded)
    finally:
        voice.stop()
    thread_id, before_load = loaded[0]
    assert thread_id != threading.get_native_id()
    assert before_load == [(thread_id, voice_module.VOICE_NICE)]


def test_lowering_priority_is_best_effort(monkeypatch):
    from medcare_reachy.bridge import voice as voice_module

    def refuse(*args):
        raise PermissionError("not allowed")

    monkeypatch.setattr(voice_module.os, "PRIO_PROCESS", 0, raising=False)
    monkeypatch.setattr(voice_module.os, "setpriority", refuse, raising=False)
    assert voice_module.lower_thread_priority() is False
