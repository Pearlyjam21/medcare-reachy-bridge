"""Reachy's thinking phrase: said once the patient's turn is handed over, and its echo removed by its words, only right
after Reachy said it, and never together with a word of the patient's.

The 0.5.3 review played the phrase the moment the patient paused, through the real Matcha voice and SenseVoice: a
patient going on under it came out garbled (「我不想…活了」 lost its 活, or became 「我不想 我再想一活了」, which the
keyword screen misses). Said after the hand-over, the phrase can no longer land inside the turn the patient is still
saying.
"""

import re
import threading
import time

import numpy as np
import pytest

from medcare_reachy.bridge.clips import THINKING, load_manifest
from medcare_reachy.bridge.tests.test_voice import (
    SYLLABLES, VAD_WINDOW, EnergyVad, chat_listener, take_when_ready, wait_for)
from medcare_reachy.bridge.voice import (
    ACK_ECHO_SECONDS, MUTE_TAIL_SECONDS, SAMPLE_RATE, STT_FILES, THINK_AFTER_SECONDS, THINKING_FORMS, DoneListener,
    without_ack, without_filler)

# ── by its words ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    # each phrase as SenseVoice wrote it back from the robot's voice through a simulated speaker and room (laptop,
    # the robot's model files), alone or after the 「嗯」's echo
    "我再想一下哦。", "我再想一下。", "嗯，我再想一下哦。", "嗯嗯，我再想一下哦。", "我在想一下哦。", "让我想一想哦。",
    "讓我想一想喔。", "嗯，让我想一想哦。", "我想想看哦。", "我想想看。", "嗯，我想想看哦。", "嗯", "嗯。",
    # what is left when the VAD missed its start ...
    "再想一下哦。", "想一下哦。", "一下哦。", "我想一想哦。", "想一想哦。", "我想一想。", "想想看哦。", "想看哦。",
    # ... or its end
    "我在想。", "让我想一想。", "让我想一。", "我想想。",
    # sound-alikes SenseVoice wrote with the patient talking at the same time
    "叫我想一想哦。", "我再享一下哦。",
])
def test_the_phrase_alone_is_dropped(text):
    assert without_filler(text) == ""


@pytest.mark.parametrize("text,words", [
    ("嗯，我再想一下哦，我不想活了。", "我不想活了。"), ("我再想一下，我不想活了。", "我不想活了。"),
    ("让我想一想哦，活了。", "活了。"), ("我想一想，我不想活了。", "我不想活了。"), ("让我想一想哦，我想死。", "我想死。"),
    ("我想想看，今天天天气怎么样？", "今天天天气怎么样？"), ("我再想一下哦 我吃过了", "我吃过了"),
])
def test_the_patients_words_after_it_stay(text, words):
    assert without_filler(text) == words


@pytest.mark.parametrize("text", [
    # a phrase with a word glued to it: kept whole (junk at worst, but no word of the patient's lost)
    "我在想我不想活了。", "让我想我不想活了。", "我想想看活了。", "我再想一下伯乐。", "我想想看我想死。",
    "我想一想死的事。", "想一下死了算了。", "我想想办法。",
    # the patient's own words
    "我想死。", "我不想活了。", "想想办法。", "一想到就难过。", "活了。", "我再想一下还是不去了", "一下就好。",
    # scraps shorter than three characters are never taken for the phrase
    "一下。", "想看。", "看哦。",
    # a phrase later in the sentence is the patient's (the server screens those words with and without it)
    "我不想，我再想一下，活了",
])
def test_nothing_but_a_phrase_at_the_start_is_taken(text):
    assert without_filler(text) == text


def test_every_thinking_phrase_in_the_manifest_is_recognised():
    manifest = load_manifest()
    for clip_id in ["ack", *manifest["variants"][THINKING]]:
        assert without_filler(manifest["clips"][clip_id]["zh-TW"]) == "", clip_id


def test_the_forms_are_the_phrases_and_their_heads_and_tails_of_three_or_more():
    assert {"我再想一下哦", "让我想一想哦", "我想想看哦", "我再想", "想一下", "一下哦", "想看哦", "我想想"} <= set(THINKING_FORMS)
    assert all(len(form) >= 3 for form in THINKING_FORMS)
    assert "一下" not in THINKING_FORMS and "想看" not in THINKING_FORMS


@pytest.mark.parametrize("text", ["嗯", "嗯，活了", "我嗯不想", "恩人来了", "mom", "hmmm, home"])
def test_a_lone_mm_is_handled_as_before(text):
    assert without_filler(text) == without_ack(text)


# ── said once the turn is handed over, and only for a turn Reachy acknowledged ──

def acknowledged_listener(tmp_path, words, phrase_seconds=1.2):
    """A chat listener (fake clock at 100 s) whose thinking phrase is recorded, said inline, `phrase_seconds` long."""
    voice, mic, vad, clock = chat_listener(tmp_path, words)
    said = []

    def think_aloud():
        said.append(clock[0])
        return phrase_seconds

    voice.think_aloud, voice._spawn = think_aloud, lambda task: task()
    return voice, mic, vad, clock, said


def hear(voice, mic, vad, clock, at, count):
    """Four VAD windows read at clock time `at`: one segment, its speech beginning 0.128 s before `at`."""
    assert wait_for(lambda: not mic.chunks and not vad.segments)
    clock[0] = at
    mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]
    assert wait_for(lambda: vad.fed == 4 * count and not vad.segments)


def handed_over_with_its_phrase(voice, clock, said, at):
    """Takes the turn at clock time `at`, then lets THINK_AFTER_SECONDS pass: the phrase is said."""
    clock[0] = at
    heard = take_when_ready(voice)
    clock[0] = at + THINK_AFTER_SECONDS
    assert wait_for(lambda: said and said[-1] == clock[0])
    return heard


def test_the_phrase_follows_the_hand_over_of_an_acknowledged_turn(tmp_path):
    voice, mic, vad, clock, said = acknowledged_listener(tmp_path, ["今天天气怎么样？"])
    voice.start()
    try:
        voice.set_active(True, mode="chat")
        hear(voice, mic, vad, clock, 100.0, 1)
        assert wait_for(lambda: voice.heard_pause() is not None)
        clock[0] = 100.2
        voice.note_ack(0.34)                          # the session's 「嗯」, at its tick
        clock[0] = 100.6
        text, metrics = take_when_ready(voice)
        assert text == "今天天气怎么样？" and metrics["ack_ms"] == 264
        clock[0] = 100.6 + THINK_AFTER_SECONDS - 0.01
        time.sleep(0.1)
        assert said == []                             # not at once: the patient may be going on
        clock[0] = 100.6 + THINK_AFTER_SECONDS
        assert wait_for(lambda: said == [clock[0]])   # then, as nobody spoke
        clock[0] = 105.0
        time.sleep(0.1)
        assert voice.take_utterance() is None and len(said) == 1   # once per turn
    finally:
        voice.stop()


def test_no_phrase_while_the_patient_is_speaking_again(tmp_path):
    """Words the patient goes on with right after the hand-over: Reachy doesn't start talking over them, and they get
    their own 「嗯」 (and phrase) once they pause."""
    voice, mic, vad, clock, said = acknowledged_listener(tmp_path, ["今天天气怎么样？", "要带伞吗？"])
    voice.start()
    try:
        voice.set_active(True, mode="chat")
        hear(voice, mic, vad, clock, 100.0, 1)
        voice.note_ack(0.34)
        clock[0] = 100.6
        assert take_when_ready(voice)[0] == "今天天气怎么样？"
        vad.speech, vad.emit = True, False            # the VAD hears them again
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]
        assert wait_for(lambda: vad.fed == 8 and voice.hears_speech())
        clock[0] = 100.6 + THINK_AFTER_SECONDS
        time.sleep(0.1)
        assert said == []
        vad.speech, vad.emit = False, True
        hear(voice, mic, vad, clock, 102.0, 3)        # their next words: a turn of its own
        voice.note_ack(0.34)
        handed_over_with_its_phrase(voice, clock, said, 102.4)
        assert said == [102.4 + THINK_AFTER_SECONDS]
    finally:
        voice.stop()


def test_no_phrase_once_reachy_is_answering(tmp_path):
    voice, mic, vad, clock, said = acknowledged_listener(tmp_path, ["你好"])
    voice.start()
    try:
        voice.set_active(True, mode="chat")
        hear(voice, mic, vad, clock, 100.0, 1)
        voice.note_ack(0.34)
        clock[0] = 100.6
        assert take_when_ready(voice)[0] == "你好"
        voice.hold()                                  # a fixed reply, already being said
        time.sleep(0.1)
        voice.release()
        clock[0] = 103.0
        time.sleep(0.1)
        assert said == []
    finally:
        voice.stop()


def test_no_phrase_for_a_turn_reachy_did_not_acknowledge(tmp_path):
    """The settings switch checkin_ack off (no 「嗯」 either), or words said while Reachy was answering."""
    voice, mic, vad, clock, said = acknowledged_listener(tmp_path, ["我早上去散步了"])
    voice.start()
    try:
        voice.set_active(True, mode="chat")
        hear(voice, mic, vad, clock, 100.0, 1)
        text, _ = take_when_ready(voice)
        clock[0] = 103.0
        time.sleep(0.1)
        assert text == "我早上去散步了" and said == []
    finally:
        voice.stop()


def test_a_cough_gets_the_mm_but_no_promise_of_an_answer(tmp_path):
    """The 「嗯」 answers the VAD's release, before decoding; a cough then decodes to nothing and no turn goes out.
    A phrase promising a reply would be followed by silence: the very complaint of 3 Oct."""
    voice, mic, vad, clock, said = acknowledged_listener(tmp_path, ["", "我早上去散步了"])
    voice.start()
    try:
        voice.set_active(True, mode="chat")
        hear(voice, mic, vad, clock, 100.0, 1)
        voice.note_ack(0.34)
        assert wait_for(lambda: voice.heard_pause() is None)   # the cough said nothing
        clock[0] = 102.0
        time.sleep(0.1)
        assert voice.take_utterance() is None and said == []
        hear(voice, mic, vad, clock, 103.0, 2)                  # long after: the patient, unacknowledged
        assert take_when_ready(voice)[0] == "我早上去散步了"
        clock[0] = 105.0
        time.sleep(0.1)
        assert said == []
    finally:
        voice.stop()


def test_its_echo_goes_and_the_patient_s_next_words_stay(tmp_path):
    voice, mic, vad, clock, said = acknowledged_listener(
        tmp_path, ["今天天气怎么样？", "我再想一下哦。", "我想一想哦，我不想活了。", "我再想一下哦，我想去公园。"])
    voice.start()
    try:
        voice.set_active(True, mode="chat")
        hear(voice, mic, vad, clock, 100.0, 1)
        voice.note_ack(0.34)
        assert handed_over_with_its_phrase(voice, clock, said, 100.6)[0] == "今天天气怎么样？"
        hear(voice, mic, vad, clock, 101.6, 2)                  # the phrase heard back: nothing else
        assert voice.take_utterance() is None and voice.heard_pause() is None
        hear(voice, mic, vad, clock, 102.4, 3)                  # its tail, then the patient's next words
        text, metrics = take_when_ready(voice)
        assert text == "我不想活了。" and metrics["echo_dropped"] == 2 and "ack_ms" not in metrics
        window_end = said[0] + 1.2 + MUTE_TAIL_SECONDS + ACK_ECHO_SECONDS
        hear(voice, mic, vad, clock, window_end + 0.128 + 0.05, 4)   # speech beginning after the window
        assert take_when_ready(voice)[0] == "我再想一下哦，我想去公园。"
        assert len(said) == 1                                   # unacknowledged turns: no phrase
    finally:
        voice.stop()


def test_without_the_clips_nothing_is_said_and_nothing_is_removed(tmp_path):
    voice, mic, vad, clock, said = acknowledged_listener(tmp_path, ["今天天气怎么样？", "我再想一下，我想去公园。"])
    voice.think_aloud = lambda: None                            # English, or no WAV
    voice.start()
    try:
        voice.set_active(True, mode="chat")
        hear(voice, mic, vad, clock, 100.0, 1)
        voice.note_ack(0.34)
        clock[0] = 100.6
        assert take_when_ready(voice)[0] == "今天天气怎么样？"
        clock[0] = 101.1
        time.sleep(0.1)
        hear(voice, mic, vad, clock, 101.4, 2)                  # past the 「嗯」's window
        assert take_when_ready(voice)[0] == "我再想一下，我想去公园。"
    finally:
        voice.stop()


def test_a_failing_phrase_costs_nothing_but_a_log_line(tmp_path, caplog):
    voice, mic, vad, clock, said = acknowledged_listener(tmp_path, ["今天天气怎么样？"])

    def broken():
        said.append("tried")
        raise OSError("speaker gone")

    voice.think_aloud = broken
    voice.start()
    try:
        voice.set_active(True, mode="chat")
        hear(voice, mic, vad, clock, 100.0, 1)
        voice.note_ack(0.34)
        assert take_when_ready(voice)[0] == "今天天气怎么样？"
        clock[0] += THINK_AFTER_SECONDS
        assert wait_for(lambda: said == ["tried"])
        time.sleep(0.05)
        assert "thinking phrase failed" in caplog.text
    finally:
        voice.stop()


def test_the_phrase_extends_a_mm_window_still_open_rather_than_replacing_it(tmp_path):
    voice, _, _, clock, _ = acknowledged_listener(tmp_path, [])
    clock[0] = 100.0
    voice.note_ack(0.34)
    clock[0] = 100.5
    voice.note_filler(1.2)
    assert voice._ack_echo == pytest.approx((100.0, 100.5 + 1.2 + MUTE_TAIL_SECONDS + ACK_ECHO_SECONDS))
    clock[0] = 110.0
    voice.note_filler(1.2)                                      # the old one closed long ago
    assert voice._ack_echo == pytest.approx((110.0, 110.0 + 1.2 + MUTE_TAIL_SECONDS + ACK_ECHO_SECONDS))


# ── 「我不想…活了」 with the 「嗯」 and the phrase, the session ticking, both echoes in the microphone ──

ECHO_MM, ECHO_PHRASE = 1.0, 2.0   # each echo's level, added to whatever the patient says meanwhile
WRITTEN = {ECHO_MM: "嗯，", ECHO_PHRASE: "我再想一下哦，"}


def conversation(models_dir, pause: float, mm_echo: bool, phrase_seconds=1.2, mm_seconds=0.34):
    """The patient says 「我不想」, pauses `pause` s, then 「活了」; audio arrives in 10 ms buffers in real time and piles
    up while SenseVoice decodes (0.6 s per s of speech). The session ticks every 0.2 s: it answers a new pause with
    a `mm_seconds` 「嗯」 and takes each turn the listener hands over; the listener then says the thinking phrase
    (`phrase_seconds`). One clip at a time, as robot.play_clip does. Each comes back through the microphone
    MUTE_TAIL_SECONDS after it starts (the 「嗯」 only when `mm_echo`), on top of what the patient says then. The
    decoder writes an echo as WRITTEN where it starts, and every syllable of the patient's it hears.
    Returns (the turns handed over, when each was, when each phrase was due)."""
    levels = {word: level for level, word in SYLLABLES.items()}
    audio = np.concatenate([np.zeros(int(0.3 * SAMPLE_RATE), np.float32)]
                           + [np.full(SAMPLE_RATE // 4, levels[word], np.float32) for word in "我不想"]
                           + [np.zeros(int(pause * SAMPLE_RATE), np.float32)]
                           + [np.full(SAMPLE_RATE // 4, levels[word], np.float32) for word in "活了"]
                           + [np.zeros(int(4.0 * SAMPLE_RATE), np.float32)])
    clock, lock = [100.0], threading.RLock()
    state = {"read": 0, "next_tick": 100.2, "acked": None, "done": False, "quiet": 0.0}
    echoes, turns, handed, phrases = [], [], [], []
    voice = None

    def play(seconds: float, level: float) -> float:
        """A clip handed to the speaker now: when it starts (after the one playing), its echo noted."""
        start = max(clock[0], state["quiet"])
        state["quiet"] = start + seconds
        if level:
            echoes.append((start + MUTE_TAIL_SECONDS, seconds, level))
        return start

    def session_ticks(until: float):
        while state["next_tick"] <= until:
            clock[0] = max(clock[0], state["next_tick"])
            state["next_tick"] += 0.2
            pause_ = voice.heard_pause()
            if pause_ is not None and pause_ != state["acked"]:   # the session: 「嗯」 once per pause
                state["acked"] = pause_
                play(mm_seconds, ECHO_MM if mm_echo else 0.0)
                voice.note_ack(mm_seconds)
            heard = voice.take_utterance_with_metrics()
            if heard:
                turns.append(heard[0])
                handed.append(clock[0])

    def think_aloud():
        with lock:
            phrases.append(clock[0])
            start = play(phrase_seconds, ECHO_PHRASE)
            return phrase_seconds + start - clock[0]   # noted when handed over, which waits for the 「嗯」 to end

    class Microphone:
        def get_audio(self):
            with lock:
                session_ticks(clock[0])
                if state["read"] >= len(audio):
                    state["done"] = True
                    return None
                buffer = audio[state["read"]:state["read"] + SAMPLE_RATE // 100].copy()
                heard_at = 100.0 + (state["read"] + np.arange(len(buffer))) / SAMPLE_RATE
                for start, seconds, level in echoes:   # heard when the audio was, not when it is read
                    buffer[(heard_at >= start) & (heard_at < start + seconds)] += level
                state["read"] += len(buffer)
                clock[0] = max(clock[0], 100.0 + state["read"] / SAMPLE_RATE)
                return buffer, SAMPLE_RATE

    def transcribe(segment):
        decoded = clock[0] + 0.6 * len(segment) / SAMPLE_RATE
        with lock:
            session_ticks(decoded)   # the session ticks on while SenseVoice decodes
        clock[0] = max(clock[0], decoded)
        heard, echoing, saying = [], None, ""
        for start in range(0, len(segment), SAMPLE_RATE // 20):
            peak = float(np.abs(segment[start:start + SAMPLE_RATE // 20]).max())
            echo = next((level for level in (ECHO_PHRASE, ECHO_MM) if peak > level * 0.9), None)
            if echo is not None and echo != echoing:
                heard.append(WRITTEN[echo])
            echoing = echo
            level = peak - (echo or 0.0)
            word = SYLLABLES[min(SYLLABLES, key=lambda value: abs(value - level))] if level > 0.01 else ""
            if word and word != saying:
                heard.append(word)
            saying = word
        return "".join(heard)

    for name in STT_FILES:
        (models_dir / name).write_bytes(b"x")
    voice = DoneListener(Microphone(), "zh-TW", models_dir, clock=lambda: clock[0],
                         load=lambda models_dir, language: (EnergyVad(), transcribe),
                         think_aloud=think_aloud, spawn=lambda task: task())
    voice.set_active(True, mode="chat")
    voice.start()
    try:
        assert wait_for(lambda: state["done"], timeout=20)
        with lock:
            session_ticks(clock[0] + 1.0)
    finally:
        voice.stop()
    return turns, handed, phrases


@pytest.mark.parametrize("mm_echo", [False, True], ids=["mm-unheard", "mm-heard-back"])
@pytest.mark.parametrize("pause", [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 3.0])
def test_a_sentence_resumed_after_a_pause_keeps_every_word_and_the_phrase_never_lands_inside_it(
        tmp_path, pause, mm_echo):
    turns, handed, phrases = conversation(tmp_path, pause, mm_echo)
    phrase = WRITTEN[ECHO_PHRASE]
    huo = 100.3 + 0.75 + pause   # when 「活了」 starts
    # Every syllable the patient said, in order (a 「嗯」 heard back inside a word is the server's to drop, as in
    # 0.5.2) ...
    assert re.sub(r"[\s，嗯]", "", "".join(turns).replace(phrase, "")) == "我不想活了", turns
    # ... the phrase came only after a turn was handed over, at most once per turn ...
    assert phrases and handed[0] + THINK_AFTER_SECONDS <= phrases[0] and len(phrases) <= len(turns)
    if len(turns) == 1:
        # ... and when the patient went on before the hand-over, their sentence went out whole, in one turn, with
        # no phrase in it.
        assert re.sub("[嗯，]", "", turns[0]) == "我不想 活了"
    else:
        # A pause longer than the hand-over: two turns, as with 0.5.2's 「嗯」 alone (the server screens each).
        assert turns[0] == "我不想" and handed[0] < huo + 0.25   # it went out before the VAD could hear 活
        if huo < phrases[0] - 0.25:   # the VAD heard them going on by then: Reachy said nothing over them
            assert phrases[0] > huo + 0.5 and phrase not in turns[1]
        elif phrase in turns[1]:
            # Only a patient going on just as the phrase starts talks over its echo, which then lands after their
            # first words (the server's keyword screen also checks the words without it).
            assert huo <= phrases[0] + MUTE_TAIL_SECONDS
        else:
            assert turns[1] == "活了"
