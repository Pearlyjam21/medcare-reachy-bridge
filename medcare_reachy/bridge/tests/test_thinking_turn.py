"""A check-in turn end to end, with the real listener and clip player in the slot session: 「嗯」 the moment the patient
pauses, the thinking phrase once their words are in and on their way to the server, then the reply.

On 3 Oct 2026 a patient asked about the weather and heard a 0.34 s 「嗯」, then ~6 s of silence. The phrase fills that
silence without ever playing over a turn the patient is still saying (test_filler_echo.py).
"""

import asyncio
import threading
import time
import wave
from pathlib import Path

import numpy as np

from medcare_reachy.bridge.clips import ClipPlayer
from medcare_reachy.bridge.tests.fakes import Harness, make_task
from medcare_reachy.bridge.tests.test_voice import VAD_WINDOW, FakeVad, Mic, wait_for
from medcare_reachy.bridge.voice import STT_FILES, THINK_AFTER_SECONDS, DoneListener

PHRASES = ("thinking_1", "thinking_2", "thinking_3")


class Speaker:
    """The robot's speaker as ClipPlayer sees it: each clip goes into the robot's call log."""

    def __init__(self, calls):
        self.calls = calls

    def play_clip(self, path, wait=True):
        self.calls.append(f"clip:{Path(path).stem}")
        return True


def write_clip(path: Path, seconds: float, rate=22050):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.zeros(int(seconds * rate), "<i2").tobytes())


def chatting(tmp_path, words, gate=None):
    """A conversation-only task in CHECKIN, past its opening line, listening with a real DoneListener (fake VAD and
    decoder: `words`, one per segment, each decoded once `gate` is set when given) and a real ClipPlayer (「嗯」 and
    three phrases). The server takes a second to answer each turn."""
    h = Harness(make_task([], reason="checkin", checkin=True), voice=True, speaker=True)
    clip_dir = tmp_path / "clips"
    write_clip(clip_dir / "zh-TW" / "ack.wav", 0.34)
    for clip_id in PHRASES:
        write_clip(clip_dir / "zh-TW" / f"{clip_id}.wav", 1.2)
    clips = ClipPlayer(Speaker(h.robot.calls), clip_dir, "zh-TW")
    models = tmp_path / "stt"
    models.mkdir()
    for name in STT_FILES:
        (models / name).write_bytes(b"x")
    mic, vad, words = Mic(), FakeVad(), iter(words)

    def transcribe(segment):
        assert gate is None or gate.wait(2)
        return next(words)

    voice = DoneListener(mic, "zh-TW", models, clock=h.clock.monotonic,
                         load=lambda models_dir, language: (vad, transcribe),
                         think_aloud=clips.think_aloud, spawn=lambda task: task())
    h.slot.voice = h.voice = voice
    h.slot.clips = h.clips = clips
    answer = h.app.conversation_turn

    async def server_thinking(conversation_id, text, metrics=None):
        h.robot.calls.append(f"server:{text}")
        h.clock.t += 1.0                                # the model, the risk check, the round trip
        await asyncio.sleep(0.3)                        # the listener works meanwhile (its own thread)
        return await answer(conversation_id, text, metrics)

    h.app.conversation_turn = server_thinking
    voice.start()
    h.find_patient()
    h.tick()                                            # the opening line
    assert h.slot.state == "CHECKIN" and h.speaker.said[-1] == "今天感觉怎么样？"
    h.tick(0.5)                                         # its echo tail has passed: listening
    return h, voice, mic, vad


def patient_speaks(h, mic, vad, count):
    """One segment (FakeVad: four windows), released by the VAD and decoded."""
    mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]
    assert wait_for(lambda: vad.fed == 4 * count and not vad.segments)


def test_mm_then_the_thinking_phrase_while_the_server_answers_then_the_reply(tmp_path):
    gate = threading.Event()
    h, voice, mic, vad = chatting(tmp_path, ["我早上去散步了"], gate)
    try:
        patient_speaks(h, mic, vad, 1)                  # released by the VAD: SenseVoice is decoding it
        h.tick(0.2)
        assert h.robot.calls[-2:] == ["clip:ack", "gesture:think"] and not h.app.named("conversation_turn")
        gate.set()
        assert wait_for(lambda: voice._segments)        # decoded
        h.tick(0.2)
        said = h.robot.calls[h.robot.calls.index("clip:ack"):]
        phrase = next(call for call in said if call.startswith("clip:thinking_"))
        assert said == ["clip:ack", "gesture:think", "server:我早上去散步了", phrase, "gesture:speak", "say:真好",
                        "gesture:None"]
        assert phrase[len("clip:"):] in PHRASES
        assert "ack_ms" in h.app.named("conversation_turn")[-1][3]
    finally:
        voice.stop()


def test_each_turn_gets_its_own_phrase_and_they_take_turns(tmp_path):
    h, voice, mic, vad = chatting(tmp_path, ["我早上去散步了", "要带伞吗？", "好的"])
    try:
        for count in (1, 2, 3):
            patient_speaks(h, mic, vad, count)
            assert wait_for(lambda: voice._segments)
            h.tick(0.2)
            h.tick(0.5)                                 # the reply's echo tail
        phrases = [call[len("clip:"):] for call in h.robot.calls if call.startswith("clip:thinking_")]
        assert sorted(phrases) == sorted(PHRASES)       # one each, in a shuffled order
        assert [call for call in h.robot.calls if call.startswith("clip:")].count("clip:ack") == 3
    finally:
        voice.stop()


def test_a_cough_gets_the_mm_and_nothing_more(tmp_path):
    """The 「嗯」 comes before decoding; a cough then decodes to nothing: no turn, and no phrase promising a reply."""
    gate = threading.Event()
    h, voice, mic, vad = chatting(tmp_path, [""], gate)
    try:
        patient_speaks(h, mic, vad, 1)
        h.tick(0.2)
        gate.set()
        assert wait_for(lambda: voice.heard_pause() is None)
        h.tick(THINK_AFTER_SECONDS + 1)
        time.sleep(0.1)
        h.tick(1)
        assert [call for call in h.robot.calls if call.startswith(("clip:", "server:"))] == ["clip:ack"]
        assert not h.app.named("conversation_turn")
    finally:
        voice.stop()


def test_with_checkin_ack_off_there_is_neither_mm_nor_phrase(tmp_path):
    h, voice, mic, vad = chatting(tmp_path, ["我早上去散步了"])
    h.slot.ack = False
    try:
        patient_speaks(h, mic, vad, 1)
        assert wait_for(lambda: voice._segments)
        h.tick(0.2)
        assert h.app.named("conversation_turn") and h.speaker.said[-1] == "真好"
        assert not [call for call in h.robot.calls if call.startswith("clip:")]
    finally:
        voice.stop()
