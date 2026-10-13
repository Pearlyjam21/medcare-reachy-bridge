"""Live voice check-ins on the robot (bridge/live.py, spec 2026-10-12 F1): the WebSocket to the server's GPT-Live
relay, the microphone frames, the playback queue and its jitter buffer, flush, Reachy's own lines through the same
queue, the playing/idle reports and how a chat ends; and the companion chat (spec 2026-10-12 companion session):
the voice detector and its speech signals, the spoken opening and a stop with a last line."""

import asyncio
import json
import sys
import time
import types

import numpy as np
import pytest

from medcare_reachy.bridge import __version__, live
from medcare_reachy.bridge.live import (
    BLOCK_SAMPLES, FRAME_SAMPLES, IDLE_QUIET_BLOCKS, JITTER_START_SECONDS, MAX_SECONDS,
    POLL_SECONDS, READY_TIMEOUT, RATE, SAY_TAIL_BLOCKS, SPEECH_ECHO_GUARD_SECONDS, SPEECH_SIGNAL_GAP_SECONDS, LiveChat, LiveOutcome, Player,
    SpeechStart)
from medcare_reachy.bridge.speech import chunks
from medcare_reachy.bridge.tests.fakes import FakeApp, FakeClock, FakeRobot, FakeSocket, FakeSpeaker, make_task
from medcare_reachy.bridge.voice import MIN_SILENCE_SECONDS, MIN_SPEECH_SECONDS, VAD_WINDOW

HELP = "我很担心您。请马上拨打一一九。"   # a 'say' line, as the server sends it (Simplified, for Matcha)
OPENING = "王奶奶，今天感觉怎么样？想跟我聊聊吗？"   # the companion relay's opening, in the voice's characters
LAST_LINE = "该吃药啰，我们等一下再聊。"             # a last line for stop(say=...), like the medicine line


def pcm(seconds, level=8000):
    """GPT-Live's voice as the server forwards it: PCM16 mono at 16 kHz, at a steady level."""
    return np.full(int(round(seconds * RATE)), level, dtype="<i2").tobytes()


def mic_chunk(first=0.25, second=-0.5, count=256):
    """What the SDK's get_audio_sample() gives: (count, 2) float32 at 16 kHz."""
    return np.column_stack([np.full(count, first, np.float32), np.full(count, second, np.float32)]), 16000


async def inline(fn, *args):
    return fn(*args)


async def no_wait(seconds):
    await asyncio.sleep(0)   # the speaker's clock runs as fast as the test lets it


async def until(condition, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0)


class Chat:
    """A LiveChat on fakes: the server's end of the socket, the robot and Reachy's voice."""

    def __init__(self, connect_error=None, live_audio=True, speaker=True, speech_start=None):
        self.clock = FakeClock()
        self.app = FakeApp(make_task([], reason="checkin", checkin=True), self.clock)
        self.robot = FakeRobot()
        self.robot.live_audio = live_audio
        self.speaker = FakeSpeaker(available=speaker)
        self.socket = FakeSocket(stamp=lambda: self.robot.blocks)
        self.connected = []
        self.connect_error = connect_error
        self.detector = speech_start
        self.live = LiveChat(self.app, self.robot, self.speaker, connect=self._connect, clock=self.clock.monotonic,
                             sleep=no_wait, run_blocking=inline, speech_start=speech_start)

    async def _connect(self, url, headers):
        self.connected.append((url, headers))
        if self.connect_error is not None:
            raise self.connect_error
        return self.socket

    def start(self):
        return asyncio.create_task(self.live.run("task-1", "zh-TW"))

    async def ready(self, conversation_id="c9", **fields):
        await until(lambda: self.socket.json_sent())
        self.socket.serve({"type": "ready", "conversation_id": conversation_id, **fields})
        await until(lambda: self.live.conversation_id == conversation_id)

    def sent(self):
        return self.socket.json_sent()


def run(scenario):
    return asyncio.run(scenario())


def play_out(player, now):
    """Every block the player gives until it has nothing more that may play at `now`."""
    blocks = []
    while True:
        block, kind = player.block(now)
        if kind is None:
            return blocks
        blocks.append((block, kind))


# ── the playback queue ───────────────────────────────────────────────────
def test_gpt_live_voice_waits_until_300_ms_are_queued():
    player = Player()
    player.add_live(pcm(0.28), now=0.0)
    assert player.block(0.0)[1] is None                  # 280 ms: still buffering
    player.add_live(pcm(0.02), now=0.01)
    block, kind = player.block(0.01)
    assert kind == "live" and len(block) == BLOCK_SAMPLES and np.all(block == 8000)
    assert player.target == JITTER_START_SECONDS and player.underruns == 0


def test_the_end_of_a_reply_plays_once_nothing_more_comes_for_300_ms():
    player = Player()
    player.add_live(pcm(0.1), now=0.0)
    assert player.block(0.29)[1] is None
    assert [kind for _, kind in play_out(player, 0.3)] == ["live"] * 5


def test_each_underrun_makes_the_buffer_200_ms_longer_up_to_1_s():
    player, now, targets = Player(), 0.0, []
    for _ in range(6):
        player.add_live(pcm(1.2), now)                   # more than any target: plays at once
        targets.append(round(player.target, 2))
        assert len(play_out(player, now)) == 60          # 1.2 s in 20 ms blocks, until it runs dry
        now += 0.4                                       # the next frame comes 0.4 s after it ran dry
    assert targets == [0.3, 0.5, 0.7, 0.9, 1.0, 1.0]
    assert player.underruns == 5


def test_a_new_reply_after_a_pause_is_no_underrun():
    player = Player()
    player.add_live(pcm(0.5), 0.0)
    play_out(player, 0.0)
    player.add_live(pcm(0.5), 2.0)                       # the patient spoke meanwhile
    assert player.target == JITTER_START_SECONDS and player.underruns == 0


def test_flush_drops_everything_and_the_voice_buffers_again():
    player = Player()
    player.add_live(pcm(0.5), 0.0)
    assert player.block(0.0)[1] == "live"
    player.flush()
    assert player.queued == 0 and player.block(0.0)[1] is None
    player.add_live(pcm(0.2), 0.0)
    assert player.block(0.0)[1] is None


def test_reachys_own_line_plays_at_once_and_after_what_came_before_it():
    player = Player()
    player.add_live(pcm(0.4), 0.0)
    player.add_say(np.full(800, 4000, np.int16), still=True)
    blocks = play_out(player, 0.0)
    assert [kind for _, kind in blocks] == ["live"] * 20 + ["still"] * 3
    assert np.all(blocks[-1][0][:160] == 4000) and np.all(blocks[-1][0][160:] == 0)
    alone = Player()
    alone.add_say(np.full(320, 4000, np.int16), still=False)
    assert alone.block(0.0)[1] == "say"                  # no buffering for Reachy's own voice


def test_finish_plays_what_is_left_without_waiting():
    player = Player()
    player.add_live(pcm(0.1), 0.0)
    assert player.block(0.0)[1] is None
    player.finish()
    assert len(play_out(player, 0.0)) == 5


def test_an_odd_byte_count_keeps_the_whole_samples():
    player = Player()
    player.add_live(b"\x10\x00\x20\x00\x30", 0.0)
    assert player.queued == 2


# ── a chat on a fake server ──────────────────────────────────────────────
def test_the_chat_connects_with_the_robot_key_and_says_start_first():
    async def scenario():
        chat = Chat()
        task = chat.start()
        await until(lambda: chat.sent())
        chat.socket.serve({"type": "end", "reason": "goodbye", "risk": False})
        return chat, await task

    chat, outcome = run(scenario)
    assert outcome == LiveOutcome("ended", "goodbye", None, False)
    assert chat.connected == [("ws://app:8001/api/device/conversations/live", {"Authorization": "Bearer rdv1.token"})]
    assert chat.sent()[0] == {"type": "start", "task_id": "task-1", "language": "zh-TW", "rate": 16000,
                              "bridge_version": __version__}
    assert chat.robot.calls[0] == "start_live_audio" and chat.robot.calls[-1] == "stop_live_audio"
    assert chat.socket.closed


def test_the_microphone_goes_out_in_40_ms_frames_of_its_first_channel_once_ready():
    async def scenario():
        chat = Chat()
        chat.robot.mic.extend([mic_chunk()] * 5)                  # heard before 'ready': nobody to send it to
        task = chat.start()
        await until(lambda: not chat.robot.mic)
        await chat.ready()
        chat.robot.mic.extend([mic_chunk()] * 10)                 # 2,560 samples: 4 frames
        await until(lambda: len(chat.socket.frames_sent()) == 4)
        chat.socket.serve({"type": "end", "reason": "goodbye", "risk": False})
        return chat, await task

    chat, outcome = run(scenario)
    frames = chat.socket.frames_sent()
    assert [len(frame) for frame in frames] == [FRAME_SAMPLES * 2] * 4
    assert all(np.all(np.frombuffer(frame, "<i2") == 8192) for frame in frames)   # 0.25 of the first channel
    assert chat.live.frames_sent == 4 and outcome.conversation_id == "c9"


def test_gpt_live_voice_plays_after_300_ms_in_20_ms_blocks():
    async def scenario():
        chat = Chat()
        task = chat.start()
        await chat.ready()
        chat.socket.serve(pcm(0.2))
        await until(lambda: chat.robot.blocks > 100)              # many speaker ticks later
        assert chat.robot.pushed == []                            # 200 ms: still buffering
        chat.socket.serve(pcm(0.1))
        await until(lambda: len(chat.robot.pushed) == 15)
        chat.socket.serve({"type": "end", "reason": "goodbye", "risk": False})
        return chat, await task

    chat, outcome = run(scenario)
    assert all(len(block) == BLOCK_SAMPLES for block in chat.robot.pushed)
    assert len(chat.robot.heard()) == 15 * BLOCK_SAMPLES and np.all(chat.robot.heard() == 8000)
    assert outcome == LiveOutcome("ended", "goodbye", "c9", False)


def test_reachy_reports_playing_at_its_first_sound_and_idle_after_0_6_s_of_quiet():
    async def scenario():
        chat = Chat()
        task = chat.start()
        await chat.ready()
        chat.socket.serve(pcm(0.4))
        await until(lambda: {"type": "idle"} in chat.sent())
        chat.socket.serve({"type": "end", "reason": "goodbye", "risk": False})
        await task
        return chat

    chat = run(scenario)
    sent = chat.sent()
    assert sent.count({"type": "playing"}) == 1 and sent.count({"type": "idle"}) == 1
    assert chat.socket.stamp_of({"type": "playing"}) == chat.robot.pushed_at[0]       # the first block with sound
    assert chat.socket.stamp_of({"type": "idle"}) == chat.robot.pushed_at[-1] + IDLE_QUIET_BLOCKS
    assert len(chat.robot.pushed) == 20
    assert [call for call in chat.robot.calls if call.startswith("gesture")] == ["gesture:speak", "gesture:None"]


def test_flush_drops_the_queue_and_clears_the_speaker_at_once():
    async def scenario():
        chat = Chat()
        task = chat.start()
        await chat.ready()
        chat.socket.serve(pcm(2.0))
        await until(lambda: len(chat.robot.pushed) >= 5)
        chat.socket.serve({"type": "flush"})
        await until(lambda: {"type": "idle"} in chat.sent())
        blocks = chat.robot.blocks
        await until(lambda: chat.robot.blocks > blocks + 100)
        chat.socket.serve({"type": "end", "reason": "goodbye", "risk": False})
        await task
        return chat

    chat = run(scenario)
    assert "clear_live_audio" in chat.robot.calls
    assert len(chat.robot.pushed) < 100                                    # far less than the 2 s queued
    assert max(chat.robot.pushed_at) <= chat.robot.cleared_at              # nothing played after the clear
    assert chat.sent().index({"type": "playing"}) < chat.sent().index({"type": "idle"})
    assert chat.live.player.queued == 0


def test_reachys_own_line_follows_the_voice_before_it_and_gpt_live_never_talks_over_it():
    async def scenario():
        chat = Chat()
        task = chat.start()
        await chat.ready()
        chat.socket.serve(pcm(0.4))
        chat.socket.serve({"type": "say", "speech_text": HELP, "reason": "risk"})
        chat.socket.serve(pcm(0.4, level=3000))                    # GPT-Live after the line: never played
        chat.socket.serve({"type": "end", "reason": "risk", "risk": True})
        return chat, await task

    chat, outcome = run(scenario)
    assert outcome == LiveOutcome("ended", "risk", "c9", True)
    assert chat.speaker.synthesised == chunks(HELP)                        # one chunk at a time
    heard = chat.robot.heard()
    heard = heard[heard != 0]
    line = len(HELP) * RATE // 100                                         # FakeSpeaker: 10 ms per character
    assert list(heard) == [8000] * 6400 + [8192] * line                    # all of it, before run() returned
    assert chat.live.dropped_audio == 1
    gestures = [call for call in chat.robot.calls if call.startswith("gesture")]
    assert gestures[:2] == ["gesture:speak", "gesture:None"]               # still for the help line
    assert "gesture:speak" not in gestures[1:]


def test_the_microphone_sends_silence_while_reachy_says_its_own_line_and_just_after():
    async def scenario():
        chat = Chat()
        task = chat.start()
        await chat.ready()
        chat.socket.serve({"type": "say", "speech_text": "好的。", "reason": "still_there"})
        await until(lambda: chat.robot.pushed)                     # the line is playing
        chat.robot.mic.extend([mic_chunk()] * 5)
        await until(lambda: len(chat.socket.frames_sent()) == 2)
        silent = list(chat.socket.frames_sent())
        last = chat.robot.pushed_at[-1]
        await until(lambda: chat.robot.blocks > last + SAY_TAIL_BLOCKS)
        chat.robot.mic.extend([mic_chunk()] * 5)
        await until(lambda: len(chat.socket.frames_sent()) == 4)
        chat.socket.serve({"type": "end", "reason": "goodbye", "risk": False})
        await task
        return chat, silent

    chat, silent = run(scenario)
    assert all(not np.any(np.frombuffer(frame, "<i2")) for frame in silent)
    assert all(np.all(np.frombuffer(frame, "<i2") == 8192) for frame in chat.socket.frames_sent()[2:])
    assert [call for call in chat.robot.calls if call.startswith("gesture")][:1] == ["gesture:speak"]


@pytest.mark.parametrize("message, expected", [
    ({"type": "fallback", "reason": "live_disabled", "conversation_id": None},
     LiveOutcome("fallback", "live_disabled", None)),
    ({"type": "end", "reason": "silence", "risk": False}, LiveOutcome("ended", "silence", None)),
])
def test_the_server_can_end_or_fall_back_before_ready(message, expected):
    async def scenario():
        chat = Chat()
        task = chat.start()
        await until(lambda: chat.sent())
        chat.socket.serve(message)
        return chat, await task

    chat, outcome = run(scenario)
    assert outcome == expected and chat.socket.frames_sent() == []


@pytest.mark.parametrize("conversation_id", ["c9", None])
def test_a_fallback_after_ready_continues_the_same_conversation(conversation_id):
    async def scenario():
        chat = Chat()
        task = chat.start()
        await chat.ready()
        chat.socket.serve({"type": "fallback", "reason": "live_ended", "conversation_id": conversation_id})
        return chat, await task

    chat, outcome = run(scenario)
    assert outcome == LiveOutcome("fallback", "live_ended", "c9")
    assert chat.robot.calls[-1] == "stop_live_audio"


def test_a_dropped_connection_plays_what_is_queued_and_falls_back_to_a_new_chat():
    async def scenario():
        chat = Chat()
        task = chat.start()
        await chat.ready()
        chat.socket.serve(pcm(0.1))                                # less than the buffer: waits
        chat.socket.drop(1006)
        return chat, await task

    chat, outcome = run(scenario)
    assert outcome == LiveOutcome("fallback", "disconnected", None)        # the server ends c9 on the drop (C2)
    assert len(chat.robot.heard()) == 5 * BLOCK_SAMPLES                    # played out: nothing more could come


@pytest.mark.parametrize("code, reason", [(4401, "not_authorised"), (4403, "consent_required"), (4409, "live_busy")])
def test_a_refused_chat_falls_back_with_the_servers_close_code(code, reason):
    async def scenario():
        chat = Chat()
        task = chat.start()
        await until(lambda: chat.sent())
        chat.socket.drop(code)
        return await task

    assert run(scenario) == LiveOutcome("fallback", reason, None)


def test_a_socket_that_does_not_open_falls_back_at_once():
    async def scenario():
        chat = Chat(connect_error=OSError("connection refused"))
        return chat, await chat.start()

    chat, outcome = run(scenario)
    assert outcome == LiveOutcome("fallback", "connect_failed", None)
    assert chat.robot.calls == ["start_live_audio", "stop_live_audio"]


def test_without_a_speaker_the_chat_falls_back_without_connecting():
    async def scenario():
        chat = Chat(live_audio=False)
        return chat, await chat.start()

    chat, outcome = run(scenario)
    assert outcome == LiveOutcome("fallback", "no_audio", None)
    assert chat.connected == [] and chat.robot.calls == ["start_live_audio"]


def test_no_answer_to_start_falls_back_and_tells_the_server():
    async def scenario():
        chat = Chat()
        task = chat.start()
        await until(lambda: chat.sent())
        chat.clock.t += READY_TIMEOUT
        return chat, await task

    chat, outcome = run(scenario)
    assert outcome == LiveOutcome("fallback", "no_answer", None)
    assert chat.sent()[-1] == {"type": "end", "reason": "no_answer"} and chat.socket.closed


def test_stop_ends_the_chat_at_once_and_drops_what_is_queued():
    async def scenario():
        chat = Chat()
        task = chat.start()
        await chat.ready()
        chat.socket.serve(pcm(2.0))
        await until(lambda: len(chat.robot.pushed) >= 5)
        chat.live.stop("stop_all")
        return chat, await task

    chat, outcome = run(scenario)
    assert outcome == LiveOutcome("ended", "stop_all", "c9")
    assert chat.sent()[-1] == {"type": "end", "reason": "stop_all"}
    assert {"type": "idle"} not in chat.sent()                             # the robot said why itself
    assert "clear_live_audio" in chat.robot.calls and len(chat.robot.pushed) < 100
    assert chat.robot.calls[-2:] == ["gesture:None", "stop_live_audio"]


def stop_in_the_drain(chat, reason):
    """Make stop() arrive once the chat has begun draining what is queued."""
    drain = chat.live._drain

    async def stopping():
        chat.live.stop(reason)
        await drain()

    chat.live._drain = stopping


@pytest.mark.parametrize("message, expected", [
    ({"type": "fallback", "reason": "live_ended", "conversation_id": "c9"},
     LiveOutcome("ended", "stop_all", "c9")),
    ({"type": "end", "reason": "goodbye", "risk": False}, LiveOutcome("ended", "stop_all", "c9")),
    ({"type": "end", "reason": "risk", "risk": True}, LiveOutcome("ended", "stop_all", "c9", True)),
])
def test_a_stop_during_the_final_drain_still_ends_the_chat(message, expected):
    async def scenario():
        chat = Chat()
        stop_in_the_drain(chat, "stop_all")
        task = chat.start()
        await chat.ready()
        chat.socket.serve(pcm(0.1))
        chat.socket.serve(message)
        return chat, await task

    chat, outcome = run(scenario)
    assert outcome == expected
    assert chat.robot.calls[-1] == "stop_live_audio" and chat.socket.closed


def test_a_speaker_that_cannot_be_cleared_never_breaks_a_flush():
    async def scenario():
        chat = Chat()
        chat.robot.clear_error = RuntimeError("clear_player failed")
        task = chat.start()
        await chat.ready()
        chat.socket.serve(pcm(2.0))
        await until(lambda: len(chat.robot.pushed) >= 5)
        chat.socket.serve({"type": "flush"})
        await until(lambda: {"type": "idle"} in chat.sent())
        chat.socket.serve({"type": "end", "reason": "goodbye", "risk": False})
        return chat, await task

    chat, outcome = run(scenario)
    assert outcome == LiveOutcome("ended", "goodbye", "c9")
    assert chat.live.player.queued == 0 and chat.robot.calls[-1] == "stop_live_audio"


def test_a_speaker_that_cannot_be_cleared_never_breaks_a_robot_ending():
    async def scenario():
        chat = Chat()
        task = chat.start()
        await chat.ready()
        chat.robot.push_error = RuntimeError("push_buffer failed")
        chat.robot.clear_error = RuntimeError("clear_player failed")
        return chat, await task

    chat, outcome = run(scenario)
    assert outcome == LiveOutcome("fallback", "robot_error", None)
    assert chat.sent()[-1] == {"type": "end", "reason": "robot_error"}
    assert "clear_live_audio" in chat.robot.calls and chat.robot.calls[-1] == "stop_live_audio"


def test_a_chat_the_server_never_ends_stops_after_max_seconds():
    async def scenario():
        chat = Chat()
        task = chat.start()
        await chat.ready()
        chat.clock.t += MAX_SECONDS
        return chat, await task

    chat, outcome = run(scenario)
    assert outcome == LiveOutcome("ended", "max_length", "c9")
    assert chat.sent()[-1] == {"type": "end", "reason": "max_length"}


def test_a_broken_speaker_ends_the_live_chat_and_falls_back_to_a_new_one():
    async def scenario():
        chat = Chat()
        task = chat.start()
        await chat.ready()
        chat.robot.push_error = RuntimeError("push_buffer failed")
        return chat, await task

    chat, outcome = run(scenario)
    assert outcome == LiveOutcome("fallback", "robot_error", None)        # the server ends c9 on the robot's end
    assert chat.sent()[-1] == {"type": "end", "reason": "robot_error"}


def test_messages_it_does_not_know_are_ignored():
    async def scenario():
        chat = Chat()
        task = chat.start()
        await chat.ready()
        chat.socket.serve("not json")
        chat.socket.serve({"type": "state", "reachy": "speaking"})
        chat.socket.serve({"type": "say", "speech_text": "", "reason": "risk"})   # nothing to say
        chat.socket.serve({"type": "end", "reason": "goodbye", "risk": False})
        return chat, await task

    chat, outcome = run(scenario)
    assert outcome == LiveOutcome("ended", "goodbye", "c9", False)
    assert chat.speaker.synthesised == []


def test_without_gestures_reachy_keeps_still():
    async def scenario():
        chat = Chat()
        chat.live.gestures = False
        task = chat.start()
        await chat.ready()
        chat.socket.serve(pcm(0.4))
        await until(lambda: {"type": "idle"} in chat.sent())
        chat.socket.serve({"type": "end", "reason": "goodbye", "risk": False})
        await task
        return chat

    chat = run(scenario)
    assert not [call for call in chat.robot.calls if call.startswith("gesture")]


# ── the companion chat: speech signals, the opening, a last line ─────────
class FakeSpeechStart:
    """SpeechStart stand-in: speech begins at a frame at least half full scale after a quieter one (the VAD's onset).
    `fed` keeps the level of every frame it was given, as the chat gave it."""

    def __init__(self):
        self.fed = []
        self._speech = False

    @property
    def speaking(self):
        return self._speech

    def accept(self, samples):
        assert samples.dtype == np.float32 and len(samples) == FRAME_SAMPLES
        level = round(float(np.max(np.abs(samples))), 2)
        self.fed.append(level)
        began = level >= 0.5 and not self._speech
        self._speech = level >= 0.5
        return began


async def hear(chat, level):
    """The microphone hears one 40 ms frame at `level` (its first channel), and the chat has taken it in."""
    count = len(chat.detector.fed)
    chat.robot.mic.append(mic_chunk(first=level, count=FRAME_SAMPLES))
    await until(lambda: len(chat.detector.fed) == count + 1)


def said(text):
    """How many samples FakeSpeaker makes of a line said through the chat's say path (10 ms per character)."""
    return sum(len(piece) for piece in chunks(text)) * RATE // 100


def fake_sherpa(monkeypatch):
    """sherpa_onnx as SpeechStart uses it (it is not installed here); returns the list of VoiceActivityDetectors built.
    `speech[i]` is what is_speech_detected() answers after window i; every fourth window ends a segment."""
    built = []

    class VadModelConfig:
        def __init__(self):
            self.silero_vad = types.SimpleNamespace()
            self.sample_rate = None

    class VoiceActivityDetector:
        def __init__(self, config, buffer_size_in_seconds):
            self.config, self.buffer = config, buffer_size_in_seconds
            self.windows, self.speech, self.segments, self.popped = [], [], [], 0
            built.append(self)

        def accept_waveform(self, samples):
            self.windows.append(np.array(samples))
            if len(self.windows) % 4 == 0:
                self.segments.append(types.SimpleNamespace(samples=np.zeros(VAD_WINDOW, np.float32)))

        def is_speech_detected(self):
            index = len(self.windows) - 1
            return index < len(self.speech) and self.speech[index]

        def empty(self):
            return not self.segments

        def pop(self):
            self.segments.pop(0)
            self.popped += 1

    module = types.ModuleType("sherpa_onnx")
    module.VadModelConfig, module.VoiceActivityDetector = VadModelConfig, VoiceActivityDetector
    monkeypatch.setitem(sys.modules, "sherpa_onnx", module)
    return built


def test_the_voice_detector_is_the_listeners_vad_and_says_where_speech_begins(tmp_path, monkeypatch):
    built = fake_sherpa(monkeypatch)
    (tmp_path / "silero_vad.onnx").write_bytes(b"x")
    detector = SpeechStart(tmp_path)
    vad, = built
    silero = vad.config.silero_vad
    assert silero.model == str(tmp_path / "silero_vad.onnx")
    assert (silero.min_silence_duration, silero.min_speech_duration, silero.max_speech_duration) == (
        MIN_SILENCE_SECONDS, MIN_SPEECH_SECONDS, 8.0)
    assert (vad.config.sample_rate, vad.buffer) == (16000, 30)
    vad.speech = [False, False, True, True, True, False, True]       # after each window: speech begins at 2 and 6
    frame = np.full(FRAME_SAMPLES, 0.1, np.float32)                   # 640 samples: 1.25 of the VAD's windows
    assert [detector.accept(frame) for _ in range(6)] == [False, False, True, False, False, True]
    assert len(vad.windows) == 7 and all(len(window) == VAD_WINDOW for window in vad.windows)
    assert np.all(np.concatenate(vad.windows) == np.float32(0.1))
    assert vad.popped == 1 and vad.segments == []                     # its finished segments never pile up


def test_the_opening_is_said_in_reachys_own_voice_before_anything_else():
    async def scenario():
        chat = Chat(speech_start=FakeSpeechStart())
        task = chat.start()
        await chat.ready(speech_text=OPENING)
        for _ in range(3):
            await hear(chat, 0.75)                                    # the patient talks over the greeting
        await until(lambda: len(chat.robot.heard()) >= said(OPENING))
        chat.socket.serve({"type": "end", "reason": "goodbye", "risk": False})
        return chat, await task

    chat, outcome = run(scenario)
    assert outcome == LiveOutcome("ended", "goodbye", "c9", False)
    assert chat.speaker.synthesised == chunks(OPENING)
    heard = chat.robot.heard()
    assert list(heard[heard != 0]) == [8192] * said(OPENING)          # the opening, and nothing else
    assert all(not np.any(np.frombuffer(frame, "<i2")) for frame in chat.socket.frames_sent())
    assert chat.detector.fed == [0.0] * 3                             # the detector hears what the server gets
    assert {"type": "speech"} not in chat.sent() and chat.live.last_speech_at is None
    assert [call for call in chat.robot.calls if call.startswith("gesture")][:1] == ["gesture:speak"]


def test_speech_goes_to_the_server_when_the_patient_begins_to_speak_never_on_reachys_own_sound():
    async def scenario():
        chat = Chat(speech_start=FakeSpeechStart())
        task = chat.start()
        await chat.ready()
        seen = []

        async def onset():
            await hear(chat, 0.75)
            seen.append((chat.sent().count({"type": "speech"}), chat.live.last_speech_at))
            await hear(chat, 0.25)

        await onset()                                                 # 1000.0: the patient begins to speak
        chat.clock.t += SPEECH_SIGNAL_GAP_SECONDS                     # (the gap never holds a signal back below)
        chat.socket.serve(pcm(0.4))                                   # GPT-Live answers: Reachy makes sound
        await until(lambda: chat.robot.pushed)
        await onset()                                                 # 1001.0: its own voice, heard back
        await until(lambda: len(chat.robot.pushed) == 20)
        chat.clock.t += SPEECH_ECHO_GUARD_SECONDS - 0.05
        await onset()                                                 # 1001.7: just under the guard: still its echo
        chat.clock.t += 0.05
        await onset()                                                 # 1001.75: at the guard: the patient again
        chat.socket.serve({"type": "end", "reason": "goodbye", "risk": False})
        await task
        return chat, seen

    chat, seen = run(scenario)
    assert (SPEECH_ECHO_GUARD_SECONDS, SPEECH_SIGNAL_GAP_SECONDS) == (0.75, 1.0)
    assert seen[:3] == [(1, 1000.0)] * 3
    assert seen[3] == (2, pytest.approx(1001.0 + SPEECH_ECHO_GUARD_SECONDS))
    assert chat.detector.fed == [0.75, 0.25] * 4                      # it hears every frame: only the signal waits
    assert len(chat.socket.frames_sent()) == 8


def test_a_failing_voice_detector_is_logged_once_and_the_chat_goes_on(caplog):
    class Failing(FakeSpeechStart):
        def accept(self, samples):
            super().accept(samples)
            raise RuntimeError("detector broke")

    async def scenario():
        chat = Chat(speech_start=Failing())
        task = chat.start()
        await chat.ready()
        await hear(chat, 0.75)
        chat.robot.mic.append(mic_chunk(first=0.0, count=FRAME_SAMPLES))   # the chat still sends what the mic hears
        await until(lambda: len(chat.socket.frames_sent()) == 2)
        chat.socket.serve({"type": "end", "reason": "goodbye", "risk": False})
        return chat, await task

    with caplog.at_level("WARNING"):
        chat, outcome = run(scenario)
    assert outcome == LiveOutcome("ended", "goodbye", "c9", False)    # not a robot_error fallback
    assert chat.detector.fed == [0.75]                                # not called again after the failure
    assert {"type": "speech"} not in chat.sent() and len(chat.socket.frames_sent()) == 2
    assert [r.levelname for r in caplog.records if "voice detector" in r.getMessage()] == ["WARNING"]


def test_speech_goes_to_the_server_at_most_once_a_second():
    async def scenario():
        chat = Chat(speech_start=FakeSpeechStart())
        task = chat.start()
        await chat.ready()
        await hear(chat, 0.75)                                        # at 1000.0: signalled
        await hear(chat, 0.25)
        chat.clock.t += 0.5
        await hear(chat, 0.75)                                        # at 1000.5: heard, not signalled
        middle = (chat.sent().count({"type": "speech"}), chat.live.last_speech_at)
        await hear(chat, 0.25)
        chat.clock.t += 0.5
        await hear(chat, 0.75)                                        # at 1001.0: a second after the signal
        chat.socket.serve({"type": "end", "reason": "goodbye", "risk": False})
        await task
        return chat, middle

    chat, middle = run(scenario)
    assert middle == (1, 1000.5)
    assert chat.sent().count({"type": "speech"}) == 2 and chat.live.last_speech_at == 1001.0


def test_a_companion_chat_has_no_length_limit_on_the_robot():
    async def scenario():
        chat = Chat()
        task = chat.start()
        await chat.ready(speech_text=OPENING)
        chat.clock.t += MAX_SECONDS
        await asyncio.sleep(3 * POLL_SECONDS)                         # the chat looks at its limits meanwhile
        going_on = not task.done()
        chat.socket.serve({"type": "end", "reason": "goodbye", "risk": False})
        return chat, going_on, await task

    chat, going_on, outcome = run(scenario)
    assert going_on and outcome == LiveOutcome("ended", "goodbye", "c9", False)
    assert {"type": "end", "reason": "max_length"} not in chat.sent()


def test_stop_with_a_last_line_lets_reachy_finish_its_sentence_then_says_the_line():
    async def scenario():
        chat = Chat()
        task = chat.start()
        await chat.ready()
        chat.socket.serve(pcm(0.4))                                   # GPT-Live's sentence
        await until(lambda: len(chat.robot.pushed) >= 5)
        chat.live.stop("medicine", say=LAST_LINE)
        chat.socket.serve(pcm(0.4, level=3000))                       # after the stop: never played
        return chat, await task

    chat, outcome = run(scenario)
    assert outcome == LiveOutcome("ended", "medicine", "c9")
    assert chat.speaker.synthesised == chunks(LAST_LINE)
    heard = chat.robot.heard()
    assert list(heard[heard != 0]) == [8000] * 6400 + [8192] * said(LAST_LINE)   # all of both, in that order
    assert [message for message in chat.sent() if message["type"] == "end"] == [{"type": "end", "reason": "medicine"}]
    assert chat.socket.stamp_of({"type": "end", "reason": "medicine"}) < chat.robot.pushed_at[20]   # sent at once
    assert "clear_live_audio" not in chat.robot.calls
    assert chat.robot.calls[-2:] == ["gesture:None", "stop_live_audio"]


# ── the real WebSocket client (websockets) ───────────────────────────────
def test_the_websocket_sends_the_robot_key_and_reports_the_servers_close_code():
    from websockets.asyncio.server import serve

    async def scenario():
        seen = {}

        async def handler(connection):
            seen["auth"] = connection.request.headers["Authorization"]
            seen["first"] = await connection.recv()
            await connection.send(json.dumps({"type": "ready", "conversation_id": "c9"}))
            await connection.send(b"\x01\x00\x02\x00")
            await connection.close(4409, "busy")

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            socket = await live.connect_websocket(f"ws://127.0.0.1:{port}/api/device/conversations/live",
                                                  {"Authorization": "Bearer rdv1.token"})
            sent = await socket.send(json.dumps({"type": "start"}))
            received = [await socket.receive(), await socket.receive(), await socket.receive()]
            sent_after = await socket.send(b"\x00\x00")
            await socket.close()
        return seen, sent, received, sent_after, socket.close_code

    seen, sent, received, sent_after, code = asyncio.run(scenario())
    assert seen == {"auth": "Bearer rdv1.token", "first": '{"type": "start"}'}
    assert sent is True and sent_after is False
    assert received == ['{"type": "ready", "conversation_id": "c9"}', b"\x01\x00\x02\x00", None]
    assert code == 4409
