"""Live check-ins (spec 2026-10-12 F2): when the task says live, CHECKIN runs the live chat (bridge/live.py) with the
speech listener suspended, and today's chat takes over when the live chat falls back."""

import asyncio
import logging
import sys
import types
from pathlib import Path

import pytest

from medcare_reachy.bridge import live as live_module
from medcare_reachy.bridge import session as session_module
from medcare_reachy.bridge.live import LiveChat, LiveOutcome
from medcare_reachy.bridge.session import (
    CAMERA_LOST_SECONDS, CHECKIN_GREETING, COMPANION_CAPTURE_FPS, MEDICINE_LINE, REST_AFTER_SECONDS, REST_LINE,
    open_live_chat)
from medcare_reachy.bridge.tests.fakes import FakeSocket, FakeSpeaker, Harness, dose, make_task
from medcare_reachy.bridge.tests.test_live import HELP, inline, no_wait, said, until

RECORDED = {"event_id": "ev1", "status": "taken"}
OPENING = "今天感觉怎么样？"   # FakeApp.conversation_start's speech_text: today's chat opened


@pytest.fixture(autouse=True)
def quick_polls(monkeypatch):
    """The session looks for a reason to stop a live chat every LIVE_POLL_SECONDS of real time: keep tests quick."""
    monkeypatch.setattr(session_module, "LIVE_POLL_SECONDS", 0.01)


MODELS = Path("stt")                                     # the listener's models folder (speech_models_dir)


@pytest.fixture(autouse=True)
def voice_detector(monkeypatch):
    """The listener's voice detector is on this robot: a live companion chat needs it (speech_models_dir)."""
    monkeypatch.setattr(session_module, "speech_models_dir", lambda voice: MODELS)


class FakeLiveChat:
    """LiveChat stand-in. run() calls `during()` (what happens while the chat runs), then returns `outcome`, or raises
    it when it is an exception; with `block` it first waits to be cancelled or stopped (`block` seconds of real time,
    5 for True: a stop that never cancels the chat fails the test, never hangs it). stop(reason, say) records the
    companion's ending and ends the chat with it."""

    def __init__(self, outcome=None, during=None, block=False):
        self.outcome = outcome if outcome is not None else LiveOutcome("ended", "goodbye", "c9")
        self.during, self.block = during, block
        self.made, self.models_dirs, self.runs, self.stops, self.cancelled = [], [], [], [], False
        self.last_speech_at = None        # LiveChat's: when the robot last heard the patient begin to speak
        self.last_voice_at = None         # LiveChat's: when it last heard the patient speaking
        self.closing = False              # LiveChat's: the ending is decided and what is queued plays out
        self.speech_start = "detector"    # LiveChat's voice detector; None once it failed
        self.stopped = None               # set by stop() while run() waits

    def factory(self, app, robot, speaker, models_dir):
        self.made.append((app, robot, speaker))
        self.models_dirs.append(models_dir)
        return self

    def stop(self, reason, say=None):
        self.stops.append((reason, say))
        self.outcome = LiveOutcome("ended", reason, "c9")
        if self.stopped is not None:
            self.stopped.set()

    async def run(self, task_id, language):
        self.runs.append((task_id, language))
        self.stopped = asyncio.Event()
        if self.during is not None:
            self.during()
        if self.block:
            try:
                await asyncio.wait_for(self.stopped.wait(), 5 if self.block is True else self.block)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            except asyncio.TimeoutError:
                pass
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def live_task(doses=(), live=True, microphone=True, **fields):
    task = make_task(list(doses), checkin=True, microphone=microphone, **fields)
    task["live"] = live
    return task


def checkin_live(chat):
    """A conversation-only task with live on, the patient found: the next tick starts the check-in."""
    h = Harness(live_task(reason="checkin"), voice=True, speaker=True, live_chat=chat.factory)
    h.find_patient()
    assert h.slot.state == "CHECKIN"
    return h


def test_a_live_check_in_runs_the_live_chat_and_ends_with_it():
    chat = FakeLiveChat()
    h = checkin_live(chat)
    seen = []
    chat.during = lambda: seen.append((h.voice.suspended, h.voice.active, h.stream.attached["session_id"],
                                       h.stream.max_fps, h.robot.calls[-1]))
    h.tick()
    assert chat.made == [(h.app, h.robot, h.speaker)] and chat.runs == [("task-1", "zh-TW")]
    assert chat.models_dirs == [MODELS]                 # the chat hears the patient begin to speak with it
    # The listener gave the microphone up, the search's camera session stayed on at the companion's rate (0.9.0),
    # and Reachy held its head still.
    assert seen == [(True, False, "s1", COMPANION_CAPTURE_FPS, "hold_head")]
    assert not h.app.named("monitor_end")
    assert h.voice.handovers == ["suspend", "resume"]
    # The server stored and ended the conversation itself; Reachy's own voice said only the greeting.
    assert not h.app.named("conversation_start") and not h.app.named("conversation_end")
    assert h.speaker.said == [CHECKIN_GREETING["zh-TW"]]
    assert h.slot.state == "WIND_DOWN"
    h.run_until(lambda: h.slot.done)
    assert h.slot.result == "completed" and ("monitor_end", "s1") in h.app.calls and h.stream.max_fps is None
    assert h.app.named("task_status")[-1][2] == {"outcomes": {"live": "ended", "checkin": "goodbye"}}


def test_a_dose_tasks_check_in_follows_the_same_rule():
    chat = FakeLiveChat(LiveOutcome("ended", "silence", "c9"))
    h = Harness(live_task([dose(1)]), voice=True, speaker=True, live_chat=chat.factory)
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="verified", recorded=RECORDED)
    h.tick()
    assert h.slot.state == "CHECKIN"
    h.tick()
    assert chat.runs == [("task-1", "zh-TW")] and not h.app.named("conversation_start")
    assert ("monitor_end", "s2") in h.app.calls and h.voice.handovers == ["suspend", "resume"]
    assert h.slot.state == "POST_SLOT_OBSERVE" and h.voice.active is False
    assert h.slot.outcomes == {1: "recorded", "live": "ended", "checkin": "silence"}


@pytest.mark.parametrize("live", [False, "missing"])
def test_without_live_in_the_task_nothing_changes(live):
    task = live_task(reason="checkin", live=live)
    if live == "missing":
        del task["live"]                                # a server from before live voice
    chat = FakeLiveChat()
    h = Harness(task, voice=True, speaker=True, live_chat=chat.factory)
    h.find_patient()
    h.tick()
    assert chat.made == [] and h.voice.handovers == []
    assert ("conversation_start", "task-1", "zh-TW") in h.app.calls and h.speaker.said[-1] == OPENING
    assert "live" not in h.slot.outcomes


def test_a_live_chat_blocked_for_a_risk_ends_with_the_risk_reason():
    h = checkin_live(FakeLiveChat(LiveOutcome("ended", "risk", "c9", risk=True)))
    h.tick()
    # The server had Reachy say the help line (`say`, through LiveChat), alerted family and ended the conversation.
    assert h.slot.outcomes == {"live": "ended", "checkin": "risk"} and h.slot.state == "WIND_DOWN"
    assert not h.app.named("conversation_end") and h.speaker.said == [CHECKIN_GREETING["zh-TW"]]


def test_a_fallback_before_the_conversation_existed_opens_todays_chat():
    h = checkin_live(FakeLiveChat(LiveOutcome("fallback", "live_unavailable", None)))
    h.tick()
    assert h.voice.handovers == ["suspend", "resume"]
    assert ("conversation_start", "task-1", "zh-TW") in h.app.calls and h.speaker.said[-1] == OPENING
    assert h.slot.chat_id == "c1" and h.voice.active and h.voice.mode == "chat"
    h.voice.speak("我早上去散步了")
    h.tick(1)
    assert h.app.named("conversation_turn")[-1][1:3] == ("c1", "我早上去散步了") and h.speaker.said[-1] == "真好"
    assert h.slot.outcomes["live"] == "live_unavailable"


def test_a_fallback_after_the_chat_started_goes_on_in_the_same_conversation():
    h = checkin_live(FakeLiveChat(LiveOutcome("fallback", "live_ended", "c7")))
    h.app.turn_ids = 4                                  # the turns the live chat stored
    h.tick()
    assert not h.app.named("conversation_start")
    assert h.speaker.said == [CHECKIN_GREETING["zh-TW"]]   # no second opening line: Reachy simply listens
    assert h.slot.chat_id == "c7" and h.voice.active and h.voice.mode == "chat"
    h.voice.speak("我早上去散步了")
    h.tick(1)
    assert h.app.named("conversation_turn")[-1][1:3] == ("c7", "我早上去散步了") and h.speaker.said[-1] == "真好"
    h.app.replies = [{"reply": "再見", "speech_text": "再见", "end": True}]
    h.voice.speak("好，拜拜")
    h.tick(1)
    assert ("conversation_end", "c7", "goodbye") in h.app.calls
    assert h.slot.outcomes == {"live": "live_ended", "checkin": "goodbye"}


def test_todays_chat_after_a_long_live_chat_counts_from_the_fallback(monkeypatch):
    """There is no time limit any more (0.9.0); a silence limit, when one is set, counts from the fallback."""
    monkeypatch.setattr(session_module, "CHECKIN_SILENCE", 25.0)
    chat = FakeLiveChat(LiveOutcome("fallback", "live_closed", "c7"))
    h = checkin_live(chat)
    h.app.turn_ids = 10

    def nine_minutes():
        h.clock.t += 540.0                              # a long live chat, then GPT-Live closed

    chat.during = nine_minutes
    h.tick()
    assert h.slot.state == "CHECKIN" and h.slot.chat_id == "c7"
    h.tick(24)
    assert h.slot.state == "CHECKIN" and not h.app.named("conversation_end")
    h.tick(2)
    assert ("conversation_end", "c7", "silence") in h.app.calls


def test_a_failing_live_chat_falls_back_to_todays_chat(caplog):
    h = checkin_live(FakeLiveChat(RuntimeError("socket bug")))
    with caplog.at_level(logging.ERROR, logger="medcare_reachy.bridge.session"):
        h.tick()
    assert "the live chat failed" in caplog.text
    assert h.voice.handovers == ["suspend", "resume"]
    assert ("conversation_start", "task-1", "zh-TW") in h.app.calls and h.slot.chat_id == "c1"
    assert h.slot.outcomes["live"] == "error" and not h.slot.done


def test_stopping_during_the_live_chat_ends_it_at_once():
    chat = FakeLiveChat(block=True)
    h = checkin_live(chat)
    chat.during = lambda: h.slot.stop("stop_all")       # a heartbeat says stop_all while the chat runs
    h.tick()
    assert chat.cancelled and h.voice.handovers == ["suspend", "resume"]
    h.tick()
    assert h.slot.done and h.slot.result == "stop_all" and h.robot.calls[-1] == "sleep"
    assert h.app.named("task_status")[-1][1:] == (
        "aborted", {"reason": "stop_all", "outcomes": {"live": "stopped", "checkin": "stopped"}})
    assert not h.app.named("conversation_end")          # the server ends a live chat whose robot left


def test_a_stop_while_the_microphone_is_handed_over_never_starts_the_live_chat():
    chat = FakeLiveChat()
    h = checkin_live(chat)
    suspend = h.voice.suspend

    def stop_meanwhile():
        suspend()
        h.slot.stop("stop_all")                         # the heartbeat's stop_all lands right then

    h.voice.suspend = stop_meanwhile
    h.tick()
    assert chat.runs == [] and h.voice.handovers == ["suspend", "resume"]
    h.tick()
    assert h.slot.done and h.slot.result == "stop_all"


def test_withdrawn_microphone_consent_stops_the_live_chat():
    chat = FakeLiveChat(block=True)
    h = checkin_live(chat)
    chat.during = lambda: h.slot.task.update(microphone=False)   # the heartbeat's "microphone": false
    h.tick()
    assert chat.cancelled and h.voice.handovers == ["suspend", "resume"]
    assert h.slot.outcomes == {"live": "microphone_withdrawn", "checkin": "microphone_withdrawn"}
    assert h.slot.state == "WIND_DOWN" and not h.app.named("conversation_start")


def test_a_live_companion_chat_has_no_time_limit():
    """0.9.0: the robot's 660 s backstop (LIVE_CHECKIN_MAX) is gone; a companion stays while the patient is there."""
    chat = FakeLiveChat(block=0.2)
    h = checkin_live(chat)

    def an_hour():
        h.clock.t += 3600.0                             # an hour of chatting, the patient in view:
        h.see(identity_status="verified")               # the camera's answers keep coming

    chat.during = an_hour
    h.tick()
    assert not chat.cancelled and chat.stops == []
    assert h.slot.outcomes == {"live": "ended", "checkin": "goodbye"}


def test_a_live_companion_rests_when_nobody_is_heard_or_seen():
    chat = FakeLiveChat(block=True)
    h = checkin_live(chat)

    def patient_leaves():
        h.see(identity_status="searching")              # nobody in view ...
        h.clock.t += REST_AFTER_SECONDS + 1             # ... and nobody heard, for ten minutes

    chat.during = patient_leaves
    h.tick()
    assert chat.stops == [("patient_left", REST_LINE["zh-TW"])] and not chat.cancelled
    assert h.slot.outcomes == {"live": "ended", "checkin": "patient_left"} and h.slot.state == "WIND_DOWN"
    h.run_until(lambda: h.slot.done)
    assert "wind_down" not in h.clips.played and h.speaker.said == [CHECKIN_GREETING["zh-TW"]]
    assert h.stream.max_fps is None and ("monitor_end", "s1") in h.app.calls


def test_a_due_medicine_ends_the_live_chat_with_reachys_line():
    chat = FakeLiveChat(block=True)
    h = checkin_live(chat)
    chat.during = h.slot.medicine_due                   # a heartbeat's task_due while the chat runs
    h.tick()
    assert chat.stops == [("medicine", MEDICINE_LINE["zh-TW"])]
    assert h.slot.outcomes == {"live": "ended", "checkin": "medicine"} and h.slot.state == "WIND_DOWN"
    h.run_until(lambda: h.slot.done)
    assert h.slot.result == "completed" and "wind_down" not in h.clips.played


def test_a_hand_to_mouth_moment_during_reachys_closing_line_is_no_news_for_family():
    chat = FakeLiveChat(block=True)
    h = checkin_live(chat)
    stop = chat.stop
    sip = {"event_id": "e1", "decision": "uncertain", "confidence": 0.3, "frame_seq": 7}

    def closing_line(reason, say=None):
        h.stream.frame_seq = 8
        h.see(identity_status="verified", extra_events=[sip])   # a sip while Reachy says its line
        asyncio.get_running_loop().call_later(0.05, stop, reason, say)   # the line takes a moment

    chat.stop = closing_line
    chat.during = h.slot.medicine_due
    h.tick()
    assert chat.stops == [("medicine", MEDICINE_LINE["zh-TW"])] and h.slot.state == "WIND_DOWN"
    in_flight = {"event_id": "e2", "decision": "uncertain", "confidence": 0.4, "frame_seq": 8}   # answered late
    h.see(identity_status="verified", extra_events=[sip, in_flight])
    h.run_until(lambda: h.slot.done)
    assert not h.app.named("extra_event") and h.slot.result == "completed"


def test_a_live_chat_whose_voice_detector_stops_goes_on_as_todays_chat(caplog):
    """Without speech signals GPT-Live could never start again (LiveChat._listen drops a failed voice detector)."""
    chat = FakeLiveChat(block=True)
    h = checkin_live(chat)
    chat.during = lambda: setattr(chat, "speech_start", None)
    with caplog.at_level(logging.WARNING, logger="medcare_reachy.bridge.session"):
        h.tick()
    assert chat.stops == [("detector_failed", None)] and not chat.cancelled   # stopped without a line
    assert [record.levelname for record in caplog.records if record.name == "medcare_reachy.bridge.session"] == [
        "WARNING"]
    assert h.voice.handovers == ["suspend", "resume"]
    # Today's chat in a new conversation, as after a fallback the robot decides itself.
    assert ("conversation_start", "task-1", "zh-TW") in h.app.calls and h.slot.chat_id == "c1"
    assert h.speaker.said[-1] == OPENING and h.slot.state == "CHECKIN"
    assert h.slot.outcomes == {"live": "detector_failed"}


def test_a_live_companion_chat_without_a_camera_goes_on_as_todays_chat(caplog):
    """Camera first: without a working camera the server could never start GPT-Live again."""
    chat = FakeLiveChat(block=True)
    h = checkin_live(chat)

    def camera_dies():
        h.stream.camera.state = "unavailable"           # the watchdog gave up on the camera (the patient may be there)
        asyncio.get_running_loop().call_later(0.05, setattr, h.clock, "t", h.clock.t + CAMERA_LOST_SECONDS)

    chat.during = camera_dies
    with caplog.at_level(logging.WARNING, logger="medcare_reachy.bridge.session"):
        h.tick()
    assert chat.stops == [("camera_failed", None)] and not chat.cancelled   # stopped without a line
    assert "the companion's camera has not worked for 30 s" in caplog.text
    # Today's chat in a new conversation, as after a fallback the robot decides itself.
    assert ("conversation_start", "task-1", "zh-TW") in h.app.calls and h.slot.chat_id == "c1"
    assert h.slot.outcomes == {"live": "camera_failed"} and h.slot.state == "CHECKIN"


def test_a_companion_ending_never_cuts_the_servers_help_line():
    """The server ended the chat for a risk and its help line plays (LiveChat's final drain) when a heartbeat says a
    medicine is due: the whole help line is heard, and no medicine ending is asked for."""
    socket, made = FakeSocket(), []

    async def connect(url, headers):
        return socket

    def real_chat(app, robot, speaker, models_dir):
        made.append(LiveChat(LIVE_APP, robot, speaker, connect=connect, clock=h.clock.monotonic, sleep=no_wait,
                             run_blocking=inline, speech_start=object()))
        return made[-1]

    h = Harness(live_task(reason="checkin"), voice=True, speaker=True, live_chat=real_chat)
    h.find_patient()
    looks = []                                          # each time the session looks at the running chat
    take_error = h.stream.take_error
    h.stream.take_error = lambda: looks.append(1) or take_error()
    opening = "王奶奶，今天感觉怎么样？"

    async def scenario():
        tick = asyncio.ensure_future(h.slot.tick())
        await until(lambda: made and socket.json_sent())   # the chat said start
        chat = made[0]
        drain = chat._drain

        async def medicine_during_the_help_line():
            async with chat._speaker_lock:              # the help line has not played yet ...
                h.slot.medicine_due()                   # ... when the heartbeat's task_due lands
                seen = len(looks)
                await until(lambda: len(looks) > seen)  # the session looks at the chat meanwhile
            await drain()

        chat._drain = medicine_during_the_help_line
        socket.serve({"type": "ready", "conversation_id": "c9", "speech_text": opening})
        socket.serve({"type": "say", "speech_text": HELP, "reason": "risk"})
        socket.serve({"type": "end", "reason": "risk", "risk": True})
        await tick

    asyncio.run(scenario())
    heard = h.robot.heard()
    assert list(heard[heard != 0]) == [8192] * (said(opening) + said(HELP))   # the opening, then the whole help line
    assert {"type": "end", "reason": "medicine"} not in socket.json_sent()
    assert not any(MEDICINE_LINE["zh-TW"] in text for text in h.speaker.synthesised)
    assert h.slot.outcomes == {"live": "ended", "checkin": "risk"} and h.slot.state == "WIND_DOWN"


def test_no_live_chat_without_the_listeners_voice_detector(monkeypatch):
    monkeypatch.setattr(session_module, "speech_models_dir", lambda voice: None)
    chat = FakeLiveChat()
    h = checkin_live(chat)
    h.tick()
    assert chat.made == [] and h.voice.handovers == []   # the server would never hear the patient begin to speak
    assert ("conversation_start", "task-1", "zh-TW") in h.app.calls and h.slot.chat_id == "c1"


def test_a_bridge_shutdown_during_the_live_chat_waits_for_it_to_close():
    chat = FakeLiveChat(block=True)
    h = checkin_live(chat)

    async def shutdown_mid_chat():
        tick = asyncio.ensure_future(h.slot.tick())
        while not chat.runs:
            await asyncio.sleep(0)
        tick.cancel()                                   # Runner.run cancels the task loop ...
        with pytest.raises(asyncio.CancelledError):
            await tick
        h.slot.stop("bridge_shutdown", abort=False)     # ... and Runner.run_slot ends the slot
        await h.slot.tick()

    asyncio.run(shutdown_mid_chat())
    assert chat.cancelled and h.voice.handovers == ["suspend", "resume"]
    assert h.slot.done and h.slot.result == "bridge_shutdown"
    assert h.app.named("task_status")[-1][1] == "in_progress"   # the lease is kept for recovery


def test_a_robot_app_that_cant_hold_a_live_chat_uses_todays_chat():
    h = Harness(live_task(reason="checkin"), voice=True, speaker=True,
                live_chat=lambda app, robot, speaker, models_dir: None)
    h.find_patient()
    h.tick()
    assert h.voice.handovers == [] and ("conversation_start", "task-1", "zh-TW") in h.app.calls
    assert h.slot.outcomes["live"] == "unsupported"


def test_no_live_chat_without_reachys_own_voice_for_the_help_line():
    chat = FakeLiveChat()
    h = Harness(live_task(reason="checkin"), voice=True, speaker=True, live_chat=chat.factory)
    h.speaker.available = False                         # no Matcha voice on this robot
    h.find_patient()
    h.tick()
    assert chat.made == [] and h.voice.handovers == []
    assert ("conversation_start", "task-1", "zh-TW") in h.app.calls


def test_no_live_chat_without_microphone_consent():
    chat = FakeLiveChat()
    h = Harness(live_task(reason="checkin", microphone=False), voice=True, speaker=True, live_chat=chat.factory)
    h.find_patient()
    h.tick()
    assert chat.made == [] and ("conversation_start", "task-1", "zh-TW") in h.app.calls


@pytest.mark.parametrize("before,now", [(False, True), (True, False)])
def test_the_task_read_again_before_the_doses_decides(before, now):
    chat = FakeLiveChat()
    h = Harness(live_task([dose(1)], live=before), voice=True, speaker=True, live_chat=chat.factory)
    h.app.current_task["live"] = now                    # the card's switch changed after the lease
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="verified", recorded=RECORDED)
    h.tick()
    h.tick()
    assert bool(chat.runs) is now and bool(h.app.named("conversation_start")) is not now


class PlainRobot:
    """A robot wrapper without live audio, like the video-file test robot."""

    def get_audio(self):
        return None


class LiveAudioRobot(PlainRobot):
    """A robot wrapper with the live audio calls (bridge/media.py)."""

    def start_live_audio(self):
        return True

    def push_audio(self, samples):
        return True

    def clear_live_audio(self):
        return True

    def stop_live_audio(self):
        pass


LIVE_APP = types.SimpleNamespace(live_url=lambda: "ws://app:8001/api/device/conversations/live",
                                 auth_header=lambda: {"Authorization": "Bearer rdv1.token"})


class FakeSpeechStart:
    """live.SpeechStart stand-in: the real one loads the silero model with sherpa_onnx (only on the robot)."""

    def __init__(self, models_dir):
        self.models_dir = models_dir


def test_the_robot_app_holds_a_live_chat_only_with_live_audio_and_a_voice_detector(monkeypatch, tmp_path):
    monkeypatch.setattr(live_module, "SpeechStart", FakeSpeechStart)
    made = []
    real = live_module.LiveChat

    def spy(*args, **kwargs):
        made.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(live_module, "LiveChat", spy)
    assert isinstance(open_live_chat(LIVE_APP, LiveAudioRobot(), FakeSpeaker(), tmp_path), LiveChat)
    assert [type(kwargs["speech_start"]) for kwargs in made] == [FakeSpeechStart]
    assert made[0]["speech_start"].models_dir == tmp_path
    assert open_live_chat(LIVE_APP, PlainRobot(), FakeSpeaker(), tmp_path) is None
    assert open_live_chat(LIVE_APP, LiveAudioRobot(), FakeSpeaker(), None) is None   # no voice detector
    assert len(made) == 1


def test_without_the_live_module_the_robot_app_says_so(monkeypatch, caplog, tmp_path):
    monkeypatch.setitem(sys.modules, "medcare_reachy.bridge.live", None)   # as if `websockets` were missing
    with caplog.at_level(logging.WARNING, logger="medcare_reachy.bridge.session"):
        assert open_live_chat(LIVE_APP, LiveAudioRobot(), FakeSpeaker(), tmp_path) is None
    assert "live voice is off in this robot app" in caplog.text
