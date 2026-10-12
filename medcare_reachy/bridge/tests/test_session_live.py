"""Live check-ins (spec 2026-10-12 F2): when the task says live, CHECKIN runs the live chat (bridge/live.py) with the
speech listener suspended, and today's chat takes over when the live chat falls back."""

import asyncio
import logging
import sys
import types

import pytest

from medcare_reachy.bridge import session as session_module
from medcare_reachy.bridge.live import LiveChat, LiveOutcome
from medcare_reachy.bridge.session import (
    CHECKIN_GREETING, CHECKIN_MAX, LIVE_CHECKIN_MAX, open_live_chat)
from medcare_reachy.bridge.tests.fakes import FakeSpeaker, Harness, dose, make_task

RECORDED = {"event_id": "ev1", "status": "taken"}
OPENING = "今天感觉怎么样？"   # FakeApp.conversation_start's speech_text: today's chat opened


@pytest.fixture(autouse=True)
def quick_polls(monkeypatch):
    """The session looks for a reason to stop a live chat every LIVE_POLL_SECONDS of real time: keep tests quick."""
    monkeypatch.setattr(session_module, "LIVE_POLL_SECONDS", 0.01)


class FakeLiveChat:
    """LiveChat stand-in. run() calls `during()` (what happens while the chat runs), then returns `outcome`, or raises
    it when it is an exception; with `block` it first waits to be cancelled (5 s at most: a stop that never cancels
    the chat fails the test, never hangs it)."""

    def __init__(self, outcome=None, during=None, block=False):
        self.outcome = outcome if outcome is not None else LiveOutcome("ended", "goodbye", "c9")
        self.during, self.block = during, block
        self.made, self.runs, self.cancelled = [], [], False

    def factory(self, app, robot, speaker):
        self.made.append((app, robot, speaker))
        return self

    async def run(self, task_id, language):
        self.runs.append((task_id, language))
        if self.during is not None:
            self.during()
        if self.block:
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
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
    chat.during = lambda: seen.append((h.voice.suspended, h.voice.active, h.stream.attached, h.robot.calls[-1]))
    h.tick()
    assert chat.made == [(h.app, h.robot, h.speaker)] and chat.runs == [("task-1", "zh-TW")]
    # The listener gave the microphone up, the camera session had ended, and Reachy held its head still.
    assert seen == [(True, False, None, "hold_head")]
    assert ("monitor_end", "s1") in h.app.calls
    assert h.voice.handovers == ["suspend", "resume"]
    # The server stored and ended the conversation itself; Reachy's own voice said only the greeting.
    assert not h.app.named("conversation_start") and not h.app.named("conversation_end")
    assert h.speaker.said == [CHECKIN_GREETING["zh-TW"]]
    assert h.slot.state == "WIND_DOWN"
    h.run_until(lambda: h.slot.done)
    assert h.slot.result == "completed"
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


def test_todays_chat_after_a_long_live_chat_counts_from_the_fallback():
    chat = FakeLiveChat(LiveOutcome("fallback", "live_closed", "c7"))
    h = checkin_live(chat)
    h.app.turn_ids = 10

    def nine_minutes():
        h.clock.t += 540.0                              # a long live chat, then GPT-Live closed

    chat.during = nine_minutes
    h.tick()
    assert 540.0 > CHECKIN_MAX and h.slot.state == "CHECKIN"
    h.tick(CHECKIN_MAX - 1)
    assert h.slot.state == "CHECKIN" and not h.app.named("conversation_end")
    h.tick(2)
    assert ("conversation_end", "c7", "finished") in h.app.calls


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


def test_a_live_chat_that_never_ends_is_stopped_by_the_robot():
    chat = FakeLiveChat(block=True)
    h = checkin_live(chat)

    def hang():
        h.clock.t += LIVE_CHECKIN_MAX                   # the server never ended it

    chat.during = hang
    h.tick()
    assert chat.cancelled and h.slot.outcomes == {"live": "too_long", "checkin": "too_long"}
    assert h.slot.state == "WIND_DOWN"


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
                live_chat=lambda app, robot, speaker: None)
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


def test_the_robot_app_holds_a_live_chat_only_with_live_audio():
    assert isinstance(open_live_chat(LIVE_APP, LiveAudioRobot(), FakeSpeaker()), LiveChat)
    assert open_live_chat(LIVE_APP, PlainRobot(), FakeSpeaker()) is None


def test_without_the_live_module_the_robot_app_says_so(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "medcare_reachy.bridge.live", None)   # as if `websockets` were missing
    with caplog.at_level(logging.WARNING, logger="medcare_reachy.bridge.session"):
        assert open_live_chat(LIVE_APP, LiveAudioRobot(), FakeSpeaker()) is None
    assert "live voice is off in this robot app" in caplog.text
