"""Check-in conversations in the slot state machine: after the last dose, or as a conversation-only task."""

import asyncio

import httpx
import pytest

from medcare_reachy.bridge.app_client import (
    AppClient, AppUnreachable, NotAuthorised, RequestRejected, SessionLost)
from medcare_reachy.bridge import session as session_module
from medcare_reachy.bridge.session import CHECKIN_GREETING, CHECKIN_MAX, UNREACHABLE_LIMIT, SlotSession
from medcare_reachy.bridge.tests.fakes import (
    FakeClips, FakeClock, FakeRobot, FakeSpeaker, FakeStream, FakeVoice, Harness, dose, make_task)

RECORDED = {"event_id": "ev1", "status": "taken"}


def after_one_dose(checkin=True, **session):
    h = Harness(make_task([dose(1)], microphone=True, checkin=checkin), voice=True, speaker=True, **session)
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="verified", recorded=RECORDED)
    h.tick()
    return h


def test_after_the_last_dose_reachy_chats_then_observes():
    h = after_one_dose()
    assert h.slot.state == "CHECKIN"
    h.tick()   # opens the conversation and speaks the opening line
    assert ("conversation_start", "task-1", "zh-TW") in h.app.calls
    assert h.speaker.said == ["今天感觉怎么样？"]
    assert h.voice.active and h.voice.mode == "chat"
    assert ("monitor_end", "s2") in h.app.calls          # the camera session ends while chatting

    h.voice.speak("我早上去散步了")
    h.tick(1)
    assert h.app.named("conversation_turn")[-1][1:3] == ("c1", "我早上去散步了") and h.speaker.said[-1] == "真好"

    h.app.replies = [{"reply": "再見", "speech_text": "再见", "end": True}]
    h.voice.speak("好，拜拜")
    h.tick(1)
    assert ("conversation_end", "c1", "goodbye") in h.app.calls
    assert h.slot.state == "POST_SLOT_OBSERVE" and h.slot.outcomes["checkin"] == "goodbye"
    assert h.voice.active is False


def test_without_checkin_consent_there_is_no_conversation():
    h = after_one_dose(checkin=False)
    assert h.slot.state == "POST_SLOT_OBSERVE"
    assert not h.app.named("conversation_start")


def test_reachy_is_muted_while_it_speaks():
    h = after_one_dose()
    h.tick()
    assert h.voice.max_holds >= 1 and h.voice.holds == 0


def test_silence_alone_never_ends_the_conversation():
    """A companion stays while the patient is quiet: no silence limit by default (0.7.1)."""
    h = after_one_dose()
    h.tick()
    h.tick(CHECKIN_MAX - 1)
    assert h.slot.state == "CHECKIN" and not h.app.named("conversation_end")


def test_the_conversation_ends_at_its_time_limit():
    h = after_one_dose()
    h.tick()
    h.tick(CHECKIN_MAX + 1)
    assert ("conversation_end", "c1", "finished") in h.app.calls and h.slot.state == "POST_SLOT_OBSERVE"


def test_a_silence_limit_can_still_be_set(monkeypatch):
    monkeypatch.setattr(session_module, "CHECKIN_SILENCE", 25.0)
    h = after_one_dose()
    h.tick()
    h.tick(24)
    assert h.slot.state == "CHECKIN"
    h.tick(2)
    assert ("conversation_end", "c1", "silence") in h.app.calls and h.slot.state == "POST_SLOT_OBSERVE"


def test_a_risk_answer_ends_with_the_risk_reason():
    h = after_one_dose()
    h.tick()
    h.app.replies = [{"reply": "請打 1925", "speech_text": "请打 1925", "end": True, "risk": True}]
    h.voice.speak("我不想活了")
    h.tick(1)
    assert ("conversation_end", "c1", "risk") in h.app.calls and h.speaker.said[-1] == "请打 1925"


def test_consent_refused_by_the_server_skips_the_conversation():
    h = after_one_dose()
    h.app.fail["conversation_start"] = [NotAuthorised(status=403, detail="checkin_consent_required")]
    h.tick()
    assert h.slot.state == "POST_SLOT_OBSERVE" and h.slot.outcomes["checkin"] == "unavailable"
    assert h.slot.done is False


def test_conversation_only_task_greets_finds_the_patient_and_chats():
    h = Harness(make_task([], reason="checkin", checkin=True), voice=True, speaker=True)
    h.find_patient()
    assert h.speaker.said[0] == CHECKIN_GREETING["zh-TW"]
    assert "wake_greeting" not in h.clips.played
    assert h.slot.state == "CHECKIN" and "in_progress" in h.statuses()
    h.tick()
    h.tick(CHECKIN_MAX + 1)
    assert h.slot.state == "WIND_DOWN"
    h.run_until(lambda: h.slot.done)
    assert h.slot.result == "completed"
    assert h.app.named("task_status")[-1][2] == {"outcomes": {"checkin": "finished"}}


def test_stopping_mid_conversation_closes_it():
    h = after_one_dose()
    h.tick()
    h.slot.stop("stop_all")
    h.tick()
    assert ("conversation_end", "c1", "stopped") in h.app.calls and h.slot.done


HEARD = {"speech_ms": 1800, "segments": 2, "vad_release_ms": 530, "stt_ms": 1100, "stt_last_ms": 600,
         "handover_ms": 1150, "echo_dropped": 0, "listen_mode": "chat"}


def test_each_turn_carries_how_long_hearing_it_took():
    h = after_one_dose()
    h.tick()
    h.voice.speak("我早上去散步了", HEARD)
    h.tick(1)
    assert h.app.named("conversation_turn") == [("conversation_turn", "c1", "我早上去散步了", HEARD)]


def test_reachys_timings_go_to_the_turn_it_spoke():
    h = after_one_dose()
    h.tick()   # the opening line has no round trip
    assert h.app.named("conversation_turn_metrics") == [("conversation_turn_metrics", "c1", 1, FakeSpeaker.STATS)]
    h.app.turn_seconds = 1.5
    h.voice.speak("我早上去散步了", HEARD)
    h.tick(1)
    assert h.app.named("conversation_turn_metrics")[-1] == (
        "conversation_turn_metrics", "c1", 3, {"round_trip_ms": 1500, **FakeSpeaker.STATS})
    h.app.replies = [{"reply": "再見", "speech_text": "再见", "end": True}]
    h.voice.speak("好，拜拜")
    h.tick(1)   # the goodbye is timed before the conversation is closed
    assert [call[0] for call in h.app.calls[-2:]] == ["conversation_turn_metrics", "conversation_end"]
    assert h.app.calls[-2][2] == 5


def test_no_timings_without_a_turn_id_to_file_them_under():
    h = after_one_dose()
    h.tick()
    h.app.replies = [{"reply_turn_id": None}]
    h.voice.speak("我早上去散步了")
    h.tick(1)
    assert len(h.app.named("conversation_turn_metrics")) == 1 and h.speaker.said[-1] == "真好"


@pytest.mark.parametrize("error", [
    AppUnreachable("down"), RequestRejected("gone", 404, "not_found"), RequestRejected("bad", 422, "invalid"),
    NotAuthorised(status=403, detail="x"), SessionLost(status=409, detail="x")])
def test_a_failing_timings_post_never_breaks_the_conversation(error):
    h = after_one_dose()
    h.app.fail["conversation_turn_metrics"] = [error, error]
    h.tick()      # the opening line's timings are refused
    h.voice.speak("我早上去散步了")
    h.tick(1)     # and so are the first reply's
    assert h.slot.state == "CHECKIN" and h.slot.chat_id == "c1" and not h.slot.done
    assert h.speaker.said[-1] == "真好"
    h.voice.speak("還不錯")
    h.tick(1)
    assert len(h.app.named("conversation_turn")) == 2 and len(h.app.named("conversation_turn_metrics")) == 3
    assert not h.app.named("conversation_end")


def test_the_listener_is_told_what_reachy_says():
    h = after_one_dose()
    h.tick()
    assert h.voice.spoken == ["今天感觉怎么样？"]
    h.voice.speak("我早上去散步了")
    h.tick(1)
    assert h.voice.spoken[-1] == "真好"


def test_timings_the_server_refuses_never_cost_the_patients_words():
    h = after_one_dose()
    h.tick()
    h.app.fail["conversation_turn"] = [RequestRejected("bad", 422, [{"loc": ["body", "metrics"]}])]
    h.voice.speak("我早上去散步了", HEARD)
    h.tick(1)
    assert [call[2:] for call in h.app.named("conversation_turn")] == [
        ("我早上去散步了", HEARD), ("我早上去散步了", None)]                # sent again without the timings
    assert h.speaker.said[-1] == "真好" and h.slot.state == "CHECKIN" and not h.slot.done


def test_words_the_server_refuses_still_end_the_check_in():
    h = after_one_dose()
    h.tick()
    refused = RequestRejected("bad", 422, "Empty turn")
    h.app.fail["conversation_turn"] = [refused, refused]
    h.voice.speak("我早上去散步了", HEARD)
    h.tick(1)
    assert len(h.app.named("conversation_turn")) == 2                    # one retry, without the timings
    assert h.slot.done and h.slot.result == "bridge_error"


def test_a_timed_out_timings_post_is_not_an_outage():
    """With the real AppClient: a slow metrics endpoint must not make the slot fail closed 10 s later."""
    clock = FakeClock()

    def handler(request):
        path = request.url.path
        if path.endswith("/metrics"):
            raise httpx.ReadTimeout("metrics endpoint too slow", request=request)
        if path == "/api/device/monitor/start":
            return httpx.Response(200, json={"session_id": "s1", "generation": "g1"})
        if path == "/api/device/conversations":
            return httpx.Response(200, json={"conversation_id": "c1", "reply": "你好", "speech_text": "你好",
                                             "end": False, "reply_turn_id": 1})
        if path.endswith("/turn"):
            return httpx.Response(200, json={"reply": "真好", "speech_text": "真好", "end": False, "risk": False,
                                             "reply_turn_id": 3, "server_ms": 900})
        return httpx.Response(200, json={"ok": True})

    async def scenario():
        async with AppClient("http://app:8001", "rdv1.token", transport=httpx.MockTransport(handler),
                             clock=clock.monotonic) as app:
            stream, voice = FakeStream(), FakeVoice(clock)
            slot = SlotSession(make_task([], reason="checkin", checkin=True), app=app, robot=FakeRobot(),
                               clips=FakeClips(), stream=stream, voice=voice, speaker=FakeSpeaker(), clock=clock)
            for _ in range(5):
                await slot.tick()
                if stream.attached:
                    stream.latest = {"session_id": "s1", "identity_status": "verified"}
            assert slot.state == "CHECKIN"
            await slot.tick()                      # the opening line; its timings time out
            voice.speak("我早上去散步了")
            clock.t += 1
            await slot.tick()                      # a reply; its timings time out too
            assert slot.chat_id == "c1" and app.unreachable_since is None
            clock.t += UNREACHABLE_LIMIT + 1       # no other request meanwhile: patient still thinking
            await slot.tick()
            return slot

    slot = asyncio.run(scenario())
    assert slot.state == "CHECKIN" and not slot.done


# ── 「嗯」 and body language ──────────────────────────────────────────────

def chatting(**session):
    """In the conversation, after Reachy's opening line."""
    h = after_one_dose(**session)
    h.tick()
    return h


def gestures(h):
    return [call for call in h.robot.calls if call.startswith("gesture:")]


def test_reachy_says_mm_and_starts_thinking_the_moment_the_patient_pauses():
    h = chatting()
    assert "ack" not in h.clips.played                        # never for its own opening line ...
    assert h.robot.calls[-3:] == ["gesture:speak", "say:今天感觉怎么样？", "gesture:None"]   # ... but lively saying it
    h.voice.pauses()                                          # the VAD released the patient's words: decoding them
    h.tick(0.2)
    assert h.clips.played.count("ack") == 1 and h.voice.acks == [(h.clock.t, FakeClips.ACK_SECONDS)]
    assert h.robot.calls[-1] == "gesture:think" and not h.app.named("conversation_turn")
    h.tick(0.2)                                               # still decoding: one 「嗯」 per pause
    h.voice.speak("我早上去散步了")
    h.tick(0.2)
    assert h.clips.played.count("ack") == 1 and h.app.named("conversation_turn")[-1][2] == "我早上去散步了"
    calls = h.robot.calls
    # Thinking until the reply's first sound, lively while speaking it, still again to listen.
    assert calls[calls.index("gesture:think"):] == ["gesture:think", "gesture:speak", "say:真好", "gesture:None"]
    h.voice.speak("還不錯")                                     # the next answer gets its own 「嗯」
    h.tick(1)
    assert h.clips.played.count("ack") == 2 and len(h.voice.acks) == 2


def test_the_mm_never_holds_up_the_patients_turn_or_mutes_the_listener():
    h = chatting()
    holds = h.voice.max_holds
    h.app.turn_seconds = 1.5
    h.voice.speak("我早上去散步了", HEARD)                       # released and decoded within one tick
    h.tick(0.2)
    assert h.clips.played[-1] == "ack" and h.app.named("conversation_turn")[-1][3] == HEARD
    assert h.app.named("conversation_turn_metrics")[-1][3]["round_trip_ms"] == 1500   # the reply's timings as before
    assert h.voice.max_holds == holds                         # hold() is only for Reachy's replies


def test_the_patients_turn_carries_the_listeners_mm_timings():
    h = chatting()
    heard = {**HEARD, "ack_ms": 180, "ack_resumed": False}
    h.voice.speak("我早上去散步了", heard)
    h.tick(1)
    assert h.app.named("conversation_turn")[-1][3] == heard and len(heard) <= 30   # the server keeps 30 keys
    assert len(h.app.named("conversation_turn_metrics")) == 2   # the 「嗯」 is no turn of Reachy's


def test_no_mm_for_words_said_while_reachy_was_answering():
    h = chatting()
    h.voice.speak("然后吃了早餐", ended=h.slot.last_spoke - 3)   # queued while Reachy was still answering
    h.tick(0.2)
    assert "ack" not in h.clips.played and "gesture:think" not in h.robot.calls
    assert h.app.named("conversation_turn")[-1][2] == "然后吃了早餐"


def test_reachy_keeps_still_while_the_patient_goes_on_after_a_pause():
    h = chatting()
    h.voice.pauses()
    h.tick(0.2)
    h.voice.speaking = True
    h.tick(0.2)
    assert h.robot.calls[-1] == "gesture:None"
    h.voice.speaking = False
    h.tick(0.2)
    assert h.robot.calls[-1] == "gesture:think" and h.clips.played.count("ack") == 1


def test_a_pause_that_was_only_a_cough_gets_one_mm_and_reachy_settles():
    h = chatting()
    h.voice.pauses()
    h.tick(0.2)
    h.voice.pause = None                                      # it decoded to nothing
    h.tick(0.2)
    assert h.robot.calls[-1] == "gesture:None" and not h.app.named("conversation_turn")
    h.voice.speak("我早上去散步了")                              # a new pause: a new 「嗯」
    h.tick(0.5)
    assert h.clips.played.count("ack") == 2


def test_no_mm_or_gestures_once_the_conversation_is_over():
    h = chatting()
    h.app.replies = [{"reply": "再見", "speech_text": "再见", "end": True}]
    h.voice.speak("好，拜拜")
    h.tick(1)
    assert h.slot.state == "POST_SLOT_OBSERVE" and h.clips.played.count("ack") == 1
    h.tick(1)
    calls = h.robot.calls
    assert gestures(h)[-1] == "gesture:None" and calls.index("hold_head", calls.index("say:再见")) > 0
    assert not any(call.startswith("gesture:") for call in calls[calls.index("say:再见") + 2:])


def test_the_help_line_after_a_risk_is_said_still():
    h = chatting()
    h.app.replies = [{"reply": "請打 1925", "speech_text": "请打 1925", "end": True, "risk": True}]
    h.voice.speak("我不想活了")
    h.tick(1)
    calls = h.robot.calls
    assert calls[calls.index("gesture:think"):calls.index("say:请打 1925") + 1] == [
        "gesture:think", "gesture:None", "say:请打 1925"]
    assert ("conversation_end", "c1", "risk") in h.app.calls


def test_the_closing_at_the_time_limit_is_not_acknowledged():
    h = chatting()
    h.tick(CHECKIN_MAX + 1)
    assert ("conversation_end", "c1", "finished") in h.app.calls
    assert "ack" not in h.clips.played and "gesture:think" not in h.robot.calls


def test_mm_and_gestures_can_each_be_switched_off():
    h = chatting(ack=False, gestures=False)
    h.voice.speak("我早上去散步了")
    h.tick(1)
    assert "ack" not in h.clips.played and h.voice.acks == [] and gestures(h) == []
    assert h.speaker.said[-1] == "真好"
    h = chatting(ack=False)
    h.voice.speak("我早上去散步了")
    h.tick(1)
    assert "ack" not in h.clips.played and "gesture:think" in h.robot.calls
    h = chatting(gestures=False)
    h.voice.speak("我早上去散步了")
    h.tick(1)
    assert h.clips.played.count("ack") == 1 and gestures(h) == []


def test_a_missing_mm_clip_or_a_failing_gesture_never_breaks_the_conversation():
    h = chatting()
    h.clips.missing.add("ack")                                # no English clips yet
    h.robot.gesture_error = ConnectionError("daemon gone")
    h.voice.speak("我早上去散步了")
    h.tick(1)
    assert h.voice.acks == [] and h.speaker.said[-1] == "真好"
    assert h.slot.state == "CHECKIN" and not h.slot.done


def test_stopping_while_reachy_thinks_puts_it_to_sleep_still():
    h = chatting()
    h.voice.pauses()
    h.tick(0.2)
    h.slot.stop("stop_all")
    h.tick()
    assert h.slot.done and h.robot.calls[-1] == "sleep"       # sleep() stops the gesture before it moves
    assert h.slot.gesture_mode is None


def test_a_failing_reply_leaves_reachy_still():
    h = chatting()

    def broken(text, stats=None, on_audio=None):
        on_audio()
        raise RuntimeError("synthesis failed")

    h.speaker.say = broken
    h.voice.speak("我早上去散步了")
    h.tick(1)
    calls = h.robot.calls
    assert calls[calls.index("gesture:think"):] == ["gesture:think", "gesture:speak", "gesture:None", "sleep"]
