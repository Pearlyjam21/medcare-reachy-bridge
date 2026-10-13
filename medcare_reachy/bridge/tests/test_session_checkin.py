"""Check-in conversations in the slot state machine: after the last dose, or as a conversation-only task."""

import asyncio

import httpx
import pytest

from medcare_reachy.bridge.app_client import (
    AppClient, AppUnreachable, NotAuthorised, RequestRejected, SessionLost)
from medcare_reachy.bridge import session as session_module
from medcare_reachy.bridge.session import (
    CHECKIN_GREETING, COMPANION_CAPTURE_FPS, MEDICINE_LINE, REST_AFTER_SECONDS, REST_LINE, UNREACHABLE_LIMIT,
    SlotSession)
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
    assert ("conversation_start", "task-1", "zh-TW") in h.app.calls and h.app.companion_starts == [True]
    assert h.speaker.said == ["今天感觉怎么样？"]
    assert h.voice.active and h.voice.mode == "chat"
    # The dose session ends; the camera stays on in an observe session, slowly: it only tells who is there (0.9.0).
    assert ("monitor_end", "s2") in h.app.calls and ("monitor_start", "observe", None) in h.app.calls
    assert h.stream.attached["session_id"] == "s3" and h.stream.max_fps == COMPANION_CAPTURE_FPS

    h.voice.speak("我早上去散步了")
    h.tick(1)
    assert h.app.named("conversation_turn")[-1][1:3] == ("c1", "我早上去散步了") and h.speaker.said[-1] == "真好"

    h.app.replies = [{"reply": "再見", "speech_text": "再见", "end": True}]
    h.voice.speak("好，拜拜")
    h.tick(1)
    assert ("conversation_end", "c1", "goodbye") in h.app.calls
    assert h.slot.state == "POST_SLOT_OBSERVE" and h.slot.outcomes["checkin"] == "goodbye"
    assert h.voice.active is False
    h.tick()   # POST_SLOT_OBSERVE opens its own camera session (the companion's was flagged)
    assert h.stream.attached["session_id"] == "s4" and h.stream.max_fps is None   # observing on, at the full rate


def test_the_observe_session_of_a_companion_session_says_companion_and_one_outside_it_does_not():
    h = Harness(make_task([dose(1)], microphone=True, checkin=True), voice=True, speaker=True)
    h.find_patient()   # the search before the dose
    assert h.app.named("monitor_start") == [("monitor_start", "observe", None)] and h.app.monitor_companion == [False]
    h.verify_and_prompt()
    h.see(identity_status="verified", recorded=RECORDED)
    h.tick()
    h.tick()   # the companion session begins; its camera session is opened for it
    assert h.app.named("monitor_start")[-1] == ("monitor_start", "observe", None) and h.app.monitor_companion[-1] is True
    h.app.replies = [{"reply": "再見", "speech_text": "再见", "end": True}]
    h.voice.speak("好，拜拜")
    h.tick(1)
    assert h.slot.state == "POST_SLOT_OBSERVE" and h.slot.companion_since is None
    h.tick()
    assert h.app.monitor_companion[-1] is False   # the watch after the dose runs on a new session, without the flag
    assert h.stream.attached["session_id"] == "s4"


def test_without_checkin_consent_there_is_no_conversation():
    h = after_one_dose(checkin=False)
    assert h.slot.state == "POST_SLOT_OBSERVE"
    assert not h.app.named("conversation_start")


def test_reachy_is_muted_while_it_speaks():
    h = after_one_dose()
    h.tick()
    assert h.voice.max_holds >= 1 and h.voice.holds == 0


def test_silence_alone_never_ends_the_conversation():
    """A companion stays while the patient is quiet: no silence limit by default (0.7.1), and no time limit while the
    patient is in view (0.9.0: the 5-minute CHECKIN_MAX is gone)."""
    h = after_one_dose()
    h.tick()
    for _ in range(6):                                    # an hour, in view: the camera's answers keep coming
        h.see(identity_status="verified")
        h.tick(REST_AFTER_SECONDS - 1)
    assert h.slot.state == "CHECKIN" and not h.app.named("conversation_end")


def test_reachy_rests_after_ten_minutes_with_no_voice_and_nobody_seen():
    h = after_one_dose()
    h.tick()
    h.see(identity_status="searching")                    # the patient left the camera's view
    h.tick(REST_AFTER_SECONDS - 1)
    assert h.slot.state == "CHECKIN" and not h.app.named("conversation_end")
    h.tick(2)
    assert h.speaker.said[-1] == REST_LINE["zh-TW"] and h.voice.spoken[-1] == REST_LINE["zh-TW"]
    assert ("conversation_end", "c1", "patient_left") in h.app.calls
    assert h.slot.state == "WIND_DOWN"                    # Reachy sleeps now: no two minutes of POST_SLOT_OBSERVE
    h.run_until(lambda: h.slot.done)
    assert "wind_down" not in h.clips.played             # it has just said why it rests
    assert h.app.named("task_status")[-1][1:] == (
        "completed", {"outcomes": {"1": "recorded", "checkin": "patient_left"}})
    assert h.stream.max_fps is None


def test_the_patient_in_view_keeps_reachy_awake():
    h = after_one_dose()
    h.tick()
    for _ in range(3):                                    # in view for 15 minutes: the camera's answers keep coming
        h.see(identity_status="verified")
        h.tick(REST_AFTER_SECONDS / 2)
    assert h.slot.state == "CHECKIN"
    h.see(identity_status="searching")                    # then they step away
    h.tick(REST_AFTER_SECONDS / 2 - 1)
    assert h.slot.state == "CHECKIN" and not h.app.named("conversation_end")   # counted from the last sighting
    h.tick(2)
    assert ("conversation_end", "c1", "patient_left") in h.app.calls


def test_a_stale_sighting_never_keeps_reachy_awake():
    """The camera stops after one verified answer (MonitorStream keeps that answer): it counts as a sighting when it
    came, not on every tick after."""
    h = after_one_dose()
    h.tick()
    h.see(identity_status="verified")                     # the last answer before the camera stopped
    h.tick(REST_AFTER_SECONDS - 1)
    assert h.slot.state == "CHECKIN"
    h.tick(2)
    assert h.speaker.said[-1] == REST_LINE["zh-TW"] and ("conversation_end", "c1", "patient_left") in h.app.calls


def test_the_patients_voice_keeps_reachy_awake():
    h = after_one_dose()
    h.tick()
    h.see(identity_status="searching")
    h.tick(REST_AFTER_SECONDS - 10)
    h.voice.speaking = True                               # the listener hears the patient (no turn yet)
    h.tick(1)
    h.voice.speaking = False
    h.tick(REST_AFTER_SECONDS - 1)
    assert h.slot.state == "CHECKIN" and not h.app.named("conversation_end")
    h.voice.speak("我還在這裡")                             # a turn counts too
    h.tick(1)
    h.tick(REST_AFTER_SECONDS - 1)
    assert h.slot.state == "CHECKIN" and not h.app.named("conversation_end")
    h.tick(2)
    assert ("conversation_end", "c1", "patient_left") in h.app.calls


def test_a_due_medicine_ends_the_chat_with_reachys_line():
    h = after_one_dose()
    h.tick()
    h.slot.medicine_due()                                 # a heartbeat's task_due
    h.tick(0.2)
    assert h.speaker.said[-1] == MEDICINE_LINE["zh-TW"]
    assert ("conversation_end", "c1", "medicine") in h.app.calls and h.slot.state == "WIND_DOWN"
    h.run_until(lambda: h.slot.done)
    assert "wind_down" not in h.clips.played and h.slot.result == "completed"
    assert h.app.named("task_status")[-1][2] == {"outcomes": {"1": "recorded", "checkin": "medicine"}}


def test_words_the_patient_just_said_go_to_the_server_before_the_medicine_line():
    h = after_one_dose()
    h.tick()
    h.voice.pauses()                                      # the patient said something; Whisper is decoding it
    h.slot.medicine_due()                                 # a heartbeat's task_due lands meanwhile
    h.tick(0.2)
    assert not h.app.named("conversation_end") and MEDICINE_LINE["zh-TW"] not in h.speaker.said   # it waits
    h.voice.speak("我剛剛頭有點暈")                         # the words are decoded
    h.tick(1)
    assert h.app.named("conversation_turn")[-1][1:3] == ("c1", "我剛剛頭有點暈") and h.speaker.said[-1] == "真好"
    assert not h.app.named("conversation_end")
    h.tick(0.2)                                           # nothing more held: now the medicine
    assert h.speaker.said[-1] == MEDICINE_LINE["zh-TW"] and ("conversation_end", "c1", "medicine") in h.app.calls


def test_a_medicine_due_while_reachy_answers_waits_for_the_answer():
    h = after_one_dose()
    h.tick()
    say = h.speaker.say

    def heartbeat_meanwhile(text, stats=None, on_audio=None):
        h.slot.medicine_due()                             # the heartbeat lands while Reachy speaks
        return say(text, stats, on_audio)

    h.speaker.say = heartbeat_meanwhile
    h.voice.speak("我早上去散步了")
    h.tick(1)
    assert h.speaker.said[-1] == "真好" and not h.app.named("conversation_end")
    h.tick(0.2)
    assert h.speaker.said[-2:] == ["真好", MEDICINE_LINE["zh-TW"]]
    assert ("conversation_end", "c1", "medicine") in h.app.calls


def test_a_due_medicine_never_cuts_a_dose_short():
    h = Harness(make_task([dose(1)], microphone=True, checkin=True), voice=True, speaker=True)
    h.find_patient()
    h.verify_and_prompt()
    h.slot.medicine_due()                                 # WATCHING: the dose task finishes first
    h.see(identity_status="verified", recorded=RECORDED)
    h.tick()
    h.tick()
    h.tick(1)
    assert h.slot.state == "CHECKIN" and h.slot.chat_id == "c1" and not h.app.named("conversation_end")


def test_hand_to_mouth_moments_while_chatting_are_no_news_for_family():
    h = after_one_dose()
    h.tick()
    sip = {"event_id": "e1", "decision": "uncertain", "confidence": 0.3}
    h.see(identity_status="verified", extra_events=[sip])     # a sip of tea while chatting
    h.tick(1)
    assert not h.app.named("extra_event")
    h.app.replies = [{"reply": "再見", "speech_text": "再见", "end": True}]
    h.voice.speak("好，拜拜")
    h.tick(1)
    assert h.slot.state == "POST_SLOT_OBSERVE"
    later = {"event_id": "e2", "decision": "uncertain", "confidence": 0.4}
    h.see(identity_status="verified", extra_events=[sip, later])   # after the chat, at the full rate: as before
    h.tick(1)
    assert h.app.named("extra_event") == [("extra_event", "e2", "uncertain", 0.4)]


def test_a_hand_to_mouth_moment_during_reachys_last_line_is_no_news_either():
    h = after_one_dose()
    h.tick()
    say = h.speaker.say
    sip = {"event_id": "e1", "decision": "uncertain", "confidence": 0.3, "frame_seq": 40}

    def sip_during_goodbye(text, stats=None, on_audio=None):
        h.stream.frame_seq = 41                                   # frames streamed while Reachy says goodbye
        h.see(identity_status="verified", extra_events=[sip])
        return say(text, stats, on_audio)

    h.speaker.say = sip_during_goodbye
    h.app.replies = [{"reply": "再見", "speech_text": "再见", "end": True}]
    h.voice.speak("好，拜拜")
    h.tick(1)
    assert h.slot.state == "POST_SLOT_OBSERVE" and not h.app.named("extra_event")
    h.speaker.say = say
    in_flight = {"event_id": "e2", "decision": "uncertain", "confidence": 0.3, "frame_seq": 41}   # answered late
    later = {"event_id": "e3", "decision": "uncertain", "confidence": 0.4, "frame_seq": 42}
    h.see(identity_status="verified", extra_events=[sip, in_flight, later])
    h.tick(1)
    assert h.app.named("extra_event") == [("extra_event", "e3", "uncertain", 0.4)]


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
    # The search's camera session goes on through the chat, slowly (0.9.0).
    assert h.stream.attached["session_id"] == "s1" and h.stream.max_fps == COMPANION_CAPTURE_FPS
    assert not h.app.named("monitor_end")
    h.see(identity_status="searching")
    h.tick(REST_AFTER_SECONDS + 1)
    assert h.slot.state == "WIND_DOWN" and h.speaker.said[-1] == REST_LINE["zh-TW"]
    h.run_until(lambda: h.slot.done)
    assert h.slot.result == "completed" and ("monitor_end", "s1") in h.app.calls
    assert h.app.named("task_status")[-1][2] == {"outcomes": {"checkin": "patient_left"}}


def test_a_withdrawn_microphone_ends_todays_companion_chat():
    h = after_one_dose()
    h.tick()
    assert h.voice.active and h.voice.mode == "chat"
    h.slot.task["microphone"] = False                     # the heartbeat's "microphone": false
    h.tick(0.2)
    assert ("conversation_end", "c1", "stopped") in h.app.calls and h.voice.active is False
    assert h.slot.outcomes["checkin"] == "microphone_withdrawn"


def test_stopping_mid_conversation_closes_it():
    h = after_one_dose()
    h.tick()
    h.slot.stop("stop_all")
    h.tick()
    assert ("conversation_end", "c1", "stopped") in h.app.calls and h.slot.done
    assert h.stream.attached is None and h.stream.max_fps is None   # no slow camera left for the next task


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
