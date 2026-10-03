"""SlotSession when overdose protection refuses a dose (409 with a reason): Reachy says why once and moves on."""

import pytest

from medcare_reachy.bridge.session import AWAY_TIMEOUT, DONE_GRACE
from medcare_reachy.bridge.tests.fakes import TOO_SOON_SPEECH, Harness, dose, lost, make_task, refused

READY_UNCERTAIN = {"event_id": "ev1", "decision": "uncertain", "confidence": 0.31, "frame_seq": 40, "ready": True}
RECORDED = {"event_id": "ev1", "status": "taken"}
NOT_DUE = "现在还不是吃这个药的时间，晚上9点以后才可以。"
EXPIRED = "早上8点的药已经错过了，请不要补吃，等下一次就好。"


def prompts(h):
    return [clip for clip in h.clips.played if clip.startswith("med_prompt")]


def watching(doses=None, **task):
    h = Harness(make_task(doses or [dose(1), dose(2)], **task), voice=True, speaker=True)
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="verified")
    return h


@pytest.mark.parametrize("detail, speech", [
    ("dose_not_due_yet", NOT_DUE), ("dose_too_soon", TOO_SOON_SPEECH),
    ("daily_max_reached", "今天的 allegra 已经吃满 4 次了，请不要再吃。"), ("dose_expired", EXPIRED)])
def test_refused_dose_start_is_explained_instead_of_prompted(detail, speech):
    h = Harness(make_task([dose(1), dose(2)], microphone=True), voice=True, speaker=True)
    h.find_patient()
    h.app.fail["monitor_start"] = [refused(detail, intk_id=1, speech_text=speech)]
    h.tick()
    assert h.speaker.said == [speech]
    assert h.voice.spoken == [speech] and h.voice.max_holds == 1 and h.voice.holds == 0   # muted while speaking
    assert h.slot.outcomes[1] == detail and h.slot.index == 1 and h.slot.state == "MED_PROMPT"
    assert prompts(h) == [] and h.app.named("confirmation") == []
    h.verify_and_prompt()                       # the next dose goes on as usual
    assert prompts(h) == ["med_prompt:20"] and h.speaker.said == [speech]


def test_every_refused_dose_is_explained_once_and_the_slot_winds_down():
    h = Harness(make_task([dose(1), dose(2), dose(3)]), speaker=True)
    h.find_patient()
    h.app.fail["monitor_start"] = [refused("dose_not_due_yet", 1, NOT_DUE), refused("dose_too_soon", 2),
                                   refused("dose_expired", 3, EXPIRED)]
    h.tick()
    h.tick()
    h.tick()
    assert h.speaker.said == [NOT_DUE, TOO_SOON_SPEECH, EXPIRED]
    assert h.slot.state == "POST_SLOT_OBSERVE" and prompts(h) == []
    h.run_until(lambda: h.slot.done)
    assert h.slot.result == "completed"
    assert h.app.named("task_status")[-1][2] == {
        "outcomes": {"1": "dose_not_due_yet", "2": "dose_too_soon", "3": "dose_expired"}}
    assert len(h.speaker.said) == 3 and h.app.named("confirmation") == []


def test_a_refusal_without_a_sentence_moves_on_silently():
    h = Harness(make_task([dose(1), dose(2)]), speaker=True)
    h.find_patient()
    h.app.fail["monitor_start"] = [refused("dose_not_due_yet", 1, speech_text=None)]   # an older server
    h.tick()
    assert h.speaker.said == [] and h.slot.outcomes[1] == "dose_not_due_yet" and h.slot.index == 1


def test_without_a_voice_the_refusal_is_silent():
    h = Harness(make_task([dose(1), dose(2)]))   # no speaker
    h.find_patient()
    h.app.fail["monitor_start"] = [refused(intk_id=1)]
    h.tick()
    assert h.slot.outcomes[1] == "dose_too_soon" and h.slot.index == 1 and not h.slot.done


def test_failed_speech_falls_back_to_silence_and_never_stops_the_slot():
    h = Harness(make_task([dose(1), dose(2)]), speaker=True)
    h.speaker.error = RuntimeError("synthesis failed")
    h.find_patient()
    h.app.fail["monitor_start"] = [refused(intk_id=1)]
    h.tick()
    assert not h.slot.done and h.slot.outcomes[1] == "dose_too_soon" and h.slot.index == 1
    h.speaker.error = None
    h.verify_and_prompt()
    assert prompts(h) == ["med_prompt:20"] and h.speaker.said == []


def test_no_one_is_told_after_the_patient_walked_away():
    h = watching()
    h.see(identity_status="searching")
    h.tick(1)
    h.tick(AWAY_TIMEOUT)
    assert h.slot.outcomes[1] == "left_pending" and h.slot.state == "MED_PROMPT"
    h.app.fail["monitor_start"] = [refused(intk_id=2)]
    h.tick()
    assert h.slot.outcomes[2] == "dose_too_soon" and h.speaker.said == []


@pytest.mark.parametrize("detail", ["dose_too_soon", "daily_max_reached"])
def test_refused_after_saying_finished_is_explained_and_nothing_more_is_filed(detail):
    h = watching(microphone=True)
    h.tick(1)
    h.voice.say("吃完")
    h.tick(1)
    h.app.fail["confirmation"] = [refused(detail, intk_id=1)]
    h.tick(DONE_GRACE)
    [call] = h.app.named("confirmation")              # one patient_claim; the server alerts family itself
    assert call[1:3] == (1, "patient_claim")
    assert h.speaker.said == [TOO_SOON_SPEECH] and h.slot.outcomes[1] == detail
    assert "confirm_with_caregiver" not in h.clips.played and h.app.named("extra_event") == []
    assert h.slot.state == "MED_PROMPT" and h.slot.index == 1 and h.voice.active is False
    h.tick(DONE_GRACE * 3)
    assert len(h.app.named("confirmation")) == 1 and len(h.speaker.said) == 1


def test_refused_after_a_camera_event_is_explained_and_nothing_more_is_filed():
    h = watching([dose(1)])
    h.see(identity_status="verified", candidate=READY_UNCERTAIN)
    h.app.fail["confirmation"] = [refused("daily_max_reached", intk_id=1)]
    h.tick()
    assert len(h.app.named("confirmation")) == 1 and h.slot.outcomes[1] == "daily_max_reached"
    assert h.speaker.said == [TOO_SOON_SPEECH] and "confirm_with_caregiver" not in h.clips.played
    assert h.slot.state == "POST_SLOT_OBSERVE"
    h.run_until(lambda: h.slot.done)
    assert len(h.app.named("confirmation")) == 1 and len(h.speaker.said) == 1


def test_refused_commit_on_the_frame_stream_is_explained_and_nothing_is_filed():
    """On-robot vision: the frame upload that would have recorded the dose comes back refused."""
    h = watching(microphone=True)
    h.tick(1)
    h.voice.say("吃完")
    h.tick(1)
    h.stream.errors.append(refused("dose_too_soon", intk_id=1))
    h.tick()
    assert h.speaker.said == [TOO_SOON_SPEECH] and h.slot.outcomes[1] == "dose_too_soon"
    assert h.slot.recoveries == 0 and h.slot.state == "MED_PROMPT" and h.slot.index == 1
    h.tick(DONE_GRACE * 2)
    assert h.app.named("confirmation") == []
    assert [call for call in h.app.named("monitor_start") if call[1:] == ("dose", 1)] == [("monitor_start", "dose", 1)]


def test_a_late_refusal_of_a_dose_already_explained_is_not_said_again():
    h = watching()
    h.see(identity_status="verified", candidate=READY_UNCERTAIN)
    h.app.fail["confirmation"] = [refused("dose_too_soon", intk_id=1)]
    h.tick()
    assert h.slot.index == 1 and h.speaker.said == [TOO_SOON_SPEECH]
    h.stream.errors.append(refused("dose_too_soon", intk_id=1))   # an upload still in flight for dose 1
    h.tick()
    assert h.speaker.said == [TOO_SOON_SPEECH] and h.slot.index == 1 and h.slot.outcomes[1] == "dose_too_soon"
    h.verify_and_prompt()
    assert prompts(h) == ["med_prompt:10", "med_prompt:20"]


def test_a_refusal_naming_no_dose_concerns_the_attached_session():
    h = watching()
    h.stream.errors.append(refused("dose_expired", speech_text=EXPIRED))   # body without intk_id
    h.tick()
    assert h.slot.outcomes[1] == "dose_expired" and h.speaker.said == [EXPIRED] and h.slot.index == 1


def test_session_restarted_after_a_loss_and_refused_moves_on_without_looping():
    h = watching()
    h.stream.errors.append(lost())
    h.tick()
    h.app.fail["monitor_start"] = [refused("dose_too_soon", intk_id=1)]   # e.g. taken in the app meanwhile
    h.tick()
    assert h.slot.outcomes[1] == "dose_too_soon" and h.slot.index == 1 and h.slot.state == "MED_PROMPT"
    assert h.speaker.said == [TOO_SOON_SPEECH] and not h.slot.done
    assert len([call for call in h.app.named("monitor_start") if call[1:] == ("dose", 1)]) == 2


def test_english_sentence_is_said_as_sent():
    h = Harness(make_task([dose(1)]), speaker=True, language="en")
    h.find_patient()
    sentence = "You already took this medicine at 12:05 AM. Please don't take it again yet."
    h.app.fail["monitor_start"] = [refused(intk_id=1, speech_text=sentence)]
    h.tick()
    assert h.speaker.said == [sentence]
