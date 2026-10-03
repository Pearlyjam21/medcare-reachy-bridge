"""SlotSession with the "I finished" listener: consent gating, muting, and the decision when the camera saw nothing."""

from medcare_reachy.bridge.session import AWAY_TIMEOUT, DONE_GRACE
from medcare_reachy.bridge.tests.fakes import Harness, dose, make_task

RECORDED = {"event_id": "ev1", "status": "taken"}
READY_UNCERTAIN = {"event_id": "ev1", "decision": "uncertain", "confidence": 0.31, "frame_seq": 40, "ready": True}


def watching(microphone=True, voice_available=True):
    h = Harness(make_task([dose(1)], microphone=microphone), voice=True)
    h.voice.available = voice_available
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="verified")
    return h


def test_prompt_invites_saying_finished_and_listens_only_while_watching():
    h = watching()
    assert h.clips.played[-2:] == ["med_prompt:10", "say_when_done"]
    assert h.voice.active is True
    assert h.voice.max_holds == 1 and h.voice.holds == 0   # muted during each clip, released after
    h.see(identity_status="verified", recorded=RECORDED)
    h.tick()
    assert h.clips.played[-1] == "thanks" and h.voice.active is False
    assert h.voice.activity == [True, False]


def test_said_finished_without_a_camera_event_goes_to_caregiver_as_patient_claim():
    h = watching()
    h.tick(1)
    h.voice.say("吃完")
    h.tick(1)
    h.tick(DONE_GRACE - 0.5)
    assert not h.app.named("confirmation") and h.slot.state == "WATCHING"
    h.tick(1)
    assert h.app.named("confirmation") == [("confirmation", 1, "patient_claim", {
        "said_done": True, "phrase": "吃完", "camera": "no_event", "degraded": False, "landmark_fps": None})]
    assert h.clips.played[-1] == "confirm_with_caregiver" and h.slot.outcomes[1] == "needs_confirm"
    assert h.voice.active is False


def test_camera_record_after_saying_finished_wins_over_the_claim():
    h = watching()
    h.tick(1)
    h.voice.say()
    h.tick(1)
    h.see(identity_status="verified", recorded=RECORDED)
    h.tick(2)
    assert h.slot.outcomes[1] == "recorded" and h.clips.played[-1] == "thanks"
    assert not h.app.named("confirmation")


def test_uncertain_camera_event_after_saying_finished_carries_the_claim_as_evidence():
    h = watching()
    h.tick(1)
    h.voice.say()
    h.tick(1)
    h.see(identity_status="verified", candidate=READY_UNCERTAIN)
    h.tick(1)
    (call,) = h.app.named("confirmation")
    assert call[2] == "uncertain_detection" and call[3]["said_done"] is True


def test_without_microphone_consent_nothing_listens_or_invites_speech():
    h = watching(microphone=False)
    assert h.clips.played[-1] == "med_prompt:10"
    assert h.voice.active is False and h.voice.activity == []


def test_missing_speech_models_behave_like_no_microphone():
    h = watching(voice_available=False)
    assert h.clips.played[-1] == "med_prompt:10" and h.voice.active is False


def test_withdrawn_microphone_consent_stops_listening_at_the_next_tick():
    h = watching()
    assert h.voice.active is True
    h.slot.task["microphone"] = False   # what the runner does on a heartbeat with microphone=false
    h.tick()
    assert h.voice.active is False


def test_patient_out_of_sight_while_watched_leaves_the_dose_pending():
    h = watching()
    h.see(identity_status="searching")
    h.tick(1)
    h.tick(AWAY_TIMEOUT - 2)
    assert h.slot.state == "WATCHING"
    h.see(identity_status="verified")   # back in view: the clock restarts
    h.tick(1)
    h.see(identity_status="searching")
    h.tick(1)
    h.tick(AWAY_TIMEOUT - 2)
    assert h.slot.state == "WATCHING"
    h.tick(2)
    assert h.slot.outcomes[1] == "left_pending" and h.slot.state != "WATCHING"
    assert not h.app.named("confirmation")
