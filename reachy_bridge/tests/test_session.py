"""SlotSession: every edge of spec 01 §7 with a fake clock, robot, clips, stream, and app client."""

import pytest

from reachy_bridge.app_client import (
    AppUnreachable, BusyOtherClient, NotAuthorised, ServiceUnavailable, SessionLost)
from reachy_bridge.tests.fakes import FakeRobot, Harness, dose, lost, make_task

READY_UNCERTAIN = {"event_id": "ev1", "decision": "uncertain", "confidence": 0.31, "frame_seq": 40, "ready": True}
RECORDED = {"event_id": "ev1", "status": "taken"}


def test_happy_path_two_doses_recorded_then_post_slot_and_wind_down():
    h = Harness(make_task([dose(1), dose(2)]))
    h.find_patient()
    assert h.slot.state == "MED_PROMPT"
    assert h.app.named("monitor_start")[0] == ("monitor_start", "observe", None)
    assert "look_around:0" in h.robot.calls and "wake" in h.robot.calls

    h.verify_and_prompt()
    assert h.app.named("monitor_start")[1] == ("monitor_start", "dose", 1)
    assert ("monitor_end", "s1") in h.app.calls          # observe session ended before the dose session
    assert h.clips.played[-1] == "med_prompt:10"
    h.see(identity_status="verified", recorded=RECORDED)
    h.tick()
    assert h.clips.played[-1] == "thanks" and h.slot.state == "MED_PROMPT"

    h.verify_and_prompt()
    assert h.app.named("monitor_start")[2] == ("monitor_start", "dose", 2)
    assert ("monitor_end", "s2") in h.app.calls          # previous dose session ended on moving on
    h.see(identity_status="verified", recorded=RECORDED)
    h.tick()
    assert h.slot.state == "POST_SLOT_OBSERVE"

    h.tick()
    assert h.app.named("monitor_start")[3] == ("monitor_start", "observe", None)
    h.tick(119)
    assert h.slot.state == "POST_SLOT_OBSERVE"
    h.tick(1.5)
    assert h.slot.state == "WIND_DOWN"
    h.tick()
    assert h.slot.done and h.slot.result == "completed"
    assert h.statuses() == ["searching", "in_progress", "completed"]
    assert h.app.named("task_status")[-1][2] == {"outcomes": {"1": "recorded", "2": "recorded"}}
    assert h.clips.played[0] == "wake_greeting" and h.clips.played[-1] == "wind_down"
    assert h.robot.calls[-1] == "sleep"
    assert h.app.named("confirmation") == []
    assert h.stream.attached is None


def test_missed_retry_announces_with_reminder_clip():
    h = Harness(make_task([dose(1)], reason="missed_retry"))
    h.to_state("SEARCHING")
    assert h.clips.played == ["reminder"]


def test_robot_unreachable_aborts_with_robot_offline():
    h = Harness(make_task([dose(1)]), robot=FakeRobot(reachable=False))
    h.tick()
    assert h.slot.done and h.slot.result == "robot_offline"
    assert h.app.named("task_status") == [("task_status", "aborted",
                                           {"reason": "robot_offline", "outcomes": {}})]
    assert h.app.named("monitor_start") == []


def test_search_scans_head_plays_searching_clip_and_times_out_as_not_found():
    h = Harness(make_task([dose(1)]))
    h.to_state("SEARCHING")
    h.tick()
    h.run_until(lambda: h.slot.done, step=1.0)
    assert h.slot.result == "not_found"
    assert h.statuses() == ["searching", "not_found"]
    scans = [call for call in h.robot.calls if call.startswith("look_around")]
    assert len(scans) >= 140 and scans[:3] == ["look_around:0", "look_around:1", "look_around:2"]
    assert h.clips.played.count("searching") == 4          # every 2 min over 10 min
    assert not any(clip.startswith("med_prompt") for clip in h.clips.played)
    assert ("monitor_end", "s1") in h.app.calls and h.robot.calls[-1] == "sleep"


def test_all_doses_resolved_when_found_plays_already_taken_and_winds_down():
    h = Harness(make_task([dose(1), dose(2), dose(3)]))
    for intk_id, stats in ((1, "taken"), (2, "pending_confirmation"), (3, "skipped")):
        h.app.dose(intk_id)["intake_stats"] = stats   # e.g. taken in the app before Reachy found the patient
    h.find_patient()
    assert h.slot.state == "ALREADY_TAKEN"
    h.run_until(lambda: h.slot.done)
    assert h.clips.played[-2:] == ["already_taken", "wind_down"]
    assert [c for c in h.app.named("monitor_start") if c[1] == "dose"] == []
    assert h.statuses() == ["searching", "in_progress", "completed"]


def test_resolved_doses_are_skipped_and_never_prompted():
    h = Harness(make_task([dose(1, stats="taken"), dose(2), dose(3, stats="skipped")]))
    h.find_patient()
    h.verify_and_prompt()
    assert h.app.named("monitor_start")[-1] == ("monitor_start", "dose", 2)
    h.see(identity_status="verified", recorded=RECORDED)
    h.tick()
    h.tick()
    assert h.slot.state == "POST_SLOT_OBSERVE"
    assert [c for c in h.clips.played if c.startswith("med_prompt")] == ["med_prompt:20"]
    assert h.slot.outcomes == {1: "already_resolved", 2: "recorded", 3: "already_resolved"}


def test_restart_recovery_rereads_doses_and_never_reprompts_resolved_ones():
    # tasks/current after a bridge restart: the task is already in progress.
    task = make_task([dose(1), dose(2), dose(3)], status="in_progress")
    h = Harness(task)
    h.app.dose(1)["intake_stats"] = "taken"
    h.app.dose(2)["intake_stats"] = "pending_confirmation"
    h.find_patient()
    assert h.app.named("tasks_current"), "doses must be re-read before prompting"
    h.verify_and_prompt()
    assert [c for c in h.app.named("monitor_start") if c[1] == "dose"] == [("monitor_start", "dose", 3)]
    assert [c for c in h.clips.played if c.startswith("med_prompt")] == ["med_prompt:30"]
    assert "searching" not in h.statuses() and "in_progress" not in h.statuses()


def test_dose_resolved_between_prompts_is_not_prompted():
    h = Harness(make_task([dose(1), dose(2)]))
    h.find_patient()
    h.verify_and_prompt()
    h.app.dose(2)["intake_stats"] = "taken"      # caregiver/app recorded dose 2 meanwhile
    h.see(identity_status="verified", recorded=RECORDED)
    h.tick()
    h.tick()
    assert h.slot.state == "POST_SLOT_OBSERVE"
    assert h.slot.outcomes[2] == "already_resolved"
    assert [c for c in h.clips.played if c.startswith("med_prompt")] == ["med_prompt:10"]


@pytest.mark.parametrize("supported, auto_record, latest, source", [
    (False, True, {"degraded": False}, "unsupported_dose"),
    (True, False, {"degraded": False}, "auto_record_off"),
    (True, True, {"degraded": False, "auto_commit": False}, "auto_record_off"),
    (True, True, {"degraded": True, "landmark_fps": 9.5}, "degraded"),
    (True, True, {"degraded": False}, "uncertain_detection"),
])
def test_ready_uncommitted_candidate_goes_to_caregiver_confirmation(supported, auto_record, latest, source):
    h = Harness(make_task([dose(1, supported=supported), dose(2)], auto_record=auto_record))
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="verified", candidate=READY_UNCERTAIN, recorded=None, **latest)
    h.tick()
    [call] = h.app.named("confirmation")
    assert call[1:3] == (1, source)
    evidence = call[3]
    assert evidence["event_id"] == "ev1" and evidence["confidence"] == 0.31
    assert evidence["degraded"] is bool(latest.get("degraded"))
    assert h.clips.played[-1] == "confirm_with_caregiver"
    assert h.slot.outcomes[1] == "needs_confirm"
    assert h.slot.state == "MED_PROMPT" and h.slot.index == 1   # continues with the next dose


def test_confirmation_retry_after_success_is_idempotent():
    h = Harness(make_task([dose(1)], auto_record=False))
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="verified", candidate=READY_UNCERTAIN)
    h.app.dose(1)["intake_stats"] = "pending_confirmation"   # first POST landed, response was lost
    h.app.fail["confirmation"] = [lost("Dose is not pending")]
    h.tick()
    assert h.slot.outcomes[1] == "needs_confirm"
    assert h.clips.played[-1] == "confirm_with_caregiver"


def test_confirmation_unreachable_is_retried_next_tick():
    h = Harness(make_task([dose(1)], auto_record=False))
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="verified", candidate=READY_UNCERTAIN)
    h.app.fail["confirmation"] = [AppUnreachable("down")]
    h.tick()
    assert h.slot.state == "WATCHING"
    h.tick()
    assert len(h.app.named("confirmation")) == 2 and h.slot.outcomes[1] == "needs_confirm"


def test_pending_candidate_is_given_time_before_help_clip():
    h = Harness(make_task([dose(1)]))
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="verified", candidate={**READY_UNCERTAIN, "ready": False})
    h.tick(200)
    assert "help" not in h.clips.played and h.app.named("confirmation") == []
    h.see(identity_status="verified", candidate=None)
    h.tick()
    assert h.clips.played[-1] == "help"


def test_watch_timeout_plays_help_then_leaves_dose_pending():
    h = Harness(make_task([dose(1), dose(2)]))
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="verified")
    h.tick(179)
    assert "help" not in h.clips.played
    h.tick(1.5)
    assert h.clips.played[-1] == "help" and h.slot.state == "WATCHING"
    h.tick(179)
    assert h.slot.state == "WATCHING"
    h.tick(1.5)
    assert h.slot.outcomes[1] == "left_pending" and h.slot.state == "MED_PROMPT"
    assert h.app.named("confirmation") == []
    assert h.clips.played.count("help") == 1


def test_head_is_held_still_while_watching_and_post_slot():
    h = Harness(make_task([dose(1)]))
    h.find_patient()
    scans_before = sum(call.startswith("look_around") for call in h.robot.calls)
    h.verify_and_prompt()
    assert h.robot.calls[-1] == "hold_head"
    h.see(identity_status="verified")
    for _ in range(20):
        h.tick(5)
    h.see(identity_status="verified", recorded=RECORDED)
    h.tick()
    h.run_until(lambda: h.slot.done)
    assert sum(call.startswith("look_around") for call in h.robot.calls) == scans_before
    assert h.robot.calls.count("hold_head") >= 3   # found, prompt, post-slot


def test_wrong_person_while_searching_waits_and_never_prompts():
    h = Harness(make_task([dose(1)]))
    h.to_state("SEARCHING")
    h.tick()
    h.see(identity_status="mismatch")
    h.tick()
    h.tick(20)
    assert "waiting_for_patient" not in h.clips.played
    h.tick(0.5)
    assert h.clips.played.count("waiting_for_patient") == 1
    h.see(identity_status="ambiguous")
    h.tick(30)
    assert h.clips.played.count("waiting_for_patient") == 1   # once per episode
    assert h.slot.state == "SEARCHING"
    assert not any(clip.startswith("med_prompt") for clip in h.clips.played)
    h.see(identity_status="searching")                        # bystander left: episode resets
    h.tick()
    h.see(identity_status="mismatch")
    h.tick()
    h.tick(21)
    assert h.clips.played.count("waiting_for_patient") == 2


def test_prompt_waits_for_verification_in_the_dose_session():
    h = Harness(make_task([dose(1), dose(2)]))
    h.find_patient()
    h.tick()
    h.see(identity_status="mismatch")
    h.tick()
    h.tick(21)
    assert h.clips.played[-1] == "waiting_for_patient"
    assert not any(clip.startswith("med_prompt") for clip in h.clips.played)
    h.tick(600)
    assert h.slot.state == "WIND_DOWN"
    assert h.slot.outcomes == {1: "not_verified", 2: "not_verified"}


def test_wrong_person_while_watching_plays_waiting_clip():
    h = Harness(make_task([dose(1)]))
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="ambiguous")
    h.tick()
    h.tick(21)
    assert h.clips.played[-1] == "waiting_for_patient" and h.slot.state == "WATCHING"


def test_extra_events_are_forwarded_once_per_event_id():
    h = Harness(make_task([dose(1)]))
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="verified", recorded=RECORDED)
    h.tick()
    h.tick()   # post-slot observe session starts
    assert h.slot.state == "POST_SLOT_OBSERVE"
    extra = [{"event_id": "x1", "decision": "uncertain", "confidence": 0.3, "frame_seq": 9}]
    h.see(identity_status="verified", extra_events=extra)
    h.app.fail["extra_event"] = [AppUnreachable("blip")]
    h.tick()
    h.tick()
    h.see(identity_status="verified", extra_events=extra + [
        {"event_id": "x2", "decision": "confirmed", "confidence": 0.7, "frame_seq": 30}])
    h.tick()
    h.tick()
    sent = h.app.named("extra_event")
    assert sent == [("extra_event", "x1", "uncertain", 0.3), ("extra_event", "x1", "uncertain", 0.3),
                    ("extra_event", "x2", "confirmed", 0.7)]


def test_extra_events_in_the_dose_session_after_a_record_are_forwarded():
    h = Harness(make_task([dose(1), dose(2)]))
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="verified", recorded=RECORDED,
          extra_events=[{"event_id": "x9", "decision": "confirmed", "confidence": 0.8}])
    h.tick()
    assert h.app.named("extra_event") == [("extra_event", "x9", "confirmed", 0.8)]
    assert h.slot.outcomes[1] == "recorded"


def test_session_lost_restarts_the_current_dose_session_without_reprompting():
    h = Harness(make_task([dose(1)]))
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="verified")
    h.tick(100)
    prompts = h.clips.played.count("med_prompt:10")
    h.stream.errors.append(lost())
    h.tick()
    assert h.stream.attached is None and h.slot.state == "WATCHING"
    reads = len(h.app.named("tasks_current"))
    h.tick()
    assert len(h.app.named("tasks_current")) == reads + 1          # re-read before continuing
    assert h.app.named("monitor_start")[-1] == ("monitor_start", "dose", 1)
    assert h.clips.played.count("med_prompt:10") == prompts
    h.see(identity_status="verified")
    h.tick(81)   # the watch timer kept running across the restart
    assert h.clips.played[-1] == "help"


def test_session_lost_after_the_dose_was_recorded_moves_on():
    h = Harness(make_task([dose(1), dose(2)]))
    h.find_patient()
    h.verify_and_prompt()
    h.app.dose(1)["intake_stats"] = "taken"   # committed just before the app restarted
    h.stream.errors.append(lost())
    h.tick()
    h.tick()
    assert h.slot.outcomes[1] == "recorded" and h.slot.state == "MED_PROMPT" and h.slot.index == 1


def test_repeated_session_loss_aborts():
    h = Harness(make_task([dose(1)]))
    h.find_patient()
    h.verify_and_prompt()
    for _ in range(12):
        h.stream.errors.append(lost())
        h.tick()
        if h.slot.done:
            break
    assert h.slot.done and h.slot.result == "session_lost"
    assert h.statuses()[-1] == "aborted"


def test_dose_start_refused_marks_dose_unavailable_and_continues():
    h = Harness(make_task([dose(1), dose(2)]))
    h.find_patient()
    h.app.fail["monitor_start"] = [lost("Dose is unavailable or does not belong to this account")]
    h.tick()
    assert h.slot.outcomes[1] == "unavailable" and h.slot.index == 1
    h.verify_and_prompt()
    assert [c for c in h.clips.played if c.startswith("med_prompt")] == ["med_prompt:20"]


def test_busy_other_client_waits_and_retries_every_30_seconds():
    h = Harness(make_task([dose(1)]))
    h.find_patient()
    h.app.fail["monitor_start"] = [BusyOtherClient(status=409, detail="busy_other_client")] * 2
    h.tick()
    assert h.slot.state == "WAITING_OTHER_CLIENT"
    h.tick()
    assert h.clips.played[-1] == "waiting_for_tablet"
    h.tick(29)
    assert h.slot.state == "WAITING_OTHER_CLIENT"
    h.tick(1.5)
    assert h.slot.state == "MED_PROMPT"
    h.tick()                                   # second attempt still busy
    assert h.slot.state == "WAITING_OTHER_CLIENT"
    h.tick(31)
    h.tick()                                   # third attempt succeeds
    assert h.slot.state == "MED_PROMPT" and h.stream.attached
    assert h.clips.played.count("waiting_for_tablet") == 1
    h.see(identity_status="verified")
    h.tick()
    assert h.slot.state == "WATCHING"


def test_busy_other_client_until_task_expiry_aborts():
    h = Harness(make_task([dose(1)], expires_in=200))
    h.find_patient()
    h.app.fail["monitor_start"] = [BusyOtherClient(status=409, detail="busy_other_client")] * 50
    h.run_until(lambda: h.slot.done, step=5)
    assert h.slot.result == "busy_other_client" and h.statuses()[-1] == "aborted"
    assert h.robot.calls[-1] == "sleep"


def test_app_unreachable_for_10_seconds_fails_closed():
    h = Harness(make_task([dose(1)]))
    h.find_patient()
    h.verify_and_prompt()
    h.app.unreachable_since = h.clock.t
    h.stream.errors.append(AppUnreachable("landmarks failed"))
    h.tick(5)
    assert not h.slot.done
    posts = len(h.app.calls)
    h.tick(5)
    assert h.slot.done and h.slot.result == "app_unreachable"
    assert h.robot.calls[-1] == "sleep" and h.stream.attached is None
    assert len(h.app.calls) == posts                     # no app calls while failing closed


def test_brief_app_outage_is_tolerated():
    h = Harness(make_task([dose(1)]))
    h.to_state("SEARCHING")
    h.app.fail["monitor_start"] = [AppUnreachable("x")]
    h.app.unreachable_since = h.clock.t
    h.tick()
    h.app.unreachable_since = None                        # next call succeeded
    h.tick(3)
    assert not h.slot.done and h.stream.attached


def test_not_authorised_fails_closed():
    h = Harness(make_task([dose(1)]))
    h.find_patient()
    h.stream.errors.append(NotAuthorised(status=403, detail="consent_required"))
    h.tick()
    assert h.slot.done and h.slot.result == "not_authorised" and h.robot.calls[-1] == "sleep"
    assert "aborted" not in h.statuses()


def test_heartbeat_stop_all_stops_the_slot():
    h = Harness(make_task([dose(1)]))
    h.find_patient()
    h.verify_and_prompt()
    h.slot.stop("stop_all")
    h.tick()
    assert h.slot.done and h.slot.result == "stop_all"
    assert ("monitor_end", "s2") in h.app.calls and h.statuses()[-1] == "aborted"
    assert h.robot.calls[-1] == "sleep" and h.stream.attached is None


def test_model_not_ready_aborts_the_task():
    h = Harness(make_task([dose(1)]))
    h.to_state("SEARCHING")
    h.app.fail["monitor_start"] = [ServiceUnavailable(status=503)]
    h.tick()
    assert h.slot.done and h.slot.result == "model_not_ready"
    assert h.app.named("task_status")[-1] == ("task_status", "aborted", {"reason": "model_not_ready", "outcomes": {}})


def test_task_gone_on_reread_stops_without_status():
    h = Harness(make_task([dose(1)]))
    h.to_state("SEARCHING")
    h.tick()
    h.app.current_task = None
    h.see(identity_status="verified")
    h.tick()
    assert h.slot.done and h.slot.result == "task_gone"
    assert h.statuses() == ["searching"]


def test_illegal_status_transition_is_logged_not_fatal():
    h = Harness(make_task([dose(1)]))
    h.app.fail["task_status"] = [SessionLost(status=409, detail="illegal transition")]
    h.to_state("ANNOUNCE")
    assert not h.slot.done and h.slot.status == "searching"


def test_task_not_found_means_the_lease_is_gone():
    from reachy_bridge.app_client import RequestRejected

    h = Harness(make_task([dose(1)]))
    h.find_patient()
    h.app.fail["monitor_start"] = [RequestRejected(status=404, detail="Task not found")]
    h.tick()
    assert h.slot.done and h.slot.result == "task_gone"
    assert "aborted" not in h.statuses() and h.robot.calls[-1] == "sleep"


def test_rejected_extra_event_does_not_abort_the_slot():
    from reachy_bridge.app_client import RequestRejected

    h = Harness(make_task([dose(1)]))
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="verified", extra_events=[{"event_id": "bad", "decision": "uncertain", "confidence": 0.2}])
    h.app.fail["extra_event"] = [RequestRejected(status=422, detail="Invalid event id")]
    h.tick()
    h.tick()
    assert not h.slot.done and len(h.app.named("extra_event")) == 1


def test_process_shutdown_ends_the_session_but_keeps_the_lease():
    h = Harness(make_task([dose(1)]))
    h.find_patient()
    h.verify_and_prompt()
    h.slot.stop("bridge_shutdown", abort=False)
    h.tick()
    assert h.slot.done and ("monitor_end", "s2") in h.app.calls
    assert h.statuses() == ["searching", "in_progress"] and h.robot.calls[-1] == "sleep"
