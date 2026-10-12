"""SlotSession and the multimodal emotion: told when a dose is watched; its summary filed with the evidence."""

from medcare_reachy.bridge.tests.fakes import Harness, dose, make_task

READY_UNCERTAIN = {"event_id": "ev1", "decision": "uncertain", "confidence": 0.31, "frame_seq": 40, "ready": True}


class FakeMood:
    def __init__(self):
        self.calls = []

    def begin(self, intk_id):
        self.calls.append(("begin", intk_id))

    def end(self):
        self.calls.append(("end",))

    def summary(self):
        return {"intk_id": 1, "samples": 4, "top": "neutral", "confidence": 0.5}


def test_watching_a_dose_brackets_the_mood_and_its_summary_goes_with_the_evidence():
    mood = FakeMood()
    h = Harness(make_task([dose(1), dose(2)]), mood=mood)
    h.find_patient()
    assert mood.calls == []
    h.verify_and_prompt()
    assert h.slot.state == "WATCHING" and mood.calls == [("begin", 1)]
    h.see(identity_status="verified", candidate=READY_UNCERTAIN)
    h.tick()
    assert mood.calls == [("begin", 1), ("end",)]
    evidence = h.app.named("confirmation")[0][-1]
    assert evidence["emotion"] == {"intk_id": 1, "samples": 4, "top": "neutral", "confidence": 0.5, "compute": None}
    h.verify_and_prompt()
    assert mood.calls[-1] == ("begin", 2)


def test_stopping_mid_dose_ends_the_mood_window():
    mood = FakeMood()
    h = Harness(make_task([dose(1)]), mood=mood)
    h.find_patient()
    h.verify_and_prompt()
    h.slot.stop("stop_all")
    h.tick()
    assert h.slot.done and mood.calls[-1] == ("end",)


def test_a_broken_mood_never_touches_the_dose_flow():
    class Broken:
        def begin(self, intk_id):
            raise RuntimeError("x")

        def end(self):
            raise RuntimeError("x")

        def summary(self):
            raise RuntimeError("x")

    h = Harness(make_task([dose(1)]), mood=Broken())
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="verified", candidate=READY_UNCERTAIN)
    h.tick()
    evidence = h.app.named("confirmation")[0][-1]
    assert "emotion" not in evidence and h.slot.state == "POST_SLOT_OBSERVE"
