"""Multimodal emotion: label maps and fusion (as the Space tests them), the governor's fallback rule, and the engine
that fuses the face with what the microphone thread heard."""

import numpy as np
import pytest

from medcare_reachy.bridge.emotion import LABELS as SERVER_LABELS
from medcare_reachy.bridge.multimodal import ComputeGovernor, MultimodalEmotion
from medcare_reachy.bridge.multimodal.engine import to_server
from medcare_reachy.bridge.multimodal.face import FaceBranch
from medcare_reachy.bridge.multimodal.fusion import OK, BranchResult, fuse
from medcare_reachy.bridge.multimodal.governor import LAPTOP, ROBOT
from medcare_reachy.bridge.multimodal.labels import BRANCH_LABELS, EMOTIONS, softmax, to_canonical


# ── labels and fusion ──
def onehot_logits(branch, emotion, high=5.0):
    z = np.zeros(7)
    z[BRANCH_LABELS[branch].index(emotion)] = high
    return z


def probs_for(emotion, p):
    x = np.full(7, (1 - p) / 6)
    x[EMOTIONS.index(emotion)] = p
    return x


W = {"speech": 0.3, "face": 0.5, "text": 0.9}


def test_each_branch_reorders_into_canonical_labels():
    for branch in BRANCH_LABELS:
        for emotion in EMOTIONS:
            assert EMOTIONS[int(to_canonical(branch, onehot_logits(branch, emotion)).argmax())] == emotion


def test_face_branch_labels_are_the_servers():
    assert tuple(BRANCH_LABELS["face"]) == SERVER_LABELS


def test_to_server_names_every_label_once_and_keeps_the_mass():
    probs = probs_for("sad", 0.7)
    out = to_server(probs)
    assert set(out) == set(SERVER_LABELS) and abs(sum(out.values()) - 1) < 1e-9
    assert out["sad"] == pytest.approx(0.7) and out["neutral"] == pytest.approx(0.05)


def test_missing_branches_are_dropped_not_counted_as_neutral():
    only_speech = fuse({"speech": BranchResult(probs_for("angry", 0.6), OK), "face": BranchResult(status="no_face"),
                        "text": BranchResult(status="unsupported_language")}, W)
    assert only_speech.used == {"speech": 1.0}
    np.testing.assert_allclose(only_speech.probs, probs_for("angry", 0.6))
    assert fuse({"face": BranchResult(status="no_face")}, W).probs is None


def test_log_linear_pooling_weights_and_normalisation():
    a, b = probs_for("happy", 0.7), probs_for("sad", 0.7)
    fused = fuse({"speech": BranchResult(a, OK), "face": BranchResult(b, OK)}, W)
    expected = np.exp(0.3 / 0.8 * np.log(a) + 0.5 / 0.8 * np.log(b))
    np.testing.assert_allclose(fused.probs, expected / expected.sum())
    assert fused.top == "sad" and abs(sum(fused.used.values()) - 1) < 1e-12


# ── the governor ──
def test_pinned_modes_never_move():
    assert ComputeGovernor("laptop").on_robot(0.0) is False
    robot = ComputeGovernor("robot")
    for i in range(10):
        robot.failed(RuntimeError("x"), float(i))
    assert robot.on_robot(10.0) is True and robot.fallbacks == 0


def test_auto_falls_back_after_consecutive_failures_and_retries_later():
    g = ComputeGovernor("auto", max_failures=3, recover_after=60.0)
    assert g.compute == ROBOT
    g.failed(RuntimeError("a"), 1.0)
    g.scored(0.1, 2.0)                       # a success resets the count
    g.failed(RuntimeError("b"), 3.0)
    g.failed(RuntimeError("c"), 4.0)
    assert g.compute == ROBOT
    g.failed(RuntimeError("d"), 5.0)
    assert g.compute == LAPTOP and g.fallbacks == 1 and "failed" in g.reason
    assert g.on_robot(64.9) is False
    assert g.on_robot(65.0) is True and g.recoveries == 1 and g.reason == "retry"


def test_auto_falls_back_when_scores_take_longer_than_the_frame_interval():
    g = ComputeGovernor("auto", target_fps=2.0, max_slow=3)
    g.scored(0.6, 1.0)
    g.scored(0.6, 2.0)
    g.scored(0.4, 3.0)                       # one fast score breaks the run
    g.scored(0.6, 4.0)
    g.scored(0.6, 5.0)
    assert g.compute == ROBOT
    g.scored(0.6, 6.0)
    assert g.compute == LAPTOP and "took over 0.50 s" in g.reason


def test_auto_falls_back_when_fewer_than_half_the_due_frames_are_scored_over_a_window():
    g = ComputeGovernor("auto", window=10.0, min_share=0.5)
    t = 0.0
    for i in range(24):                      # due every 0.5 s, scored every third time
        t = i * 0.5
        g.due(t)
        if i % 3 == 0:
            g.scored(0.1, t)
        if t < 10.0:
            assert g.compute == ROBOT, "no verdict before a full window"
    assert g.compute == LAPTOP and "of" in g.reason
    assert g.share(t) is not None and g.share(t) < 0.5


def test_a_robot_that_keeps_up_stays_on_the_robot():
    g = ComputeGovernor("auto")
    for i in range(60):
        g.due(i * 0.5)
        g.scored(0.2, i * 0.5 + 0.2)
    assert g.compute == ROBOT and g.share(30.0) == 1.0 and g.status()["fallbacks"] == 0


# ── the engine ──
class FakeFaceModel:
    """EmotionEngine stand-in: logits in the server's order."""

    def __init__(self, emotion="happy", high=4.0):
        self.emotion, self.high, self.calls = emotion, high, []

    def logits(self, frame, box):
        self.calls.append(list(box))
        z = np.zeros(7)
        z[SERVER_LABELS.index(self.emotion)] = self.high
        return z


class FakeSpeech:
    def __init__(self, emotion="sad", fail=False):
        self.emotion, self.fail, self.calls = emotion, fail, 0

    def score(self, samples):
        self.calls += 1
        if self.fail:
            raise RuntimeError("no voice model")
        return BranchResult(probs_for(self.emotion, 0.8), OK), np.zeros(3)


class FakeText:
    def __init__(self, emotion="sad"):
        self.emotion, self.calls = emotion, []

    def classify(self, text, lang):
        self.calls.append((text, lang))
        if lang != "zh":
            return BranchResult(status="unsupported_language"), text
        return BranchResult(probs_for(self.emotion, 0.9), OK), text


PACKET = {"faces": [{"box": [0.1, 0.1, 0.4, 0.4], "points": [[0.3, 0.3]] * 9}], "hands": []}
BOX = [0.1, 0.1, 0.4, 0.4]
FRAME = np.zeros((480, 640, 3), np.uint8)


def make_engine(speech=None, text=None, clock=None, language="zh-TW"):
    clock = clock or (lambda: 100.0)
    face = FaceBranch(FakeFaceModel(), temperature=6.0)
    return MultimodalEmotion(face, speech, text, language=language, clock=clock), face.engine


def test_alone_the_face_is_reported_as_the_server_expects_untempered():
    mood, model = make_engine()
    report = mood.score(FRAME, PACKET, BOX)
    assert report["face_index"] == 0 and set(report["probabilities"]) == set(SERVER_LABELS)
    expected = softmax(np.where(np.array(SERVER_LABELS) == "happy", 4.0, 0.0))
    assert report["probabilities"]["happy"] == pytest.approx(float(expected.max()))
    assert mood.last["used"] == {"face": 1.0} and mood.last["top"] == "happy"
    assert model.calls == [BOX]


def test_nothing_fair_to_score_gives_no_report():
    mood, model = make_engine()
    covered = {"faces": [{"box": BOX, "points": [[0.3, 0.3]] * 9}], "hands": [[[0.3, 0.3]] * 13]}
    assert mood.score(FRAME, covered, BOX) is None               # a hand at the mouth
    assert mood.score(FRAME, PACKET, None) is None               # nobody verified
    assert mood.score(FRAME, {"faces": [], "hands": []}, BOX) is None
    assert model.calls == []


def test_a_recent_voice_is_fused_with_the_face_and_fades_after_its_ttl():
    now = {"t": 100.0}
    speech, text = FakeSpeech("sad"), FakeText("sad")
    mood, _ = make_engine(speech, text, clock=lambda: now["t"])
    mood.begin(7)
    mood.hear(np.zeros(16000, np.float32), 99.0, 100.0, "我好難過")
    assert speech.calls == 1 and text.calls == [("我好難過", "zh")]
    report = mood.score(FRAME, PACKET, BOX)
    assert mood.last["top"] == "sad" and set(mood.last["used"]) == {"face", "speech", "text"}
    assert report["probabilities"]["sad"] > report["probabilities"]["happy"]
    now["t"] = 111.0                                             # VOICE_TTL (10 s) has passed
    mood.score(FRAME, PACKET, BOX)
    assert mood.last["used"] == {"face": 1.0} and mood.last["top"] == "happy"
    summary = mood.summary()
    assert summary["intk_id"] == 7 and summary["samples"] == 2 and summary["fused_samples"] == 1
    assert summary["modalities"] == {"face": 2, "speech": 1, "text": 1}
    assert set(summary["probabilities"]) == set(EMOTIONS) and summary["top"] in EMOTIONS


def test_hearing_counts_only_while_a_dose_is_watched_and_english_words_are_not_classified():
    speech, text = FakeSpeech(), FakeText()
    mood, _ = make_engine(speech, text, language="en")
    mood.hear(np.zeros(16000, np.float32), 1.0, 2.0, "I feel sad")
    assert speech.calls == 0 and text.calls == []                # not watching: nothing is scored
    mood.begin(1)
    mood.hear(np.zeros(16000, np.float32), 1.0, 2.0, "I feel sad")
    assert speech.calls == 1 and text.calls == [("I feel sad", "en")]
    mood.score(FRAME, PACKET, BOX)
    assert set(mood.last["used"]) == {"face", "speech"}         # English words: unsupported, left out
    mood.end()
    mood.hear(np.zeros(16000, np.float32), 3.0, 4.0, "x")
    assert speech.calls == 1


def test_a_branch_that_keeps_failing_is_switched_off_and_the_rest_carries_on():
    speech = FakeSpeech(fail=True)
    mood, _ = make_engine(speech, FakeText())
    mood.begin(1)
    for _ in range(5):
        mood.hear(np.zeros(16000, np.float32), 1.0, 2.0, "我好難過")
    assert speech.calls == 3 and mood.branches == {"face": True, "speech": False, "text": True}
    mood.score(FRAME, PACKET, BOX)
    assert set(mood.last["used"]) == {"face", "text"}


def test_summary_without_samples_still_says_what_was_available():
    mood, _ = make_engine()
    mood.begin(3)
    assert mood.summary() == {"intk_id": 3, "samples": 0, "modalities": {"face": 0, "speech": 0, "text": 0},
                              "branches": {"face": True, "speech": False, "text": False}}


def test_streamed_frames_are_scored_without_landmarks():
    """No packet (landmarks live on the server): the verified box alone is scored; the server gates the mouth."""
    mood, model = make_engine()
    report = mood.score(FRAME, None, BOX)
    assert report["face_index"] is None and model.calls == [BOX]
