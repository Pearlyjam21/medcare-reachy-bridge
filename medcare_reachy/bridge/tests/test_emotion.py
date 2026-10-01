from pathlib import Path

import numpy as np
import pytest

from medcare_reachy.bridge import emotion

APP_ROOT = Path(__file__).resolve().parents[2]
SERVER_MODEL = APP_ROOT / "models" / "emotion_seed43" / "model_fp32.onnx"


def face(box, mouth=(0.3, 0.35)):
    points = [[0.3, 0.3]] * 9
    points[5] = points[8] = list(mouth)
    return {"box": box, "points": points}


def test_target_face_is_the_best_overlap_with_the_verified_box():
    packet = {"faces": [face([0.6, 0.1, 0.3, 0.3]), face([0.1, 0.1, 0.4, 0.4])], "hands": []}
    assert emotion.target_face(packet, [0.12, 0.1, 0.4, 0.4]) == 1
    assert emotion.target_face(packet, None) is None
    assert emotion.target_face(packet, [0.0, 0.8, 0.1, 0.1]) is None     # nobody near the verified box


def test_mouth_rule_matches_the_server():
    f = face([0.1, 0.1, 0.4, 0.4])
    target = [0.1, 0.1, 0.4, 0.4]
    assert not emotion.mouth_covered(f, [], target)
    assert emotion.mouth_covered(f, [[[0.32, 0.4]] * 13], target)          # wrist within 0.6 x face width
    assert not emotion.mouth_covered(f, [[[0.95, 0.95]] * 13], target)


def test_score_skips_a_covered_mouth_and_reports_the_face_index(monkeypatch):
    engine = emotion.EmotionEngine.__new__(emotion.EmotionEngine)
    monkeypatch.setattr(engine, "predict", lambda frame, box: {"sad": 1.0}, raising=False)
    packet = {"faces": [face([0.1, 0.1, 0.4, 0.4])], "hands": []}
    assert engine.score(None, packet, [0.1, 0.1, 0.4, 0.4]) == {"face_index": 0, "probabilities": {"sad": 1.0}}
    packet["hands"] = [[[0.3, 0.35]] * 13]
    assert engine.score(None, packet, [0.1, 0.1, 0.4, 0.4]) is None


@pytest.mark.skipif(not SERVER_MODEL.exists(), reason="needs the server's emotion model")
def test_robot_scores_match_the_server_emotion_service(tmp_path):
    import shutil

    cv2 = pytest.importorskip("cv2")
    from app.services import emotion_service

    shutil.copy(SERVER_MODEL, tmp_path / emotion.MODEL_FILE)
    shutil.copy(SERVER_MODEL.with_name("model_fp32_metadata.json"), tmp_path / "emotion_seed43_metadata.json")
    robot = emotion.EmotionEngine(tmp_path / emotion.MODEL_FILE)
    rng = np.random.default_rng(3)
    frame = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    frame = cv2.GaussianBlur(frame, (9, 9), 3)          # face-like smoothness, not pure noise
    box = [0.3, 0.2, 0.3, 0.4]
    pixel_box = (int(box[0] * 640), int(box[1] * 480), int(box[2] * 640), int(box[3] * 480))
    crop = emotion_service.crop_face(frame, pixel_box)
    import onnxruntime as ort

    session = ort.InferenceSession(str(SERVER_MODEL), providers=["CPUExecutionProvider"])
    logits = session.run(None, {session.get_inputs()[0].name: emotion_service.prepare_face(crop, 112)})[0][0]
    server = np.exp(logits - logits.max())
    server /= server.sum()
    ours = robot.predict(frame, box)
    assert max(abs(ours[name] - float(server[i])) for i, name in enumerate(emotion.LABELS)) < 0.02
