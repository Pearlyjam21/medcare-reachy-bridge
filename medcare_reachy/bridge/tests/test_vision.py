"""VisionEngine wiring with fake models: detection cadence, tracking, pose reuse, packet shape.

Model accuracy against real MediaPipe is checked by the medcare_reachy app's test_vision_accuracy.py, which has
the ONNX files.
"""

import numpy as np
import pytest

from medcare_reachy.bridge import mp_geometry as geo
from medcare_reachy.bridge import vision


class FakeModel:
    """Stands in for OnnxModel: a detector with one strong detection, or a landmarker with centred points."""

    def __init__(self, path):
        self.name = path.name
        self.calls = 0
        if "detector.onnx" in self.name and "landmarks" not in self.name:
            spec = {"face": vision.FACE, "hand": vision.HAND, "pose": vision.POSE}[self.name.split("_")[0]].detector
            self.anchors = len(geo.ssd_anchors(spec.size, spec.strides))
            self.box_values = 4 + 2 * spec.keypoints
            self.outputs = [("boxes", self.anchors * self.box_values), ("scores", self.anchors)]
            self.spec = spec
        else:
            spec = {"face": vision.FACE, "hand": vision.HAND, "pose": vision.POSE}[self.name.split("_")[0]]
            self.spec = spec
            self.outputs = [("points", spec.landmark_count * spec.landmark_values), ("presence", 1)]
            if spec.heatmap:
                self.outputs.append(("heatmap", int(np.prod(spec.heatmap))))

    def output_named_by_size(self, size):
        return next(name for name, count in self.outputs if count == size)

    def run(self, tensor):
        self.calls += 1
        if "boxes" in dict(self.outputs):
            boxes = np.zeros((self.anchors, self.box_values), np.float32)
            boxes[:, 2:4] = self.spec.size * 0.3
            for k in range(self.spec.keypoints):   # keypoints fanned out so rotations are well defined
                boxes[:, 4 + 2 * k] = (k - 1) * 5.0
                boxes[:, 5 + 2 * k] = -k * 5.0
            scores = np.full(self.anchors, -20.0, np.float32)
            scores[self.anchors // 2] = 5.0
            return {"boxes": boxes, "scores": scores}
        size = self.spec.landmark_size
        rng = np.random.default_rng(self.calls)
        points = np.zeros((self.spec.landmark_count, self.spec.landmark_values), np.float32)
        points[:, 0] = size / 2 + rng.uniform(-size / 4, size / 4, self.spec.landmark_count)
        points[:, 1] = size / 2 + rng.uniform(-size / 4, size / 4, self.spec.landmark_count)
        if self.spec.landmark_values == 5:
            points[:, 3] = 4.0   # visibility logit
        out = {"points": points, "presence": np.array([[0.9]], np.float32)}
        if self.spec.heatmap:
            out["heatmap"] = np.full(self.spec.heatmap, -10.0, np.float32)
        return out


@pytest.fixture
def engine(tmp_path):
    for name in vision.MODEL_FILES:
        (tmp_path / name).write_bytes(b"x")
    engine = vision.VisionEngine(tmp_path, model_factory=FakeModel)
    yield engine
    engine.close()


def calls(engine):
    return {name: (tracker.detector.calls, tracker.landmarker.calls) for name, tracker in engine.trackers.items()}


def frame():
    return np.zeros((480, 640, 3), np.uint8)


def test_missing_models_are_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="face_detector.onnx"):
        vision.VisionEngine(tmp_path, model_factory=FakeModel)


def test_packet_has_the_browser_worker_shape(engine):
    packet = engine.process(frame(), 1, 1000.0)
    assert packet["frame_seq"] == 1 and packet["timestamp"] == 1.0
    assert (packet["width"], packet["height"]) == (640, 480)
    assert len(packet["faces"]) == 1 and len(packet["faces"][0]["points"]) == 9
    assert len(packet["hands"]) == 1 and len(packet["hands"][0]) == 13
    pose = packet["poses"][0]
    assert set(pose) == {"nose", "shoulders", "wrists"} and pose["wrists"][0][2] == pytest.approx(0.982, abs=1e-3)


def test_detectors_run_on_a_cadence_and_tracking_fills_the_frames_between(engine):
    packets = [engine.process(frame(), seq, seq * 66.0) for seq in range(1, 7)]
    counts = calls(engine)
    # Six frames, DETECT_EVERY=3: detection on frames 0 and 3 only, because each tracker kept its target.
    assert counts["face"][0] == 2 and counts["hand"][0] == 2
    assert counts["face"][1] >= 6 and counts["hand"][1] >= 6   # a landmark pass every frame (+1 if a re-detection
    # disagreed with the track; the duplicate is merged, never reported twice)
    assert all(len(p["faces"]) == 1 and len(p["hands"]) == 1 for p in packets)
    # Pose runs every POSE_EVERY=2 frames (0, 2, 4) and detects on frame 0 only (frame 3 was a repeat).
    assert counts["pose"][0] == 1 and counts["pose"][1] >= 3


def test_pose_repeats_its_last_result_between_runs(engine):
    first = engine.process(frame(), 1, 0.0)["poses"]
    second = engine.process(frame(), 2, 66.0)["poses"]
    assert second == first


def test_lost_target_is_dropped_and_redetected_on_cadence(engine):
    engine.process(frame(), 1, 0.0)
    hand = engine.trackers["hand"]
    original_run = hand.landmarker.run
    hand.landmarker.run = lambda tensor: {**original_run(tensor), "presence": np.array([[0.1]], np.float32)}
    assert engine.process(frame(), 2, 66.0)["hands"] == []
    assert hand.rois == []
    hand.landmarker.run = original_run
    assert engine.process(frame(), 3, 132.0)["hands"] == []   # frame index 2: not a detection frame
    assert len(engine.process(frame(), 4, 198.0)["hands"]) == 1   # index 3: detects again


def test_reset_forgets_tracked_targets(engine):
    engine.process(frame(), 1, 0.0)
    engine.reset()
    assert all(not t.rois and not t.last and t.frames == 0 for t in engine.trackers.values())
