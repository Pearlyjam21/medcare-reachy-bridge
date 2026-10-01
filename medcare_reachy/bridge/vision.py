"""Face, hand and pose landmarks on ONNX Runtime, mirroring the MediaPipe Tasks graphs the browser worker uses.

The Reachy Mini Wireless's Raspberry Pi 4 can't load the MediaPipe library: its ARM64 builds need AES instructions
the Pi lacks and die with SIGILL. So the same Tasks models (face_landmarker, hand_landmarker, pose_landmarker_lite),
converted to ONNX, run here on ONNX Runtime, which reachy-mini itself already runs on the robot. The graph logic
around the models (anchors, NMS, crops, tracking) is in mp_geometry.

Like MediaPipe's VIDEO mode, each landmark model re-crops around its own previous result. To fit a Pi 4 the
detectors run only every DETECT_EVERY frames (the browser runs them every frame; the face detector is cheap
enough to run every frame while no face is tracked), and pose, which the server uses only to decide whose hand is
whose, runs every POSE_EVERY frames and repeats its last result in between.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np

from medcare_reachy.bridge import mp_geometry as geo
from medcare_reachy.bridge.packets import MAX_FACES, MAX_HANDS, MAX_POSES, build_packet

MODEL_FILES = (
    "face_detector.onnx", "face_landmarks_detector.onnx",
    "hand_detector.onnx", "hand_landmarks_detector.onnx",
    "pose_detector.onnx", "pose_landmarks_detector.onnx",
)
DETECT_EVERY = 3
POSE_EVERY = 2
MIN_DETECTION = 0.5      # Tasks defaults: min_*_detection_confidence, min_*_presence_confidence
MIN_PRESENCE = 0.5
SAME_INSTANCE_IOU = 0.5  # AssociationNormRectCalculator's min_similarity_threshold


class Point:
    __slots__ = ("x", "y", "visibility")

    def __init__(self, x: float, y: float, visibility: float | None = None):
        self.x, self.y, self.visibility = x, y, visibility


@dataclass(frozen=True)
class PipelineSpec:
    detector: geo.DetectorSpec
    landmark_file: str
    landmark_size: int
    landmark_values: int                      # numbers per landmark in the model's screen-landmark output
    landmark_count: int
    max_instances: int
    roi_from_detection: Callable[[geo.Detection], geo.Roi]
    roi_from_landmarks: Callable[[np.ndarray], geo.Roi]
    heatmap: tuple[int, int, int] | None = None   # (rows, cols, landmarks) refinement heatmap output
    cheap_detector: bool = False                  # detect every frame while nothing is tracked


def _face_roi(det: geo.Detection) -> geo.Roi:
    angle = geo.rotation(det.keypoints[0], det.keypoints[1], 0)            # right eye -> left eye
    return geo.transform_roi(geo.Roi(*det.box, angle), scale=1.5)


def _face_roi_from_landmarks(points: np.ndarray) -> geo.Roi:
    angle = geo.rotation(points[33], points[263], 0)
    (x1, y1), (x2, y2) = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    return geo.transform_roi(geo.Roi((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, angle), scale=1.5)


def _hand_roi(det: geo.Detection) -> geo.Roi:
    angle = geo.rotation(det.keypoints[0], det.keypoints[2], 90)           # wrist -> middle finger base
    return geo.transform_roi(geo.Roi(*det.box, angle), scale=2.6, shift_y=-0.5)


_HAND_PARTIAL = [0, 1, 2, 5, 6, 9, 10, 13, 14, 17, 18]


def _hand_roi_from_landmarks(points: np.ndarray) -> geo.Roi:
    # HandLandmarksToRectCalculator: wrist -> blend of the index/ring/middle PIP joints, rotated box of the palm.
    wrist = points[0, :2]
    toward = ((points[6, :2] + points[14, :2]) / 2 + points[10, :2]) / 2
    angle = geo.rotation(wrist, toward, 90)
    roi = geo.roi_from_points(points[_HAND_PARTIAL, :2], angle)
    return geo.transform_roi(roi, scale=2.0, shift_y=-0.1)


def _alignment_roi(center: np.ndarray, scale_point: np.ndarray) -> geo.Roi:
    size = 2 * float(math.hypot(*(scale_point - center)))
    angle = geo.rotation(center, scale_point, 90)
    return geo.transform_roi(geo.Roi(float(center[0]), float(center[1]), size, size, angle), scale=1.25)


FACE = PipelineSpec(geo.DetectorSpec(128, (8, 16, 16, 16), 6, (-1.0, 1.0)), "face_landmarks_detector.onnx",
                    256, 3, 478, MAX_FACES, _face_roi, _face_roi_from_landmarks, cheap_detector=True)
HAND = PipelineSpec(geo.DetectorSpec(192, (8, 16, 16, 16), 7, (0.0, 1.0)), "hand_landmarks_detector.onnx",
                    224, 3, 21, MAX_HANDS, _hand_roi, _hand_roi_from_landmarks)
POSE = PipelineSpec(geo.DetectorSpec(224, (8, 16, 32, 32, 32), 4, (-1.0, 1.0)), "pose_landmarks_detector.onnx",
                    256, 5, 39, MAX_POSES,
                    lambda det: _alignment_roi(det.keypoints[0], det.keypoints[1]),   # hip centre, body scale
                    lambda points: _alignment_roi(points[33, :2], points[34, :2]), heatmap=(64, 64, 39))
LANDMARK_RANGE = (0.0, 1.0)


def crop(image, roi: geo.Roi, size: int, value_range: tuple[float, float]) -> np.ndarray:
    """Rotated ROI of a Pillow RGB image -> (1, size, size, 3) float32 tensor; outside the frame is black."""
    from PIL import Image

    patch = image.transform((size, size), Image.Transform.AFFINE, geo.crop_coefficients(roi, size),
                            resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    low, high = value_range
    return (np.asarray(patch, dtype=np.float32) * ((high - low) / 255.0) + low)[None]


class OnnxModel:
    def __init__(self, path: Path, threads: int = 1):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        # Models converted from TFLite take NHWC; OpenCV Zoo's pose detector takes NCHW.
        self.channels_first = len(model_input.shape) == 4 and model_input.shape[1] == 3
        self.outputs = [(o.name, int(np.prod([d for d in o.shape if isinstance(d, int)]))) for o in self.session.get_outputs()]

    def output_named_by_size(self, size: int) -> str:
        """The first output with `size` elements (model output order), e.g. 63 = 21 hand landmarks x 3."""
        for name, count in self.outputs:
            if count == size:
                return name
        raise ValueError(f"model has no output of size {size}: {self.outputs}")

    def run(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        if self.channels_first:
            tensor = np.ascontiguousarray(tensor.transpose(0, 3, 1, 2))
        values = self.session.run(None, {self.input_name: tensor})
        return {name: value for (name, _), value in zip(self.outputs, values)}


class Tracker:
    """One MediaPipe landmarker (detector + landmark model) in VIDEO mode, for up to max_instances people/hands."""

    def __init__(self, spec: PipelineSpec, detector, landmarker, detect_every: int = DETECT_EVERY,
                 run_every: int = 1):
        self.spec, self.detector, self.landmarker = spec, detector, landmarker
        self.detect_every, self.run_every = detect_every, run_every
        self.last: list[np.ndarray] = []
        self.anchors = geo.ssd_anchors(spec.detector.size, spec.detector.strides)
        box_values = 4 + 2 * spec.detector.keypoints
        self.box_output = detector.output_named_by_size(len(self.anchors) * box_values)
        self.score_output = detector.output_named_by_size(len(self.anchors))
        self.landmark_output = landmarker.output_named_by_size(spec.landmark_count * spec.landmark_values)
        self.presence_output = landmarker.output_named_by_size(1)
        self.heatmap_output = landmarker.output_named_by_size(int(np.prod(spec.heatmap))) if spec.heatmap else None
        self.rois: list[geo.Roi] = []
        self.frames = 0

    def detect(self, image) -> list[geo.Roi]:
        width, height = image.size
        side = float(max(width, height))
        square = geo.Roi(width / 2, height / 2, side, side, 0.0)   # letterbox, like the Tasks detector graphs
        det = self.spec.detector
        outputs = self.detector.run(crop(image, square, det.size, det.value_range))
        scores, boxes, keypoints = geo.decode(outputs[self.box_output], outputs[self.score_output],
                                              self.anchors, det, MIN_DETECTION)
        rois = []
        for score, box, points in geo.weighted_nms(scores, boxes, keypoints, limit=self.spec.max_instances):
            to_px = lambda u, v: ((u - 0.5) * side + width / 2, (v - 0.5) * side + height / 2)  # noqa: E731
            cx, cy = to_px(box[0], box[1])
            pixel_points = np.array([to_px(u, v) for u, v in points], dtype=np.float32)
            rois.append(self.spec.roi_from_detection(
                geo.Detection(score, (cx, cy, float(box[2] * side), float(box[3] * side)), pixel_points)))
        return rois

    def landmarks(self, image, roi: geo.Roi) -> tuple[float, np.ndarray]:
        size = self.spec.landmark_size
        outputs = self.landmarker.run(crop(image, roi, size, LANDMARK_RANGE))
        raw = outputs[self.landmark_output].reshape(self.spec.landmark_count, self.spec.landmark_values)
        presence = float(outputs[self.presence_output].reshape(-1)[0])
        if not 0.0 <= presence <= 1.0:
            presence = float(geo.sigmoid(np.array(presence)))
        points = raw.copy()
        if self.heatmap_output is not None:
            heatmap = outputs[self.heatmap_output].reshape(self.spec.heatmap)
            points[:, :2] = geo.refine_from_heatmap(raw[:, :2], heatmap, size)
        points[:, :2] = geo.project(points[:, :2], roi, size)
        return presence, points

    def step(self, image) -> list[np.ndarray]:
        """Landmarks (N, values) in image pixels for everyone found this frame."""
        index = self.frames
        self.frames += 1
        if index % self.run_every:
            return self.last
        rois = list(self.rois)
        due = index % self.detect_every == 0 or (not rois and self.spec.cheap_detector)
        if len(rois) < self.spec.max_instances and due:
            for candidate in self.detect(image):
                if len(rois) >= self.spec.max_instances:
                    break
                if all(geo.iou_bounds(candidate.bounds(), r.bounds()) < SAME_INSTANCE_IOU for r in rois):
                    rois.append(candidate)
        found, next_rois = [], []
        for roi in rois:
            presence, points = self.landmarks(image, roi)
            if presence < MIN_PRESENCE:
                continue
            next_roi = self.spec.roi_from_landmarks(points)
            if any(geo.iou_bounds(next_roi.bounds(), r.bounds()) >= SAME_INSTANCE_IOU for r in next_rois):
                continue   # two crops converged on the same face/hand/person
            next_rois.append(next_roi)
            found.append(points)
        self.rois, self.last = next_rois, found
        return found


def _normalised(points: np.ndarray, width: int, height: int, visibility: bool = False) -> list[Point]:
    out = []
    for row in points:
        vis = float(geo.sigmoid(np.array(row[3]))) if visibility else None
        out.append(Point(float(row[0]) / width, float(row[1]) / height, vis))
    return out


class VisionEngine:
    """Face (4) + hand (8) + pose-lite (4) landmarks per frame, in the browser worker's packet format."""

    def __init__(self, models_dir: Path, detect_every: int = DETECT_EVERY, pose_every: int = POSE_EVERY,
                 model_factory=OnnxModel):
        models_dir = Path(models_dir)
        missing = [name for name in MODEL_FILES if not (models_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"vision models missing in {models_dir}: {', '.join(missing)}")
        load = lambda name: model_factory(models_dir / name)   # noqa: E731
        self.trackers = {
            "face": Tracker(FACE, load("face_detector.onnx"), load(FACE.landmark_file), detect_every),
            "hand": Tracker(HAND, load("hand_detector.onnx"), load(HAND.landmark_file), detect_every),
            "pose": Tracker(POSE, load("pose_detector.onnx"), load(POSE.landmark_file), detect_every, pose_every),
        }
        # One thread per landmarker: ONNX Runtime and Pillow release the GIL, so the three run on separate cores.
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="medcare-vision")

    def process(self, frame_bgr: np.ndarray, frame_seq: int, timestamp_ms: float) -> dict:
        from PIL import Image

        height, width = frame_bgr.shape[:2]
        image = Image.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
        futures = {name: self._pool.submit(tracker.step, image) for name, tracker in self.trackers.items()}
        faces, hands, poses = (futures[name].result() for name in ("face", "hand", "pose"))
        return build_packet(
            SimpleNamespace(face_landmarks=[_normalised(p, width, height) for p in faces]),
            SimpleNamespace(hand_landmarks=[_normalised(p, width, height) for p in hands]),
            SimpleNamespace(pose_landmarks=[_normalised(p, width, height, visibility=True) for p in poses]),
            frame_seq, timestamp_ms, width, height)

    def reset(self) -> None:
        for tracker in self.trackers.values():
            tracker.rois, tracker.last, tracker.frames = [], [], 0

    def close(self) -> None:
        self._pool.shutdown(wait=True)
