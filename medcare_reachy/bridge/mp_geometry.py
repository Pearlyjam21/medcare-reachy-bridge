"""The parts of MediaPipe's Tasks graphs that live outside the .tflite models, in plain numpy.

SSD anchors, detection decoding, weighted non-max suppression, and the region-of-interest maths that turns a
detection (or the previous frame's landmarks) into the rotated square crop the landmark model sees. Constants are
the ones in MediaPipe's face/hand/pose landmarker graph configs. All geometry here is in image pixels.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectorSpec:
    size: int                      # square model input, pixels
    strides: tuple[int, ...]       # SSD feature-map strides; repeated strides share one map
    keypoints: int
    value_range: tuple[float, float]


@dataclass(frozen=True)
class Detection:
    score: float
    box: tuple[float, float, float, float]   # x_center, y_center, width, height (pixels)
    keypoints: np.ndarray                      # (K, 2) pixels


@dataclass(frozen=True)
class Roi:
    x: float          # centre, pixels
    y: float
    width: float
    height: float
    rotation: float   # radians, MediaPipe convention (image y axis points down)

    def bounds(self) -> tuple[float, float, float, float]:
        """Axis-aligned (x1, y1, x2, y2) around the rotated rectangle."""
        cos, sin = abs(math.cos(self.rotation)), abs(math.sin(self.rotation))
        half_w = (self.width * cos + self.height * sin) / 2
        half_h = (self.width * sin + self.height * cos) / 2
        return self.x - half_w, self.y - half_h, self.x + half_w, self.y + half_h


def ssd_anchors(size: int, strides: tuple[int, ...]) -> np.ndarray:
    """Anchor centres (N, 2) in [0, 1], MediaPipe SsdAnchorsCalculator with fixed_anchor_size and aspect 1.0.

    Each layer adds two anchors per cell (aspect 1.0 plus the interpolated scale); consecutive layers with the
    same stride share one feature map.
    """
    centres = []
    layer = 0
    while layer < len(strides):
        last = layer
        while last < len(strides) and strides[last] == strides[layer]:
            last += 1
        per_cell = 2 * (last - layer)
        cells = math.ceil(size / strides[layer])
        ys, xs = np.mgrid[0:cells, 0:cells]
        grid = np.stack([(xs.ravel() + 0.5) / cells, (ys.ravel() + 0.5) / cells], axis=1)
        centres.append(np.repeat(grid, per_cell, axis=0))
        layer = last
    return np.concatenate(centres).astype(np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(values, dtype=np.float64), -100.0, 100.0)))


def decode(raw_boxes: np.ndarray, raw_scores: np.ndarray, anchors: np.ndarray, spec: DetectorSpec,
           threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Model outputs -> (scores, boxes (N, 4) cx/cy/w/h, keypoints (N, K, 2)), all normalised to the input square."""
    raw_boxes = raw_boxes.reshape(len(anchors), -1)
    scores = sigmoid(raw_scores.reshape(len(anchors)))
    keep = np.nonzero(scores >= threshold)[0]
    raw, anchor = raw_boxes[keep], anchors[keep]
    boxes = np.empty((len(keep), 4), dtype=np.float32)
    boxes[:, 0:2] = raw[:, 0:2] / spec.size + anchor
    boxes[:, 2:4] = raw[:, 2:4] / spec.size
    keypoints = raw[:, 4:4 + 2 * spec.keypoints].reshape(len(keep), spec.keypoints, 2) / spec.size + anchor[:, None, :]
    return scores[keep], boxes, keypoints


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    ax1, ay1 = box[0] - box[2] / 2, box[1] - box[3] / 2
    ax2, ay2 = box[0] + box[2] / 2, box[1] + box[3] / 2
    bx1, by1 = others[:, 0] - others[:, 2] / 2, others[:, 1] - others[:, 3] / 2
    bx2, by2 = others[:, 0] + others[:, 2] / 2, others[:, 1] + others[:, 3] / 2
    w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = w * h
    union = box[2] * box[3] + others[:, 2] * others[:, 3] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def weighted_nms(scores: np.ndarray, boxes: np.ndarray, keypoints: np.ndarray, iou_threshold: float = 0.3,
                 limit: int | None = None) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """MediaPipe's WEIGHTED non-max suppression: each cluster becomes the score-weighted mean of its members."""
    order = np.argsort(-scores, kind="stable")
    remaining = order
    out = []
    while remaining.size and (limit is None or len(out) < limit):
        top = remaining[0]
        overlap = _iou(boxes[top], boxes[remaining])
        members = remaining[overlap > iou_threshold]
        if members.size == 0:
            members = remaining[:1]
        weights = scores[members][:, None]
        total = float(weights.sum())
        box = (boxes[members] * weights).sum(axis=0) / total
        points = (keypoints[members] * weights[:, :, None]).sum(axis=0) / total
        out.append((float(scores[top]), box, points))
        remaining = remaining[overlap <= iou_threshold]
    return out


def refine_from_heatmap(points: np.ndarray, heatmap: np.ndarray, crop_size: int, kernel: int = 7,
                        min_confidence: float = 0.5) -> np.ndarray:
    """RefineLandmarksFromHeatmapCalculator: snap each landmark (crop pixels) to its heatmap's local centroid.

    heatmap is (rows, cols, landmarks) of logits covering the whole crop.
    """
    rows, cols = heatmap.shape[:2]
    offset = (kernel - 1) // 2
    refined = points.copy()
    for index in range(min(len(points), heatmap.shape[2])):
        col = int(points[index, 0] / crop_size * cols)
        row = int(points[index, 1] / crop_size * rows)
        if not (0 <= col < cols and 0 <= row < rows):
            continue
        r0, r1 = max(0, row - offset), min(rows, row + offset + 1)
        c0, c1 = max(0, col - offset), min(cols, col + offset + 1)
        confidence = sigmoid(heatmap[r0:r1, c0:c1, index])
        total = float(confidence.sum())
        if confidence.max() < min_confidence or total <= 0:
            continue
        grid_r, grid_c = np.mgrid[r0:r1, c0:c1]
        refined[index, 0] = (float((grid_c * confidence).sum()) / total + 0.5) / cols * crop_size
        refined[index, 1] = (float((grid_r * confidence).sum()) / total + 0.5) / rows * crop_size
    return refined


def normalize_radians(angle: float) -> float:
    return angle - 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))


def rotation(start: tuple[float, float], end: tuple[float, float], target_degrees: float) -> float:
    """MediaPipe: target_angle - atan2(-(y1 - y0), x1 - x0), in pixel coordinates."""
    return normalize_radians(math.radians(target_degrees) - math.atan2(-(end[1] - start[1]), end[0] - start[0]))


def transform_roi(roi: Roi, scale: float, shift_y: float = 0.0, square_long: bool = True) -> Roi:
    """RectTransformationCalculator: shift along the rotated axes, square to the long side, then scale."""
    x, y = roi.x, roi.y
    if shift_y:
        x += -roi.height * shift_y * math.sin(roi.rotation)
        y += roi.height * shift_y * math.cos(roi.rotation)
    width, height = roi.width, roi.height
    if square_long:
        width = height = max(width, height)
    return Roi(x, y, width * scale, height * scale, roi.rotation)


def roi_from_points(points: np.ndarray, angle: float) -> Roi:
    """The tightest rectangle at `angle` around `points` (N, 2) pixels."""
    cos, sin = math.cos(angle), math.sin(angle)
    # Express points in the rectangle's own axes (rotate by -angle), take the box, rotate its centre back.
    local_x = points[:, 0] * cos + points[:, 1] * sin
    local_y = -points[:, 0] * sin + points[:, 1] * cos
    min_x, max_x = float(local_x.min()), float(local_x.max())
    min_y, max_y = float(local_y.min()), float(local_y.max())
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    return Roi(cx * cos - cy * sin, cx * sin + cy * cos, max_x - min_x, max_y - min_y, angle)


def crop_coefficients(roi: Roi, out_size: int) -> tuple[float, ...]:
    """Pillow AFFINE coefficients mapping each crop pixel to its source image pixel."""
    cos, sin = math.cos(roi.rotation), math.sin(roi.rotation)
    a, b = cos * roi.width / out_size, -sin * roi.height / out_size
    d, e = sin * roi.width / out_size, cos * roi.height / out_size
    c = roi.x - (cos * roi.width - sin * roi.height) / 2
    f = roi.y - (sin * roi.width + cos * roi.height) / 2
    return a, b, c, d, e, f


def project(points: np.ndarray, roi: Roi, in_size: int) -> np.ndarray:
    """Landmarks in crop pixels (N, 2) -> image pixels, the inverse of the crop."""
    u = points[:, 0] / in_size - 0.5
    v = points[:, 1] / in_size - 0.5
    cos, sin = math.cos(roi.rotation), math.sin(roi.rotation)
    x = roi.x + cos * u * roi.width - sin * v * roi.height
    y = roi.y + sin * u * roi.width + cos * v * roi.height
    return np.stack([x, y], axis=1)


def iou_bounds(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
