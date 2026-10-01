import math

import numpy as np
import pytest

from medcare_reachy.bridge import mp_geometry as geo


@pytest.mark.parametrize("size,strides,count", [
    (128, (8, 16, 16, 16), 896),        # face detector
    (192, (8, 16, 16, 16), 2016),       # palm detector
    (224, (8, 16, 32, 32, 32), 2254),   # pose detector
])
def test_anchor_counts_match_the_mediapipe_detectors(size, strides, count):
    anchors = geo.ssd_anchors(size, strides)
    assert anchors.shape == (count, 2)
    assert anchors.min() > 0 and anchors.max() < 1


def test_first_anchors_are_two_per_cell_at_cell_centres():
    anchors = geo.ssd_anchors(128, (8, 16, 16, 16))
    assert np.allclose(anchors[0], anchors[1])
    assert np.allclose(anchors[0], [0.5 / 16, 0.5 / 16])
    assert np.allclose(anchors[2], [1.5 / 16, 0.5 / 16])
    assert np.allclose(anchors[512], [0.5 / 8, 0.5 / 8])   # second map: 6 anchors per cell


def test_decode_and_weighted_nms_merge_one_face_and_keep_another():
    spec = geo.DetectorSpec(128, (8, 16, 16, 16), 6, (-1, 1))
    anchors = geo.ssd_anchors(spec.size, spec.strides)
    raw = np.zeros((len(anchors), 16), np.float32)
    logits = np.full(len(anchors), -10.0, np.float32)
    raw[:, 2:4] = 20.0   # 20 px boxes
    for index, logit in ((100, 3.0), (101, 1.0), (700, 2.0)):
        logits[index] = logit
    raw[101, 0] = 2.0     # same cell, shifted 2 px: merges into anchor 100's cluster
    scores, boxes, keypoints = geo.decode(raw, logits, anchors, spec, 0.5)
    assert len(scores) == 3
    kept = geo.weighted_nms(scores, boxes, keypoints)
    assert len(kept) == 2
    score, box, points = kept[0]
    assert score == pytest.approx(geo.sigmoid(np.array(3.0)))
    w100, w101 = geo.sigmoid(np.array(3.0)), geo.sigmoid(np.array(1.0))
    expected_x = anchors[100, 0] + (2.0 / 128) * w101 / (w100 + w101)
    assert box[0] == pytest.approx(expected_x, abs=1e-6)
    assert points.shape == (6, 2)


def test_rotation_follows_mediapipe_convention():
    assert geo.rotation((0, 0), (10, 0), 0) == pytest.approx(0)
    # End point lower in the image (y down) -> positive rotation.
    assert geo.rotation((0, 0), (10, 10), 0) == pytest.approx(math.pi / 4)
    # Hands/pose: pointing straight up the image is upright.
    assert geo.rotation((0, 10), (0, 0), 90) == pytest.approx(0)


def test_crop_and_project_are_inverses():
    roi = geo.Roi(300.0, 200.0, 120.0, 120.0, 0.6)
    a, b, c, d, e, f = geo.crop_coefficients(roi, 256)
    crop_points = np.array([[0.0, 0.0], [128.0, 128.0], [256.0, 40.0], [17.0, 230.0]])
    via_affine = np.stack([a * crop_points[:, 0] + b * crop_points[:, 1] + c,
                           d * crop_points[:, 0] + e * crop_points[:, 1] + f], axis=1)
    assert np.allclose(geo.project(crop_points, roi, 256), via_affine)
    assert np.allclose(geo.project(np.array([[128.0, 128.0]]), roi, 256), [[300.0, 200.0]])


def test_transform_roi_shifts_along_the_rotated_axis_then_squares_and_scales():
    upright = geo.transform_roi(geo.Roi(100, 100, 40, 20, 0.0), scale=2.0, shift_y=-0.5)
    assert (upright.x, upright.y, upright.width, upright.height) == pytest.approx((100, 90, 80, 80))
    turned = geo.transform_roi(geo.Roi(100, 100, 40, 20, math.pi / 2), scale=1.0, shift_y=-0.5)
    assert (turned.x, turned.y) == pytest.approx((110, 100))


def test_roi_from_points_is_tight_in_the_rotated_frame():
    corners = np.array([[-2, -1], [2, -1], [2, 1], [-2, 1]], dtype=float)
    angle = 0.5
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    points = corners @ rot.T + [50, 60]
    roi = geo.roi_from_points(points, angle)
    assert (roi.x, roi.y, roi.width, roi.height) == pytest.approx((50, 60, 4, 2))


def test_iou_bounds():
    assert geo.iou_bounds((0, 0, 2, 2), (0, 0, 2, 2)) == pytest.approx(1)
    assert geo.iou_bounds((0, 0, 2, 2), (1, 0, 3, 2)) == pytest.approx(1 / 3)
    assert geo.iou_bounds((0, 0, 1, 1), (5, 5, 6, 6)) == 0
