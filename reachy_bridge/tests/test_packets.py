import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reachy_bridge.packets import FACE_INDICES, HAND_INDICES, build_packet

APP_ROOT = Path(__file__).resolve().parents[2]


def point(x, y, visibility=None):
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=visibility, presence=None)


def face_landmarks(offset=0.0):
    # 478 points on a diagonal so min/max and index picks are easy to predict.
    return [point(0.2 + offset + i / 10_000, 0.3 + i / 20_000) for i in range(478)]


def hand_landmarks(offset=0.0):
    return [point(0.5 + offset + i / 100, 0.6 - i / 100) for i in range(21)]


def pose_landmarks():
    points = [point(i / 100, i / 200, visibility=i / 40) for i in range(33)]
    points[16] = point(0.9, 0.8, visibility=None)
    return points


def results(faces=(), hands=(), poses=()):
    return (SimpleNamespace(face_landmarks=list(faces)),
            SimpleNamespace(hand_landmarks=list(hands)),
            SimpleNamespace(pose_landmarks=list(poses)))


def _tuple_constant(source: str, name: str) -> tuple:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise AssertionError(f"{name} not found")


BACKEND = APP_ROOT / "app" / "services" / "monitor_service.py"
WORKER = APP_ROOT / "frontend_source" / "src" / "workers" / "monitorWorker.ts"


# Only meaningful inside the full MedAiCarePlus repo; a standalone copy of the package has neither file.
@pytest.mark.skipif(not (BACKEND.exists() and WORKER.exists()), reason="needs the full app repository")
def test_indices_match_backend_and_browser_worker():
    backend = BACKEND.read_text(encoding="utf-8")
    assert _tuple_constant(backend, "FACE_INDICES") == FACE_INDICES
    assert _tuple_constant(backend, "HAND_INDICES") == HAND_INDICES
    worker = WORKER.read_text(encoding="utf-8")
    assert f"const faceIndices = [{', '.join(map(str, FACE_INDICES))}];" in worker
    assert f"const handIndices = [{', '.join(map(str, HAND_INDICES))}];" in worker


def test_packet_shape_mirrors_monitor_worker():
    face, hand, pose = results([face_landmarks()], [hand_landmarks(), hand_landmarks(0.1)], [pose_landmarks()])
    packet = build_packet(face, hand, pose, frame_seq=7, timestamp_ms=12_345.0, width=640, height=480)

    assert list(packet) == ["frame_seq", "timestamp", "width", "height", "faces", "hands", "poses"]
    assert packet["frame_seq"] == 7 and packet["timestamp"] == 12.345
    assert (packet["width"], packet["height"]) == (640, 480)

    [f] = packet["faces"]
    assert list(f) == ["box", "points"]
    x0, y0 = 0.2, 0.3
    assert f["box"][:2] == [x0, y0]
    assert abs(f["box"][2] - 477 / 10_000) < 1e-9 and abs(f["box"][3] - 477 / 20_000) < 1e-9
    assert f["points"] == [[0.2 + i / 10_000, 0.3 + i / 20_000] for i in FACE_INDICES]

    assert len(packet["hands"]) == 2
    assert packet["hands"][0] == [[0.5 + i / 100, 0.6 - i / 100] for i in HAND_INDICES]

    [p] = packet["poses"]
    assert list(p) == ["nose", "shoulders", "wrists"]
    assert p["nose"] == [0.0, 0.0]
    assert p["shoulders"] == [[0.11, 0.055], [0.12, 0.06]]
    assert p["wrists"][0] == [0.15, 0.075, 15 / 40]
    assert p["wrists"][1] == [0.9, 0.8, 0]   # missing visibility -> 0, like `?? 0`
    json.dumps(packet)   # plain JSON types only


def test_face_box_is_clamped_to_the_frame():
    landmarks = [point(-0.1, -0.2)] + [point(0.5, 0.5)] * 476 + [point(1.3, 1.4)]
    face, hand, pose = results([landmarks])
    box = build_packet(face, hand, pose, 1, 0, 640, 480)["faces"][0]["box"]
    assert box == [0, 0, 1, 1]


def test_empty_results_give_empty_lists():
    face, hand, pose = results()
    packet = build_packet(face, hand, pose, 1, 66.0, 640, 480)
    assert packet["faces"] == [] and packet["hands"] == [] and packet["poses"] == []


def test_limits_match_worker_model_options():
    face, hand, pose = results([face_landmarks()] * 6, [hand_landmarks()] * 10, [pose_landmarks()] * 5)
    packet = build_packet(face, hand, pose, 1, 0, 640, 480)
    assert len(packet["faces"]) == 4 and len(packet["hands"]) == 8 and len(packet["poses"]) == 4
