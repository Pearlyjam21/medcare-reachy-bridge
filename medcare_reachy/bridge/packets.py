"""Landmark packets in exactly the shape the browser worker posts (monitorWorker.ts)."""

# Must equal FACE_INDICES / HAND_INDICES in app/services/monitor_service.py (checked by a test).
FACE_INDICES = (1, 10, 13, 14, 33, 61, 152, 263, 291)
HAND_INDICES = (0, 4, 5, 6, 8, 9, 10, 12, 14, 16, 17, 18, 20)
MAX_FACES, MAX_HANDS, MAX_POSES = 4, 8, 4   # model options, and the server's payload limits


def _pair(point) -> list[float]:
    return [float(point.x), float(point.y)]


def _visible(point) -> list[float]:
    # JS `point.visibility ?? 0`: only a missing value becomes 0.
    visibility = getattr(point, "visibility", None)
    return [float(point.x), float(point.y), 0 if visibility is None else float(visibility)]


def _face(landmarks) -> dict:
    xs = [point.x for point in landmarks]
    ys = [point.y for point in landmarks]
    x = max(0, min(xs))
    y = max(0, min(ys))
    return {"box": [float(x), float(y), float(min(1, max(xs)) - x), float(min(1, max(ys)) - y)],
            "points": [_pair(landmarks[index]) for index in FACE_INDICES]}


def _pose(landmarks) -> dict:
    return {"nose": _pair(landmarks[0]),
            "shoulders": [_pair(landmarks[11]), _pair(landmarks[12])],
            "wrists": [_visible(landmarks[15]), _visible(landmarks[16])]}


def build_packet(face_result, hand_result, pose_result, frame_seq: int, timestamp_ms: float,
                 width: int, height: int) -> dict:
    """MediaPipe Tasks results -> landmark packet (without session_id/generation)."""
    return {
        "frame_seq": frame_seq,
        "timestamp": timestamp_ms / 1000,
        "width": width,
        "height": height,
        "faces": [_face(lm) for lm in face_result.face_landmarks[:MAX_FACES]],
        "hands": [[_pair(lm[index]) for index in HAND_INDICES] for lm in hand_result.hand_landmarks[:MAX_HANDS]],
        "poses": [_pose(lm) for lm in pose_result.pose_landmarks[:MAX_POSES]],
    }
