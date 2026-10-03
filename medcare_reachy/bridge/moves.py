"""Pollen's recorded emotion moves (the Reachy Mini Emotions Library), played inside Reachy's own gesture stream.

Reachy thinks with "inquiring3" ("A fast movement that lets you ask a question", 2.92 s at 50 Hz), kept in moves/
(see NOTICE there) so that it works offline whatever the robot's Hugging Face cache holds. Its sound is not kept.

The SDK's way of playing a move doesn't fit a conversation: play_move() blocks for the whole move, jumps to the
move's first frame (inquiring3's is not the neutral pose: the head 3.6° and 12 mm off, the body 6.9°, the left
antenna 20°) and plays the move's sound; cancel_move() also stops the app's audio output, which carries Reachy's
speech; the daemon's recorded-move route always plays the sound and ignores other targets while it runs. Here a move
is data only: its offsets from its first frame, which gestures.py adds to the neutral pose like its other motions,
so it starts from where the robot is, fades out the moment Reachy speaks, and never touches the audio.
"""

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from medcare_reachy.bridge.gestures import Pose

MOVES_DIR = Path(__file__).resolve().parent / "moves"
THINKING_MOVE = "inquiring3"
# The files as taken from pollen-robotics/reachy-mini-emotions-library at revision 873ae49 (7 Jul 2026).
SHA256 = {"inquiring3": "dcaaf83b8ee6ec19cde32df8fea9117103fc9ca55ac52c96168d0f11abaafc86"}


def head_angles(pose) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees of a head pose's rotation (4x4 or 3x3): the inverse of media.head_pose."""
    r = np.asarray(pose, dtype=float)[:3, :3]
    roll = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    yaw = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


class RecordedMove:
    """A recorded move as offsets from its first frame (gestures.Pose: head roll, pitch, yaw, body yaw and antennas
    in degrees, head position in mm), interpolated linearly between frames as the SDK does."""

    def __init__(self, data: dict, name: str = ""):
        times = np.asarray(data["time"], dtype=float)
        frames = data["set_target_data"]
        if len(frames) != len(times) or len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError(f"move {name}: frames and times don't match")
        first = np.asarray(frames[0]["head"], dtype=float)
        first_antennas = np.asarray(frames[0]["antennas"], dtype=float)
        first_body = float(frames[0].get("body_yaw", 0.0))
        rows = []
        for frame in frames:
            head = np.asarray(frame["head"], dtype=float)
            yaw, pitch, roll = head_angles(head[:3, :3] @ first[:3, :3].T)   # the turn since the first frame
            right, left = np.degrees(np.asarray(frame["antennas"], dtype=float) - first_antennas)
            body = math.degrees(float(frame.get("body_yaw", 0.0)) - first_body)
            rows.append((roll, pitch, yaw, body, right, left, *((head[:3, 3] - first[:3, 3]) * 1000)))
        self.name = name
        self.description = data.get("description", "")
        self.times = times - times[0]
        # One row per frame, in Pose's field order. The recorded matrices are rounded to 6 decimals, so the first
        # frame's own turn comes out ~1e-5° rather than 0: taken off, the move starts exactly at the neutral pose.
        self.offsets = np.asarray(rows, dtype=float) - np.asarray(rows[0], dtype=float)
        self.duration = float(self.times[-1])

    def at(self, t: float) -> Pose:
        """The offsets `t` seconds into the move; before it, its first frame (the neutral pose), after it, its last."""
        t = min(max(float(t), 0.0), self.duration)
        return Pose(*(float(np.interp(t, self.times, column)) for column in self.offsets.T))


@lru_cache(maxsize=None)
def load(name: str = THINKING_MOVE) -> RecordedMove:
    """A move kept in moves/, refused unless it is the file it was taken as."""
    raw = (MOVES_DIR / f"{name}.json").read_bytes()
    if name in SHA256 and hashlib.sha256(raw).hexdigest() != SHA256[name]:
        raise ValueError(f"move {name}: moves/{name}.json is not the file taken from the emotions library")
    return RecordedMove(json.loads(raw), name)
