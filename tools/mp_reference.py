"""Reference landmarks from the real MediaPipe Tasks (x86), to check the ONNX port against.

Runs face/hand/pose landmarkers in IMAGE mode (no tracking) on the sample images and writes
reference.json: per image, every face/hand/pose as a list of [x, y] (normalized) plus pose visibility.
Usage: .venv-reachy python mp_reference.py
"""

import json
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

HERE = Path(__file__).parent
MODELS = Path(r"D:\medcareai2_20260920\spike\tmp")
IMAGES = sorted((HERE / "img").glob("*.jpg"))


def make():
    def opt(name):
        return BaseOptions(model_asset_path=str(MODELS / f"dl_{name}.task"))
    mode = vision.RunningMode.IMAGE
    return (vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=opt("face_landmarker"), running_mode=mode, num_faces=4)),
            vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=opt("hand_landmarker"), running_mode=mode, num_hands=8)),
            vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=opt("pose_landmarker_lite"), running_mode=mode, num_poses=4)))


def main():
    face, hand, pose = make()
    out = {}
    for path in IMAGES:
        image = mp.Image.create_from_file(str(path))
        f = face.detect(image)
        h = hand.detect(image)
        p = pose.detect(image)
        out[path.name] = {
            "size": [image.width, image.height],
            "faces": [[[lm.x, lm.y] for lm in face_lms] for face_lms in f.face_landmarks],
            "hands": [[[lm.x, lm.y] for lm in hand_lms] for hand_lms in h.hand_landmarks],
            "poses": [[[lm.x, lm.y, lm.visibility] for lm in pose_lms] for pose_lms in p.pose_landmarks],
        }
        print(path.name, image.width, image.height, len(f.face_landmarks), len(h.hand_landmarks), len(p.pose_landmarks))
    (HERE / "reference.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
