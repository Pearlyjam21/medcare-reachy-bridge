"""MediaPipe Tasks landmarks, configured exactly like the browser worker (monitorWorker.ts)."""

from pathlib import Path

import numpy as np

from reachy_bridge.packets import build_packet


class VisionEngine:
    """Face (4) + hand (8) + pose-lite (4) landmarkers in VIDEO mode, CPU, same .task files as the frontend."""

    def __init__(self, models_dir: Path):
        import mediapipe as mp   # lazy: tests and the state machine never need it
        from mediapipe.tasks.python import BaseOptions, vision

        models_dir = Path(models_dir)
        mode = vision.RunningMode.VIDEO

        def options(name):
            path = models_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"MediaPipe model missing: {path}")
            return BaseOptions(model_asset_path=str(path))

        self._mp = mp
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=options("face_landmarker.task"), running_mode=mode, num_faces=4))
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=options("hand_landmarker.task"), running_mode=mode, num_hands=8))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=options("pose_landmarker_lite.task"), running_mode=mode, num_poses=4))
        self._last_ms = -1

    def process(self, frame_bgr: np.ndarray, frame_seq: int, timestamp_ms: float) -> dict:
        import cv2

        height, width = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode rejects non-increasing timestamps across the landmarker's lifetime.
        mp_ms = max(int(timestamp_ms), self._last_ms + 1)
        self._last_ms = mp_ms
        face = self.face.detect_for_video(image, mp_ms)
        hand = self.hand.detect_for_video(image, mp_ms)
        pose = self.pose.detect_for_video(image, mp_ms)
        return build_packet(face, hand, pose, frame_seq, timestamp_ms, width, height)

    def close(self) -> None:
        for landmarker in (self.face, self.hand, self.pose):
            landmarker.close()
