# Vision models: where they come from

`medcare_reachy/vision_models/` holds seven ONNX files (and the emotion model's metadata), pinned by SHA-256 in
`medcare_reachy/models.py`.

| File | Source |
|---|---|
| `face_detector.onnx`, `face_landmarks_detector.onnx` | `face_landmarker.task` (float16/1), the web app's browser model |
| `hand_detector.onnx`, `hand_landmarks_detector.onnx` | `hand_landmarker.task` (float16/1) |
| `pose_landmarks_detector.onnx` | `pose_landmarker_lite.task` (float16/1) |
| `emotion_seed43.onnx` (+ `_metadata.json`) | the MedAiCarePlus server's `models/emotion_seed43/model_fp32.onnx`, unchanged (MobileNetV3-Large, 7 emotions, 112×112 grayscale input) |
| `pose_detector.onnx` | OpenCV Zoo `person_detection_mediapipe_2023mar.onnx` (HF `opencv/person_detection_mediapipe`) |

The `.task` files are zip archives of TFLite models, from
`https://storage.googleapis.com/mediapipe-models/<task>/<task>/float16/1/<task>.task`
(SHA-256 `64184e22…`, `fbc2a300…`, `59929e1d…`, the same files as `frontend_source/public/models`). Licence: Apache-2.0
(MediaPipe models and OpenCV Zoo).

## Conversion

Python 3.11, `tensorflow-cpu==2.15.1 tf2onnx==1.16.1 onnx==1.16.2 numpy<2 protobuf<4.25`:

```bash
unzip face_landmarker.task      # face_detector.tflite, face_landmarks_detector.tflite, ...
python -m tf2onnx.convert --tflite face_detector.tflite --output face_detector.onnx --opset 13
# same for face_landmarks_detector, hand_detector, hand_landmarks_detector, pose_landmarks_detector
```

`pose_detector.tflite` stores sparse weights that tf2onnx can't read (TF's interpreter segfaults listing its tensors),
so OpenCV Zoo's ONNX export of the same MediaPipe pose detector is used instead. On MediaPipe's `pose.jpg` it finds the
same person as the TFLite original: same box size, centre within a pixel. It only seeds the pose crop; the
landmarks come from the converted Tasks model.

## Accuracy check

`mp_reference.py` runs the real MediaPipe Tasks (x86, `mediapipe==1.0.1`) on MediaPipe's sample images
(`portrait.jpg`, `pose.jpg`, `thumb_up.jpg`, `pointing_up.jpg` from `storage.googleapis.com/mediapipe-assets/`) and
writes `reference.json`. `tests/test_vision_accuracy.py` compares this app's pipeline with it:

| Image | Points | Mean error |
|---|---|---|
| portrait.jpg | the server's 9 face points | 1.1 px |
| pointing_up.jpg | 21 hand points | 1.1 px |
| thumb_up.jpg | 21 hand points | 5.8 px |
| pose.jpg | nose, shoulders, wrists | 5.5 px |

On a 90-frame moving clip (VIDEO mode on both sides) the mouth and fingertip points differ by 0.6–0.8 px on average,
with jitter equal to MediaPipe's.
