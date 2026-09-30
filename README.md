# MedCare Reachy Bridge

Robot-side SDK that connects a **Reachy Mini Wireless** to the MedAiCarePlus medication-intake app. At dose
time it wakes the robot, finds the enrolled patient, prompts each medication with prerecorded clips, and streams
camera landmarks to the app, which decides what is recorded.

Also published as a Hugging Face Space: https://huggingface.co/spaces/pearlyjam21/medcare-reachy-bridge

## What it does

- Pulls camera frames from the robot (`reachy_mini` SDK, WebRTC backend), centre-crops 16:9 → 4:3 at 640×480.
- Runs MediaPipe face / hand / pose landmarks (VIDEO mode) and sends packets in the same shape as the web app's
  browser worker, ~15 fps, with JPEG frames at ≤ 5 fps for identity and expression checks.
- Drives a pure, fully unit-tested state machine (`session.py`): wake → announce → search for the patient →
  prompt and watch each dose → post-slot observation → wind down.
- Talks only to the app's private device API (`/api/device/*`). **It never decides that a dose was taken.** The
  server records under its own policy; anything it cannot verify goes to a family member for confirmation.
- Fails closed: if the app is unreachable for 10 s or the device key stops working, it stops and puts the robot
  to sleep.
- **No microphone** is opened by this version.

## Status and limits

- **Not a medical device.** Research and prototype code. It observes hand-to-mouth movement and **cannot
  identify which pill was taken or how many.**
- The robot SDK calls and the Linux GStreamer/WebRTC setup in `Dockerfile.bridge` have **not yet been tested
  against a real robot**.
- The prompt wording in `reachy_bridge/clips/manifest.json` (Traditional Chinese and English) is marked
  **pending clinician review**. No audio files are included.
- It needs the companion MedAiCarePlus server (not published here) to do anything useful.

## Run the tests

```bash
pip install httpx numpy opencv-python-headless pytest
python -m pytest reachy_bridge/tests -q
```

No robot, network, `mediapipe` or `reachy_mini` needed. One index-parity test is skipped outside the full app
repository.

See `reachy_bridge/README.md` for configuration (`APP_INTERNAL_URL`, `REACHY_DEVICE_TOKEN`, `REACHY_ROBOT_HOST`,
`ROBOT_BACKEND=video` for hardware-free replay) and the module layout.
