---
title: MedCare Reachy
emoji: 💊
colorFrom: blue
colorTo: green
sdk: static
pinned: false
short_description: Medication-time companion that reminds and watches each dose
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# MedCare Reachy

A Reachy Mini app that turns the robot into a medication-time companion for a MedAiCarePlus home server. At dose
time the server sends the robot a reminder task. The robot wakes, looks for the patient, asks them to take each
medicine with a short prerecorded prompt, and watches with its own camera. **Intake tracking and emotion run on the
robot**: it sends the server landmark points and emotion scores, plus two snapshots a second that the server uses only
to confirm it's the enrolled patient (the face data stays on the server). The server decides what is recorded.
Anything it can't verify goes to a family member to confirm on LINE.

**This app needs a MedAiCarePlus server on your home network.** On its own it does nothing.

## Setup

1. In the MedAiCarePlus web app: **Settings → Reachy robot** → read and accept the robot notice → **Pair**. Copy the
   robot key (`rdv1.…`); it is shown only once.
2. On the server, let the robot reach the device port: set `DEVICE_BIND` in `.env` to the server's home-network IP
   and restart (`docker compose up -d app`). Never expose this port to the internet.
3. Install **MedCare Reachy** from the robot's dashboard and open its settings page. Enter the server address
   (e.g. `http://192.168.1.20:8001`) and the robot key, then **Save and connect**.

The vision and emotion models ship inside the app (about 55 MB) and are checked against pinned SHA-256s at every start; nothing
is downloaded at run time.

## What it does and doesn't do

- Reminds, finds the patient, prompts each medicine, and watches for hand-to-mouth movement. **It cannot identify
  which pill was taken or how many.**
- Never decides on its own that a dose was taken; the server records under its own policy.
- Fails safe: stops and puts the robot to sleep if the server is unreachable or the robot key is revoked.
- No microphone in this version.
- **Not a medical device.** Prototype software. Prompt wording is pending clinician review, and no voice clips are
  included yet, so the robot is silent until they're added.

## How vision runs on the robot

The Reachy Mini Wireless's Raspberry Pi 4 can't load the MediaPipe library (its ARM64 builds need AES instructions the
Pi lacks; 0.1.0 crashed with `SIGILL … compiled with aes enabled`). 0.2.0 runs the same MediaPipe Tasks models the web
app's browser uses (face, hand, pose-lite), converted to ONNX, on ONNX Runtime, which Reachy Mini's own software
already uses on the robot. The detection, cropping and tracking steps around the models are reimplemented in numpy.
No MediaPipe, OpenCV or AES instructions are needed.

Emotion uses the MedAiCarePlus server's own seven-emotion model, also on ONNX Runtime, at most twice a second, and
only on the face the server has confirmed is the patient, never while a hand covers the mouth. The server re-checks
both rules before it accepts a score. Its scores match the server's to within 0.02.

Checked against the real MediaPipe on its sample images: face points within 1–2 px, hand points within 1–6 px, pose
within 5–8 px. On a moving clip the error is under 1 px, with the same frame-to-frame jitter
(`tests/test_vision_accuracy.py`, `tools/MODELS.md`).

### Is the robot fast enough?

To keep up on a Pi, the detectors run every 3rd frame and pose every 2nd frame, and the three models run on separate
cores. The server only records automatically from sessions that keep **12 landmark frames per second**; below that,
every dose goes to a family member to confirm on LINE (nothing is recorded wrongly, it just isn't automatic). Measure it
on your robot over SSH:

```bash
/venvs/apps_venv/bin/python -m medcare_reachy.bench
```

While a reminder runs, the settings page also shows the live **Camera landmarks** rate.

## Status

- **0.3.0**: emotion is scored on the robot too; snapshots to the server are identity-only, 2 a second (was 5).
- **0.2.0**: vision runs on the robot (above). Tested against MediaPipe and the Reachy Mini simulator; frame rate on
  a real robot is not measured yet; run the bench above.
- Upgrading from 0.1.0: the old MediaPipe package is no longer used, so it can't crash the app, but you can remove it
  over SSH with `/venvs/apps_venv/bin/pip uninstall -y mediapipe`.
- Runs end to end against the Reachy Mini simulator (`reachy-mini-daemon --mockup-sim`): it picks up a reminder,
  wakes, and searches for the patient. The simulator has no camera.
- Tests: `pip install -e ".[test]"` then `python -m pytest tests medcare_reachy/bridge/tests -q`.

Source of the robot-side logic: https://github.com/Pearlyjam21/medcare-reachy-bridge
