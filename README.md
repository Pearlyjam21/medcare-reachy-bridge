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
medicine with a short prerecorded prompt, and streams camera landmarks to the server, which decides what is recorded.
Anything it can't verify goes to a family member to confirm on LINE.

**This app needs a MedAiCarePlus server on your home network.** On its own it does nothing.

## Setup

1. In the MedAiCarePlus web app: **Settings → Reachy robot** → read and accept the robot notice → **Pair**. Copy the
   robot key (`rdv1.…`); it is shown only once.
2. On the server, let the robot reach the device port: set `DEVICE_BIND` in `.env` to the server's home-network IP
   and restart (`docker compose up -d app`). Never expose this port to the internet.
3. Install **MedCare Reachy** from the robot's dashboard and open its settings page. Enter the server address
   (e.g. `http://192.168.1.20:8001`) and the robot key, then **Save and connect**.

The first start downloads three MediaPipe models (about 17 MB) from Google's official storage and checks each
against a pinned SHA-256.

## What it does and doesn't do

- Reminds, finds the patient, prompts each medicine, and watches for hand-to-mouth movement. **It cannot identify
  which pill was taken or how many.**
- Never decides on its own that a dose was taken; the server records under its own policy.
- Fails safe: stops and puts the robot to sleep if the server is unreachable or the robot key is revoked.
- No microphone in this version.
- **Not a medical device.** Prototype software. Prompt wording is pending clinician review, and no voice clips are
  included yet, so the robot is silent until they're added.

## Status

- Runs end to end against the Reachy Mini simulator (`reachy-mini-daemon --mockup-sim`): it picks up a reminder,
  wakes, and searches for the patient. The simulator has no camera, so the vision path and frame rate still need a
  real robot.
- Tests: `pip install -e ".[test]"` then `python -m pytest tests medcare_reachy/bridge/tests -q`.

Source of the robot-side logic: https://github.com/Pearlyjam21/medcare-reachy-bridge
