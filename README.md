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
- Follows the server's overdose protection: when the server refuses a dose (not due yet, too soon after the last one,
  the day's maximum reached, or missed too long ago), the robot says the server's one-sentence reason once and moves
  on to the next medicine. It never retries that dose in the same reminder.
- Fails safe: stops and puts the robot to sleep if the server is unreachable or the robot key is revoked.
- Uses the microphone only while the patient's microphone consent is on: to hear 「我吃完了」 while a dose is
  watched, and to talk in check-ins.
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

## Emotion while a dose is watched (0.9.0)

From the prompt until the dose is resolved or left, the robot senses the patient's emotion from three things, the
way the [Reachy Mini Multimodal Emotion](https://huggingface.co/spaces/pearlyjam21/reachy_mini_multimodal_emotion)
app does: the **face** (the bundled `emotion_seed43` model, which is that app's FER+ MobileNetV3, on the face the
server verified, never with a hand at the mouth), the **voice** (its acoustic model, on each sentence the
"I finished" listener cuts; needs the microphone consent) and the **words** (its Traditional Chinese text model, on
the listener's transcript; left out for English). The three are fused with the weights from
[`pearlyjam21/multimodal-emotion-zh`](https://huggingface.co/pearlyjam21/multimodal-emotion-zh) (text 0.9, face 0.5,
voice 0.3); a modality without fresh evidence is left out, never counted as "neutral". The engine is in
`medcare_reachy/bridge/multimodal/` (`labels`, `fusion`, `speech` and `text` are the Space's own modules).

- The voice and text models (about 110 MB) are fetched from the Hub on first start with the robot's existing Hugging
  Face login (`hf auth login`, or `HF_TOKEN`), then reused from the cache. If they can't be had, emotion runs on the
  face alone and the log says so. `MULTIMODAL_MODELS_DIR` points at a local copy instead.
- What the server receives keeps its contract: with on-robot landmarks the score rides on the landmark packet as
  before; with streamed frames it goes along as an `emotion` form field of `/monitor/frame` (servers before the
  multimodal update ignore it). A dose watched in silence is reported exactly as 0.3.0 did (the face model's own
  softmax); only when a second modality is present are the fused probabilities sent.
- The dose's summary (average fused emotion, how many face, voice and text samples, which branches were on, where it
  was computed) is filed with the dose's confirmation evidence under `emotion`.

**Where it is computed.** By default on the robot (`emotion_compute: auto`). The governor
(`multimodal/governor.py`) hands over to the laptop when the robot scored fewer than half the frames due in the last
10 s, took over 0.5 s three times in a row, or failed three times in a row; the robot then only streams its camera
(even when it computes landmarks itself) and the server scores the frames with its own copy of the face model. After
60 s the robot is tried again. The settings page shows where emotion runs, the branches in use, the rate and the
reason for a fallback; "robot" and "laptop" pin the choice. Voice and words have no laptop path: a branch that fails
three times in a row is switched off for the session.

### Is the robot fast enough?

To keep up on a Pi, the detectors run every 3rd frame and pose every 2nd frame, and the three models run on separate
cores. The server only records automatically from sessions that keep **12 landmark frames per second**; below that,
every dose goes to a family member to confirm on LINE (nothing is recorded wrongly, it just isn't automatic). Measure it
on your robot over SSH:

```bash
/venvs/apps_venv/bin/python -m medcare_reachy.bench
```

While a reminder runs, the settings page also shows the live **Camera landmarks** rate.

## When the camera stops

From 0.7.0 the app watches its camera while it looks for the patient or watches a dose:

- **No new frame for 2 seconds** is a stall. The app logs an error with the robot daemon's media status and tries to
  get the camera back, at most once every 5 seconds: when the daemon says the camera was released, the app asks for it
  back; otherwise it reopens its own camera feed. A slow camera is not a stall; only a 2-second gap is. The stall is
  over only after 5 seconds of steady frames, so a camera that gives one frame and stops again is still in the same
  stall.
- **Still no steady frames 15 seconds after the stall began:** the app ends the search with the reason
  `camera_unavailable` and puts the robot to sleep. The server then handles the reminder as for a patient it could not
  find; the camera alone never records or skips a dose.
- **The speaker and microphone share the camera's link.** When the daemon refuses to give the camera back, they are
  off too. The app then asks for them back with the heartbeat and before each reminder or check-in, at most once a
  minute while the daemon keeps refusing, and when it gives up on the camera while they are still off, it logs an
  error saying so.
- The heartbeat tells the server the camera state, and the web app's Reachy card shows "Robot camera not working".

**Don't run TCU's Reachy AI Companion at the same time.** It connects to the robot without media
(`media_backend="no_media"`), and Reachy Mini's SDK 1.11.0 then makes the daemon release the camera and audio. This
app's camera feed stops. From 0.7.0 the app asks for it back when it notices, during a search or a dose watch (see
above), but the companion can take it again at any time. Disable the companion at boot over SSH
(`sudo systemctl disable --now reachy-ai-companion`); it also uses this app's port, 8042. To check the camera over
SSH: `curl -s localhost:8000/api/media/status` shows `"released": true` when another program took it, and
`curl -s -X POST localhost:8000/api/media/acquire` gives it back.

## Live voice check-ins (0.8.0, test)

From 0.8.0 a check-in can be a live voice chat with OpenAI's GPT-Live, **for testing only**: no patient notice covers
it yet. It runs only when the server allows it (`LIVE_VOICE_TEST=1` in the server's `.env`), the robot's "Live
voice (test)" switch is on in the web app's Reachy card, and a live model on the web app's AI connections card has
passed its Check. Otherwise check-ins work as before.

- The robot sends what its microphone hears to the MedAiCarePlus server as it hears it, and plays the voice the server
  sends back. The server talks with GPT-Live; the OpenAI key stays on the server.
- Reachy no longer waits for its own speech-to-text (about 9 seconds a turn): GPT-Live answers by itself.
- The server holds the start of every reply for half a second and can stop Reachy's voice at once. When the patient
  talks about harming themselves, Reachy says the help line in its own voice and family are alerted, as in the text
  chat.
- While the chat runs, the server checks every 10 seconds that the robot is still paired and that the patient's
  consents and live switch are still on. If not, the chat ends; when only the memory consent was switched off, the
  check-in goes on as the text chat, without the notes.
- When the server can't reach GPT-Live, or GPT-Live or the server fails during the chat, the check-in goes on as the
  text chat. When OpenAI ends the session itself (its moderation), the server first judges the patient's last words:
  the text chat takes over only when the safety check answers explicitly that there is no risk. Any other answer, or
  none, counts as a risk: Reachy says the help line, family are alerted and the chat ends. When the robot's own
  connection to the server drops, the server ends that chat and the robot starts a new text chat.
- One sound owner: the app's speech listener pauses during the live chat, and Reachy's own lines go through the same
  speaker queue as GPT-Live's voice.
- It uses the `websockets` package, which the robot's apps environment already has.

## Companion session (0.9.0)

From 0.9.0 a check-in is a companion session: Reachy stays with the patient until they say goodbye (拜拜, 再見 and
similar). The same rules hold for the live voice chat and for the text chat; only the "brain" differs. Update the
MedAiCarePlus server at the same time: a robot older than 0.9.0 keeps its old rules, and an older server does not know
the new ones.

- When Reachy finds the patient, it greets them in its own voice and listens. The camera stays on during the chat, at
  2 frames a second.
- Live voice: GPT-Live starts only when the patient speaks and the camera recognised them in the last 15 seconds.
  After 30 seconds without the patient speaking, the server pauses GPT-Live (no cost while paused) and Reachy keeps
  listening; the patient's next words start it again, with the conversation so far. The robot tells the server when
  the patient starts to speak, with the speech listener's voice detector model; it ignores Reachy's own sound until
  about 0.75 seconds after it. Without that model, or if the detector fails, check-ins use the text chat.
- Reachy rests after 10 minutes with no voice and nobody seen: it says 「我先休息一下，有需要再叫我喔。」 and goes to
  sleep. While the patient is in view, it stays awake.
- When a medicine falls due, Reachy says 「該吃藥囉，我們等一下再聊。」, ends the chat and starts the medicine reminder.
  It first lets the patient finish: words they have just said go to the server and get their answer, for at most
  30 seconds (`MEDICINE_WAIT_SECONDS`; in the live chat, until they stopped speaking 2 seconds before,
  `MEDICINE_SPEECH_GAP_SECONDS`). A dose the patient left pending before the chat counts as such a medicine when the
  server's reminder retry for it comes during the chat.
- Only a fresh camera answer counts as the patient in view: if the camera stops, Reachy rests after 10 minutes as if
  nobody were there. A live chat whose camera has not worked for 30 seconds (`CAMERA_LOST_SECONDS`) goes on as the
  text chat, because GPT-Live starts only for a patient the camera recognises.
- When GPT-Live cannot start for words the patient just said, Reachy asks 「不好意思，我剛剛沒聽清楚，可以再說一次嗎？」 and the
  text chat listens.
- The chat also ends on a risk (Reachy says the help line and family are alerted), a stop from the web app, or a
  consent withdrawn (the text chat too, at once, when the microphone consent is withdrawn). Once the server has ended
  a live chat, the robot never cuts its last line short for a medicine or a rest. There is no time limit any more.

## Status

- **0.10.0**: merges the Space's 0.9.0 companion session (GPT-Live check-ins that stay until goodbye) with 0.9.1
  (multimodal emotion while a dose is watched, face following, the microphone start for SDK 1.8.3).
- **0.9.0** (for testing; not on this space yet): the companion session (above). A check-in stays until a goodbye,
  rests after 10 minutes with no voice and nobody seen, and stops for a medicine that falls due.
- **0.9.1**: the robot app starts the microphone itself. Reachy Mini SDK 1.8.3 (the robot after a reset) never starts it, so every read was empty and neither 「我吃完了」 nor a check-in was heard.
- **0.9.0**: multimodal emotion (face + voice + words) while a dose is watched, computed on the robot with a laptop
  fallback (above); the head follows the patient's face while a dose is watched (*Follow the patient's face*,
  `bridge/follow.py`: at most every 0.4 s, only when the face moved, still while a hand-to-mouth event is checked).
  New settings: *Sense emotion…*, *Where emotion is computed*, *Follow the patient's face*. `sherpa-onnx` is now a
  declared dependency (a fresh robot image lacks it, and without it the robot never hears 「我吃完了」).
- **0.8.0** (for testing; not on this space yet): live voice check-ins (above). Without the server's live voice
  switch, check-ins are unchanged.
- **0.7.1**: today's check-in chat no longer ends when the patient is quiet (`CHECKIN_SILENCE = None`); it ends on a
  goodbye, a risk, a stop, or after 5 minutes. Runs on the robot since 12 Oct 2026; not on this space yet.
- **0.7.0**: a camera watchdog (above): a stalled camera is recovered, or the search ends with `camera_unavailable`
  about 17 seconds after the last frame instead of searching for 10 minutes. The heartbeat reports the camera state.
  Two quick restarts from the settings page no longer start two copies of the app's worker.
- 0.7.0 is the version on this space since 12 Oct 2026. The space's history also holds an ElevenLabs voice line
  (0.6.0, 4-5 Oct) that is not in 0.7.0; it is kept for a later version.
- **0.3.0**: emotion is scored on the robot too; snapshots to the server are identity-only, 2 a second (was 5).
- **0.2.0**: vision runs on the robot (above). Tested against MediaPipe and the Reachy Mini simulator; frame rate on
  a real robot is not measured yet; run the bench above.
- Upgrading from 0.1.0: the old MediaPipe package is no longer used, so it can't crash the app, but you can remove it
  over SSH with `/venvs/apps_venv/bin/pip uninstall -y mediapipe`.
- Runs end to end against the Reachy Mini simulator (`reachy-mini-daemon --mockup-sim`): it picks up a reminder,
  wakes, and searches for the patient. The simulator has no camera.
- Tests: `pip install -e ".[test]"` then `python -m pytest tests medcare_reachy/bridge/tests -q`.

Source of the robot-side logic: https://github.com/Pearlyjam21/medcare-reachy-bridge
