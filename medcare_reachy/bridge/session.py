"""Pure slot state machine (spec 01 §7).

`SlotSession.tick()` is called a few times per second by the runner. It never blocks on
hardware directly: robot and clip calls go through `run_blocking` (inline in tests,
`asyncio.to_thread` in the runner). The frame pipeline (`stream`) posts frames to whatever
monitor session is attached and exposes the latest `public()` state it got back.

While a dose is watched, and only while the task says the patient's microphone consent is
current, `voice` listens for the patient saying they have finished. Saying so is a claim, not
evidence: if the camera does not resolve the dose within DONE_GRACE, it goes to caregiver
confirmation as `patient_claim`.

While a dose is watched, `mood` (multimodal emotion, optional) is told when the watching starts and ends and
what it saw and heard goes into the dose's confirmation evidence ("emotion").

In a check-in conversation Reachy says 「嗯」 the moment the patient pauses (`ack`) and moves
while it prepares and speaks its reply (`gestures`); either can be turned off in the settings.

Every observed result is advisory: the server decides whether anything is recorded. When the
patient's overdose protection refuses a dose (not due yet, too soon after the last one, the
day's maximum reached, or missed too long ago), Reachy says the server's one-sentence reason
once and moves on to the next dose. It files nothing more: after "I finished" or a camera
event, the server itself alerts family to a possible double dose.

When the task says `live` (spec 2026-10-12 F2), the check-in is a live chat instead (`bridge/live.py`):
the robot streams its microphone to the server's GPT-Live relay and plays the voice that comes back.
The speech listener is suspended meanwhile (the microphone has one reader), and when the server
falls back, today's chat goes on.

From 0.9.0 a check-in is a companion session, whichever chat runs (spec 2026-10-12 companion, section 4): the
camera stays on (the observe session, at COMPANION_CAPTURE_FPS), there is no time limit, and it ends on a goodbye, a
risk, a stop (also a withdrawn microphone consent), a medicine task falling due (medicine_due: MEDICINE_LINE, once the
patient's words have gone to the server, MEDICINE_WAIT_SECONDS at most) or the rest rule (no patient voice and nobody
seen for REST_AFTER_SECONDS: REST_LINE). A live companion chat whose camera stops working for CAMERA_LOST_SECONDS goes
on as today's chat.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from medcare_reachy.bridge.app_client import (
    AppUnreachable, BridgeError, BusyOtherClient, DoseRefused, NotAuthorised, RequestRejected, ServiceUnavailable,
    SessionLost)

log = logging.getLogger(__name__)

RESOLVED = frozenset({"taken", "pending_confirmation", "skipped"})
SEARCH_TIMEOUT = 600.0          # 10 min to find the patient (also: patient gone before a later prompt)
WATCH_TIMEOUT = 180.0           # 3 min without an event -> help clip
HELP_TIMEOUT = 180.0            # 3 more min -> leave the dose pending
DONE_GRACE = 8.0                # after "I finished": time for the camera to resolve the dose first
MED_CLIP_FETCH_SECONDS = 20.0   # the server renders a medicine's spoken prompt in a few seconds; never wait longer
AWAY_TIMEOUT = 90.0             # patient out of sight this long while watched -> leave the dose pending
CHECKIN_SILENCE: float | None = None   # check-in: off (0.7.1): a companion stays while the patient is quiet;
                                       # seconds here end the chat that long after Reachy last spoke, unanswered
REST_AFTER_SECONDS = 600.0      # companion: no patient voice and nobody seen this long -> REST_LINE, then sleep
COMPANION_CAPTURE_FPS = 2.0     # companion: the camera stays on at this rate (it only tells who is there)
CAMERA_RETRY_SECONDS = 5.0      # companion: a camera session that did not open is tried again this often
COMPANION_STOP_SECONDS = 45.0   # live companion: time for Reachy's closing line and the chat's end (its drain: 30 s)
COMPANION_ENDINGS = ("patient_left", "medicine")   # Reachy said why it stops: it sleeps at once, no wind_down clip
LIVE_POLL_SECONDS = 0.2         # live check-in: how often the session looks for a reason to stop it at once
LIVE_STOP_SECONDS = 5.0         # live check-in: how long a stopped live chat may take to close
DETECTOR_FAILED = "detector_failed"   # live companion chat: its voice detector stopped; today's chat goes on
CAMERA_FAILED = "camera_failed"       # live companion chat: its camera stopped working; today's chat goes on
CAMERA_LOST_SECONDS = 30.0      # live companion: no camera session, or its camera not "ok", this long -> CAMERA_FAILED
MEDICINE_WAIT_SECONDS = 30.0    # companion: the medicine ending waits for the patient's words at most this long
MEDICINE_SPEECH_GAP_SECONDS = 2.0   # live companion: ... while the patient speaks or stopped less than this long ago
LIVE_AUDIO = ("start_live_audio", "push_audio", "fill_live_audio", "clear_live_audio", "stop_live_audio")   # media.py
SPEECH_MODEL = "silero_vad.onnx"   # the listener's voice detector (voice.py STT_FILES); live.SpeechStart uses it too
# Spoken (not prerecorded) lines, already in the voice's Simplified characters (喔 written 哦, as in the clips).
CHECKIN_GREETING = {"zh-TW": "您好！我来找您聊聊天。", "en": "Hello! I came to have a chat with you."}
REST_LINE = {"zh-TW": "我先休息一下，有需要再叫我哦。", "en": "I'll rest for now. Call me when you need me."}
MEDICINE_LINE = {"zh-TW": "该吃药啰，我们等一下再聊。", "en": "It's time for your medicine. Let's talk again later."}
ACK_CLIP = "ack"                # check-in: the prerecorded 「嗯」 said the moment the patient pauses
POST_SLOT_SECONDS = 120.0
WRONG_PERSON_SECONDS = 20.0
UNREACHABLE_LIMIT = 10.0        # fail closed after the app has been unreachable this long
BUSY_RETRY_SECONDS = 30.0
SCAN_INTERVAL = 4.0
SEARCH_CLIP_INTERVAL = 120.0
MAX_RECOVERIES = 10
OPEN_ORDER = ("leased", "searching", "in_progress")

WAKE, ANNOUNCE, SEARCHING = "WAKE", "ANNOUNCE", "SEARCHING"
MED_PROMPT, WATCHING, POST_SLOT_OBSERVE = "MED_PROMPT", "WATCHING", "POST_SLOT_OBSERVE"
ALREADY_TAKEN, WIND_DOWN, WAITING_OTHER_CLIENT, SLEEP = "ALREADY_TAKEN", "WIND_DOWN", "WAITING_OTHER_CLIENT", "SLEEP"
CHECKIN = "CHECKIN"


class TaskGone(Exception):
    """The leased task is no longer this device's current task (expired, aborted server-side)."""


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def wall(self) -> datetime:
        return datetime.now(timezone.utc)


async def _inline(fn, *args):
    return fn(*args)


def _ms(seconds: float) -> int:
    return max(0, int(round(seconds * 1000)))


def _parse_time(value) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def speech_models_dir(voice) -> Path | None:
    """The speech listener's models folder (voice.DoneListener.models_dir) when its voice detector, SPEECH_MODEL, is
    there, else None. A live companion chat hears the patient begin to speak with it (live.SpeechStart)."""
    models_dir = getattr(voice, "models_dir", None)
    if models_dir is None or not (Path(models_dir) / SPEECH_MODEL).is_file():
        return None
    return Path(models_dir)


def open_live_chat(app, robot, speaker, models_dir: Path | None):
    """A LiveChat (bridge/live.py) when this robot app can hold one, else None (today's chat runs instead): the robot
    has the live audio calls (media.py's ReachyRobot; not the video-file robot), the live module loads (it needs
    `websockets`, which the robot's apps_venv has), and the voice detector loads from `models_dir`
    (speech_models_dir): the server starts GPT-Live only when the robot tells it the patient began to speak."""
    if models_dir is None or not all(callable(getattr(robot, name, None)) for name in LIVE_AUDIO):
        return None
    try:
        from medcare_reachy.bridge.live import LiveChat, SpeechStart
    except ImportError as exc:
        log.warning("live voice is off in this robot app: %s", exc)
        return None
    try:
        speech_start = SpeechStart(Path(models_dir))
    except Exception:
        log.exception("live voice is off: the voice detector did not load from %s", models_dir)
        return None
    hardware = None
    if all(callable(getattr(robot, name, None)) for name in ("goto", "get_frame", "hold_head")):
        from medcare_reachy.bridge.reachy_hardware import ReachyMiniHardware
        hardware = ReachyMiniHardware(robot)   # the GPT-Live agent's camera and emote requests (live.py)
    return LiveChat(app, robot, speaker, speech_start=speech_start, hardware=hardware)


class SlotSession:
    def __init__(self, task: dict, *, app, robot, clips, stream, voice=None, speaker=None, language="zh-TW",
                 clock=None, run_blocking=_inline, ack=True, gestures=True, mood=None, follow=None,
                 live_chat=open_live_chat):
        self.task = task
        self.mood = mood   # multimodal emotion for the dose being watched (begin/end/summary), or None
        self.follow = follow   # follow.FaceFollower: the head keeps the verified face centred while a dose is watched
        self.task_id = task["task_id"]
        self.status = task.get("status") or "leased"
        self.doses = [dict(dose) for dose in task.get("doses", [])]
        self.app, self.robot, self.clips, self.stream, self.voice = app, robot, clips, stream, voice
        self.speaker, self.language = speaker, language
        self.ack, self.gestures = ack, gestures
        self.live_chat = live_chat   # (app, robot, speaker, models_dir) -> a LiveChat, or None (open_live_chat)
        self.chat_id: str | None = None
        self.last_spoke = 0.0
        self.acked: float | None = None          # the patient's pause Reachy last said 「嗯」 to
        self.gesture_mode: str | None = None     # what Reachy's body is doing (robot.gesture)
        self.clock = clock or SystemClock()
        self.run_blocking = run_blocking

        self.state = WAKE
        self.since = self.clock.monotonic()
        self.entered = False
        self.done = False
        self.result: str | None = None
        self.outcomes: dict[int, str] = {}
        self.index = 0
        self.monitor: dict | None = None
        self.monitor_key: tuple | None = None
        self.monitor_companion = False   # whether the open camera session was opened as a companion session
        self.sent_extra: set[str] = set()
        self.chat_end: tuple[str, int] | None = None   # at the chat's end: (camera session id, its last frame_seq)
        self.stop_reason: str | None = None
        self.stop_abort = True
        self.needs_recovery = False
        self.recoveries = 0
        self.resume_state: str | None = None
        self.busy_announced = False
        self.help_at: float | None = None
        self.wrong_since: float | None = None
        self.wrong_announced = False
        self.scan_step = 0
        self.next_scan_at = 0.0
        self.next_search_clip_at = 0.0
        self.done_at: float | None = None
        self.done_word: str | None = None
        self.away_since: float | None = None
        self.seen_at: float | None = None         # when the patient was last seen verified, in any session
        self.explained: set[int] = set()          # doses whose refusal Reachy has already (tried to) explain
        self.companion_since: float | None = None   # when the companion session (CHECKIN) began
        self.heard_at: float | None = None          # when the patient's voice was last heard in it
        self.medicine_wanted = False                # medicine_due(): it ends for a medicine task
        self.medicine_at: float | None = None       # ... since then (MEDICINE_WAIT_SECONDS)
        self.camera_lost_since: float | None = None   # the live chat's camera unusable since then (_camera_lost)
        self.camera_retry_at = 0.0                  # when its camera may be tried again (_watch)
        self.handlers = {
            WAKE: self._wake, ANNOUNCE: self._announce, SEARCHING: self._searching,
            MED_PROMPT: self._med_prompt, WATCHING: self._watching, POST_SLOT_OBSERVE: self._post_slot,
            ALREADY_TAKEN: self._already_taken, WIND_DOWN: self._wind_down, CHECKIN: self._checkin,
            WAITING_OTHER_CLIENT: self._waiting_other_client,
        }

    # ── driver ───────────────────────────────────────────────────────────
    def stop(self, reason: str, abort: bool = True) -> None:
        """Finish at the next tick. `abort=False` (process shutdown) leaves the task leased for recovery."""
        if not self.stop_reason:
            self.stop_reason, self.stop_abort = reason, abort

    def medicine_due(self) -> None:
        """A heartbeat says a medicine task is due (spec 2026-10-12 companion 4.5). A companion session (CHECKIN) ends
        for it: once Reachy's current line is over and the patient's words have gone to the server (at most
        MEDICINE_WAIT_SECONDS from now) it says MEDICINE_LINE and ends as "medicine", and the runner then leases the
        medicine task. In any other state nothing changes (a dose task finishes first; every heartbeat says it
        again)."""
        if self.state == CHECKIN and not self.done and not self.medicine_wanted:
            log.info("slot %s: a medicine task is due; the companion session ends for it", self.task_id)
            self.medicine_wanted = True
            self.medicine_at = self.clock.monotonic()

    async def tick(self) -> None:
        if self.done:
            return
        if self.stop_reason:
            await self._shutdown(self.stop_reason, status="aborted" if self.stop_abort else None)
            return
        since = self.app.unreachable_since
        if since is not None and self.clock.monotonic() - since >= UNREACHABLE_LIMIT:
            await self._fail_closed("app_unreachable")
            return
        try:
            error = self.stream.take_error()
            if error is not None:
                raise error
            if self.needs_recovery:
                await self._recover()
            await self._forward_extra_events()
            await self.handlers[self.state]()
        except AppUnreachable as exc:
            log.warning("app unreachable in %s: %s", self.state, exc)   # retried; fail closed after 10 s
        except NotAuthorised as exc:
            log.error("device no longer authorised (%s); stopping", exc.detail)
            await self._fail_closed("not_authorised")
        except BusyOtherClient:
            self._enter_waiting_other_client()
        except DoseRefused as exc:
            await self._stream_refused(exc)
        except SessionLost as exc:
            self.recoveries += 1
            log.warning("monitor session lost in %s (%s)", self.state, exc.detail)
            if self.recoveries > MAX_RECOVERIES:
                await self._shutdown("session_lost", status="aborted")
            else:
                self._drop_session()
                self.needs_recovery = True
        except ServiceUnavailable:
            await self._shutdown("model_not_ready", status="aborted")
        except TaskGone:
            await self._shutdown("task_gone", status=None)
        except RequestRejected as exc:
            if exc.status == 404:   # the task is no longer leased to this device
                await self._shutdown("task_gone", status=None)
            else:
                log.error("request rejected in %s: %s", self.state, exc)
                await self._shutdown("bridge_error", status="aborted")
        except Exception:
            log.exception("slot session failed in %s", self.state)
            await self._shutdown("bridge_error", status="aborted")
        self._sync_voice()

    # ── states ───────────────────────────────────────────────────────────
    async def _wake(self):
        if not await self._robot("is_reachable"):
            await self._shutdown("robot_offline", status="aborted")
            return
        # A camera recovery the daemon refused may have left the speaker and the microphone off: the greeting needs
        # them (ReachyRobot.restore_media, which never raises).
        await self._robot("restore_media")
        await self._robot("wake")
        await self._advance("searching")
        self._go(ANNOUNCE)

    async def _announce(self):
        if self._is_checkin_task():
            await self._say(CHECKIN_GREETING.get(self.language, CHECKIN_GREETING["zh-TW"]))
        else:
            await self._play("reminder" if self.task.get("reason") == "missed_retry" else "wake_greeting")
        self._go(SEARCHING)

    async def _searching(self):
        await self._ensure_session("observe")
        now = self.clock.monotonic()
        if not self.entered:
            self.entered = True
            self.next_scan_at = now
            self.next_search_clip_at = now + SEARCH_CLIP_INTERVAL
        if self.stream.camera.state == "unavailable":
            # The camera watchdog gave up (no new frame for 15 s after the camera stalled, recoveries included): end
            # the search now, as for any not_found, rather than look around blind for 10 minutes.
            log.warning("slot %s: the camera is unavailable; the search ends", self.task_id)
            await self._shutdown("camera_unavailable", status="not_found")
            return
        if now - self.since >= SEARCH_TIMEOUT:
            await self._shutdown("not_found", status="not_found")
            return
        identity = self._latest().get("identity_status")
        if identity == "verified":
            await self._robot("hold_head")
            if self._is_checkin_task():
                await self._advance("in_progress")
                self._go(CHECKIN)
                return
            # Restart recovery and "taken earlier in the app": re-read before any prompt.
            await self._refresh_doses()
            if not any(self._unresolved(dose) for dose in self.doses):
                self._go(ALREADY_TAKEN)
                return
            await self._advance("in_progress")
            self.index = 0
            self._go(MED_PROMPT)
            return
        await self._check_wrong_person(identity)
        if now >= self.next_scan_at:
            await self._robot("look_around", self.scan_step)
            self.scan_step += 1
            self.next_scan_at = now + SCAN_INTERVAL
        if now >= self.next_search_clip_at:
            self.next_search_clip_at = now + SEARCH_CLIP_INTERVAL
            await self._play("searching")

    async def _med_prompt(self):
        if not self.entered:
            # Re-read right before prompting: never prompt a dose that became taken,
            # pending_confirmation or skipped (restart recovery, app or caregiver action).
            await self._refresh_doses()
            while self.index < len(self.doses) and not self._unresolved(self.doses[self.index]):
                self.outcomes[self.doses[self.index]["intk_id"]] = "already_resolved"
                self.index += 1
            if self.index >= len(self.doses):
                self._go(POST_SLOT_OBSERVE)
                return
            self.entered = True
        dose = self.doses[self.index]
        try:
            await self._ensure_session("dose", dose["intk_id"])
        except BusyOtherClient:
            raise
        except DoseRefused as exc:
            await self._refused(dose, exc)   # said instead of the prompt
            return
        except SessionLost:
            # The server refused this dose (resolved elsewhere, no pills left, deactivated).
            await self._refresh_doses()
            self.outcomes[dose["intk_id"]] = "unavailable" if self._unresolved(dose) else "already_resolved"
            self._next_dose()
            return
        identity = self._latest().get("identity_status")
        if identity != "verified":
            # Never prompt anyone the server has not verified in this session.
            await self._check_wrong_person(identity)
            if self.clock.monotonic() - self.since >= SEARCH_TIMEOUT:
                for later in self.doses[self.index:]:
                    self.outcomes.setdefault(later["intk_id"], "not_verified")
                self._go(WIND_DOWN)
            return
        await self._robot("hold_head")
        position = self.index + 1
        await self._fetch_med_clip(dose, position)
        await self._speak(self.clips.play_med_prompt, dose.get("med_id"), position)
        if self._may_listen():
            await self._play("say_when_done")
        self._go(WATCHING)

    async def _fetch_med_clip(self, dose: dict, position: int) -> None:
        """Reachy names the medicine it asks for: the server's spoken prompt for this medicine and its position in
        the slot (「請現在吃第一種藥，Adapalene。」, services/med_clips.py), fetched once and kept beside the other
        clips; a kept clip is checked against the server's key, so a renamed medicine gets a new one. Best effort:
        without a clip (no server voice, a failure, an older server) the generic prompt plays, never a guessed
        name. At most MED_CLIP_FETCH_SECONDS, so the prompt is never long delayed."""
        med_id = dose.get("med_id")
        if med_id is None or not callable(getattr(self.app, "med_clip", None)):
            return
        try:
            key = self.clips.med_clip_key(med_id, position)
            found = await asyncio.wait_for(self.app.med_clip(med_id, position, self.language, key),
                                           MED_CLIP_FETCH_SECONDS)
        except Exception as exc:   # asyncio.TimeoutError included: the prompt goes on without the name
            log.warning("slot %s: no spoken prompt for medicine %s (%r)", self.task_id, med_id, exc)
            return
        if found is None or found == "unchanged":
            return
        data, new_key = found
        try:
            await self.run_blocking(self.clips.save_med_clip, med_id, position, data, new_key)
            log.info("slot %s: spoken prompt for medicine %s (position %d) kept", self.task_id, med_id, position)
        except Exception:
            log.exception("slot %s: the spoken prompt for medicine %s could not be kept", self.task_id, med_id)

    async def _file_event(self, dose: dict, outcome: str, latest: dict) -> None:
        """A watched dose ended without a result: family get the camera's clip of it and the AI's reading from the
        server (POST /tasks/{id}/event, services/dose_event.py). Best effort, never raises."""
        if not callable(getattr(self.app, "task_event", None)):
            return
        evidence = {"said_done": self.done_at is not None, "phrase": self.done_word,
                    "degraded": bool(latest.get("degraded")), "landmark_fps": latest.get("landmark_fps"),
                    "seconds_watched": round(self.clock.monotonic() - self.since, 1), **self._mood_evidence()}
        try:
            await self.app.task_event(self.task_id, dose["intk_id"], outcome, evidence)
            log.info("slot %s: dose %s %s: family get the clip", self.task_id, dose["intk_id"], outcome)
        except Exception as exc:
            log.warning("slot %s: the %s event of dose %s was not filed (%r)", self.task_id, outcome,
                        dose["intk_id"], exc)

    async def _watching(self):
        dose = self.doses[self.index]
        try:
            await self._ensure_session("dose", dose["intk_id"])   # re-created after a lost session
        except DoseRefused as exc:
            # e.g. the patient took this medicine in the app meanwhile; retrying would only be refused again.
            await self._refused(dose, exc)
            return
        latest = self._latest()
        now = self.clock.monotonic()
        self._note_done(now)
        await self._follow_face(latest, now)
        recorded = latest.get("recorded")
        if recorded and recorded.get("status", "taken") == "taken":
            self.outcomes[dose["intk_id"]] = "recorded"
            await self._play("thanks")
            self._next_dose()
            return
        candidate = latest.get("candidate")
        if candidate and candidate.get("ready"):
            await self._needs_confirm(dose, candidate, latest)
            return
        await self._check_wrong_person(latest.get("identity_status"))
        if candidate:
            return   # an event is still being verified; let the server finish it
        if self.done_at is not None:
            if now - self.done_at >= DONE_GRACE:
                await self._patient_claim(dose, latest)
            return   # the patient says they finished: give the camera a moment to resolve the dose first
        if self._walked_away(latest.get("identity_status"), now):
            self.outcomes[dose["intk_id"]] = "left_pending"
            await self._file_event(dose, "patient_left", latest)
            self._next_dose()
            return
        if self.help_at is None and now - self.since >= WATCH_TIMEOUT:
            self.help_at = now
            await self._play("help")
        elif self.help_at is not None and now - self.help_at >= HELP_TIMEOUT:
            self.outcomes[dose["intk_id"]] = "left_pending"
            await self._file_event(dose, "not_seen", latest)
            self._next_dose()

    async def _needs_confirm(self, dose: dict, candidate: dict, latest: dict):
        source = self._confirm_source(dose, latest)
        evidence = {"event_id": candidate.get("event_id"), "decision": candidate.get("decision"),
                    "confidence": candidate.get("confidence"), "frame_seq": candidate.get("frame_seq"),
                    "degraded": bool(latest.get("degraded")), "landmark_fps": latest.get("landmark_fps"),
                    "said_done": self.done_at is not None, **self._mood_evidence()}
        await self._confirm(dose, source, evidence)

    async def _patient_claim(self, dose: dict, latest: dict):
        """Heard "I finished" but the camera resolved nothing: a caregiver checks the pill box."""
        evidence = {"said_done": True, "phrase": self.done_word, "camera": "no_event",
                    "degraded": bool(latest.get("degraded")), "landmark_fps": latest.get("landmark_fps"),
                    **self._mood_evidence()}
        await self._confirm(dose, "patient_claim", evidence)

    async def _confirm(self, dose: dict, source: str, evidence: dict):
        try:
            await self.app.confirmation(self.task_id, dose["intk_id"], source, evidence)
        except DoseRefused as exc:
            # Family is never asked to confirm a refused dose. When the patient said they finished or the camera
            # saw a hand-to-mouth event (dose_too_soon, daily_max_reached), the server alerts family to a possible
            # double dose itself: nothing more to file.
            await self._refused(dose, exc)
            return
        except SessionLost:
            # A retried POST that already succeeded, or the dose was resolved elsewhere.
            await self._refresh_doses()
            if dose.get("intake_stats") != "pending_confirmation":
                self.outcomes[dose["intk_id"]] = ("already_resolved" if not self._unresolved(dose)
                                                  else "confirmation_rejected")
                self._next_dose()
                return
        self.outcomes[dose["intk_id"]] = "needs_confirm"
        await self._play("confirm_with_caregiver")
        self._next_dose()

    # ── overdose protection ──────────────────────────────────────────────
    async def _refused(self, dose: dict, exc: DoseRefused) -> None:
        """The server refused this dose: say why, then go on as after any dose. The dose stays as the server has it
        (pending, or missed for good); its outcome is the refusal code. Never retried in this task."""
        log.info("slot %s: dose %s refused: %s", self.task_id, dose["intk_id"], exc.detail)
        self.outcomes[dose["intk_id"]] = exc.detail
        await self._explain(dose["intk_id"], exc.speech_text)
        self._next_dose()

    async def _stream_refused(self, exc: DoseRefused) -> None:
        """A refusal from the frame stream: with on-robot vision the server commits a dose while handling an upload.
        It concerns the attached session's dose; one Reachy has already moved on from is left as it was."""
        intk_id = exc.intk_id if exc.intk_id is not None else (self.monitor_key or (None, None))[1]
        current = (self.doses[self.index] if self.state in (MED_PROMPT, WATCHING) and self.index < len(self.doses)
                   else None)
        if current is None or current["intk_id"] != intk_id:
            log.warning("slot %s: dose %s refused (%s) after Reachy moved on", self.task_id, intk_id, exc.detail)
            return
        await self._refused(current, exc)

    async def _explain(self, intk_id: int, text: str | None) -> None:
        """Say the server's sentence once per dose in a task, and only to a patient who is there: seen verified within
        AWAY_TIMEOUT, or who has just said they finished. Best effort: no voice, no sentence or a failed synthesis
        leaves Reachy silent, never stops the slot."""
        if intk_id in self.explained or not text or self.speaker is None or not self.speaker.available:
            return
        self.explained.add(intk_id)
        now = self.clock.monotonic()
        if self.done_at is None and (self.seen_at is None or now - self.seen_at >= AWAY_TIMEOUT):
            log.info("slot %s: the patient isn't here to hear why dose %s was refused", self.task_id, intk_id)
            return
        try:
            await self._say(text)
        except Exception:
            log.exception("slot %s: could not say why dose %s was refused", self.task_id, intk_id)

    async def _checkin(self):
        """The companion session (spec 2026-10-12 companion 4.1), one controller for both chats. On entering it the
        camera stays on (`_watch`: the observe session, at COMPANION_CAPTURE_FPS), then the live chat runs
        (`_live_checkin`; Reachy says the server's opening line in its own voice) or today's chat: the patient's
        words become text on the robot; the server answers.

        Each turn carries how long hearing it took, and each of Reachy's lines is followed by how long it took to
        arrive and to speak (milliseconds, for the patient's conversation history).

        There is no time limit. Today's chat ends on a goodbye or a risk (the server's answer says so), a stop, the
        patient's microphone consent withdrawn (the heartbeat: at once, reason "stopped", outcome
        microphone_withdrawn), a medicine task falling due (medicine_due: Reachy says MEDICINE_LINE, reason
        "medicine") or the rest rule (`_resting`: REST_LINE, reason "patient_left"). A turn is always handled first:
        words the patient finished saying go to the server and Reachy says the answer (a risk or a goodbye then
        decides), and the medicine ending waits while the listener hears or holds the patient's words, at most
        MEDICINE_WAIT_SECONDS after medicine_due.
        """
        if not self.entered:
            self.entered = True
            self.companion_since = self.clock.monotonic()
            self.stream.set_max_fps(COMPANION_CAPTURE_FPS)
            await self._watch()
            if not (self._may_go_live() and await self._live_checkin()):
                await self._open_chat()
            return
        if self.chat_id is None:
            self._after_checkin()
            return
        await self._watch()   # a camera session the server lost is opened again
        if not self.task.get("microphone"):
            log.warning("slot %s: microphone consent withdrawn; today's chat stops", self.task_id)
            await self._close_chat("stopped", outcome="microphone_withdrawn")   # /end has no microphone reason
            return
        heard = None
        if self.voice is not None:
            await self._answer_pause()
            self._note_heard()
            heard = self.voice.take_utterance_with_metrics()
        if heard:
            text, metrics = heard
            sent = self.clock.monotonic()
            try:
                answer = await self._send_turn(text, metrics)
            except SessionLost:
                await self._close_chat("stopped")
                return
            stats = {"round_trip_ms": _ms(self.clock.monotonic() - sent)}
            await self._say(answer.get("speech_text"), stats, lively=not answer.get("risk"))
            self.last_spoke = self.clock.monotonic()
            await self._send_playback(answer.get("reply_turn_id"), stats)
            if answer.get("end"):
                await self._close_chat("risk" if answer.get("risk") else "goodbye")
            return
        now = self.clock.monotonic()
        if self.medicine_wanted:
            if self._listener_idle() or now - self.medicine_at >= MEDICINE_WAIT_SECONDS:
                await self._end_companion("medicine", MEDICINE_LINE)
            return   # else the patient's words go to the server first, at a later tick
        if CHECKIN_SILENCE is not None and now - self.last_spoke >= CHECKIN_SILENCE:
            await self._close_chat("silence")
        elif self._resting(now):
            await self._end_companion("patient_left", REST_LINE)

    async def _open_chat(self) -> None:
        """Today's chat: POST /conversations, then Reachy's opening line."""
        try:
            opened = await self.app.conversation_start(self.task_id, self.language, companion=True)
        except (NotAuthorised, RequestRejected, SessionLost) as exc:
            log.warning("slot %s: check-in unavailable (%s)", self.task_id, exc)
            self.outcomes["checkin"] = "unavailable"
            self._after_checkin()
            return
        self.chat_id = opened["conversation_id"]
        await self._robot("hold_head")
        stats: dict = {}
        await self._say(opened.get("speech_text"), stats)
        self.last_spoke = self.clock.monotonic()
        await self._send_playback(opened.get("reply_turn_id"), stats)

    async def _live_checkin(self) -> bool:
        """The live chat (spec F2). False when this robot app can't hold one: today's chat runs instead.

        One microphone reader: the speech listener is suspended for the whole chat and resumed after it, also when the
        chat fails or is stopped. The server stores the conversation and ends it. On its `fallback`, today's chat goes
        on: in the same conversation when the server already made one (Reachy says nothing new, it simply listens),
        else in a new one; the silence limit (CHECKIN_SILENCE) then counts from the fallback. A stop() wins over all
        of it.

        outcomes["live"] says how the live chat went: "ended" (it ended; outcomes["checkin"] has its reason), the
        fallback's reason, the robot's own reason to cut it short ("stopped", "microphone_withdrawn", or the
        companion's "patient_left" / "medicine" when the chat did not end after Reachy's closing line), or
        "unsupported".
        """
        chat = self.live_chat(self.app, self.robot, self.speaker, speech_models_dir(self.voice))
        if chat is None:
            self.outcomes["live"] = "unsupported"
            return False
        chat.gestures = self.gestures   # the settings page's check-in gestures switch (LiveChat reads it per block)
        chat.mood = self.mood           # the multimodal emotion engine, for the server's emotion tool request
        await self._robot("hold_head")
        if self.voice is not None:
            await self.run_blocking(self.voice.suspend)
        try:
            if self.stop_reason:   # stop() landed while the microphone was handed over: no chat
                kind, reason, conversation_id = "cut", "stopped", None
            else:
                kind, reason, conversation_id = await self._run_live(chat)
        finally:
            if self.voice is not None:
                self.voice.resume()
        if self.stop_reason:
            kind, reason = "cut", "stopped"
        if kind == "fallback":
            log.info("slot %s: the live chat fell back (%s); today's chat goes on", self.task_id, reason)
            self.outcomes["live"] = reason
            self.since = self.last_spoke = self.clock.monotonic()
            if conversation_id:
                self.chat_id = conversation_id
            else:
                await self._open_chat()
            return True
        self.outcomes["live"] = "ended" if kind == "ended" else reason
        self.outcomes["checkin"] = reason
        self._after_checkin()
        return True

    async def _run_live(self, chat) -> tuple[str, str, str | None]:
        """Run the live chat to its end: (kind, reason, conversation_id), kind "ended" or "fallback" as its LiveOutcome
        says. Every LIVE_POLL_SECONDS the session looks for a reason to stop it at once: stop() (the heartbeat's
        stop_all, the app shutting down) or the patient's microphone consent withdrawn (heartbeat). It is then
        cancelled: ("cut", that reason). A chat that fails: ("fallback", "error").

        Meanwhile the companion session goes on around it (spec 2026-10-12 companion 4.2-4.5): its camera
        (`_keep_watching`), the patient's voice (LiveChat.last_speech_at) and the medicine and rest rules
        (`_companion_ending`). Those ask the chat to end with Reachy's line (LiveChat.stop with `say`); a chat that has
        not ended COMPANION_STOP_SECONDS later is cancelled: ("cut", that reason). What the camera reports until the
        chat has ended (Reachy's closing line included) is chat-time (`_forward_extra_events`). The medicine ending
        waits while the patient speaks or stopped less than MEDICINE_SPEECH_GAP_SECONDS ago (LiveChat.last_voice_at,
        the voice detector's state), so their words reach the server's checks, at most MEDICINE_WAIT_SECONDS after
        medicine_due. Once the server has decided the ending (LiveChat.closing: its end or fallback came, and what it
        queued, the help line above all, plays out), the session asks for no ending of its own: only stop() and a
        withdrawn microphone still cut that.

        A voice detector that stopped (LiveChat.speech_start None after a failure), or a camera unusable for
        CAMERA_LOST_SECONDS (no camera session, or its state not "ok"; never merely the patient out of view), would
        leave GPT-Live unable to start again (camera first): the chat is stopped without a line and today's chat goes
        on in a new conversation, as after a fallback the robot decides itself: ("fallback", DETECTOR_FAILED or
        CAMERA_FAILED)."""
        run = asyncio.ensure_future(chat.run(self.task_id, self.language))
        ending, asked = None, 0.0
        self.camera_lost_since = None
        try:
            while True:
                done, _ = await asyncio.wait({run}, timeout=LIVE_POLL_SECONDS)
                if done:
                    break
                if self.stop_reason:
                    return "cut", "stopped", None
                if not self.task.get("microphone"):
                    log.warning("slot %s: microphone consent withdrawn; the live chat stops", self.task_id)
                    return "cut", "microphone_withdrawn", None
                now = self.clock.monotonic()
                if ending is not None:
                    if now - asked >= COMPANION_STOP_SECONDS:
                        log.error("slot %s: the live chat did not end within %.0f s of the robot's stop (%s); the "
                                  "robot stops it", self.task_id, COMPANION_STOP_SECONDS, ending)
                        return ("fallback" if ending in (DETECTOR_FAILED, CAMERA_FAILED) else "cut"), ending, None
                    await self._forward_extra_events()   # still chat-time: never to family
                    continue
                await self._keep_watching()
                if chat.last_speech_at is not None and (self.heard_at is None or chat.last_speech_at > self.heard_at):
                    self.heard_at = chat.last_speech_at
                if chat.closing:
                    continue   # the server ended it: its help line (or what it queued) is never cut by the robot
                ending = self._companion_ending(now)
                if ending == "medicine" and self._still_talking(chat.last_voice_at, now):
                    ending = None   # the patient's words first
                elif ending is not None:
                    asked = now
                    log.info("slot %s: the companion session ends (%s)", self.task_id, ending)
                    chat.stop(ending, say=self._line(MEDICINE_LINE if ending == "medicine" else REST_LINE))
                elif chat.speech_start is None:
                    log.warning("slot %s: the live chat's voice detector stopped, so GPT-Live could not start again; "
                                "today's chat goes on in a new conversation", self.task_id)
                    ending, asked = DETECTOR_FAILED, now
                    chat.stop(ending)
                elif self._camera_lost(now):
                    log.warning("slot %s: the companion's camera has not worked for %.0f s, so GPT-Live could not "
                                "start again; today's chat goes on in a new conversation", self.task_id,
                                CAMERA_LOST_SECONDS)
                    ending, asked = CAMERA_FAILED, now
                    chat.stop(ending)
        finally:
            if not run.done():
                run.cancel()
                if not (await asyncio.wait({run}, timeout=LIVE_STOP_SECONDS))[0]:
                    log.warning("slot %s: the live chat did not close within %.0f s", self.task_id,
                                LIVE_STOP_SECONDS)
        try:
            outcome = run.result()
        except Exception:
            log.exception("slot %s: the live chat failed", self.task_id)
            return "fallback", "error", None
        if ending in (DETECTOR_FAILED, CAMERA_FAILED) and outcome.kind == "ended" and not outcome.risk:
            return "fallback", ending, None
        return outcome.kind, outcome.reason, outcome.conversation_id

    # ── the companion session around either chat ─────────────────────────
    async def _watch(self) -> None:
        """The companion's camera (spec 2026-10-12 companion 4.2): the observe session stays attached for the whole
        session (a check-in task's search session goes on; after a dose, a new one starts). One the server lost is
        opened again at once; one that did not open is tried again every CAMERA_RETRY_SECONDS. A camera that won't
        open never ends the chat; a device no longer authorised (NotAuthorised) fails closed, as everywhere."""
        if self.monitor is not None and self.monitor_key == ("observe", None):
            return
        now = self.clock.monotonic()
        if now < self.camera_retry_at:
            return
        try:
            await self._ensure_session("observe")
        except NotAuthorised:
            raise
        except BridgeError as exc:
            self.camera_retry_at = now + CAMERA_RETRY_SECONDS
            log.warning("slot %s: the companion's camera did not open (%s); Reachy chats on", self.task_id, exc)

    async def _keep_watching(self) -> None:
        """While the live chat holds this session's tick: the frame stream's errors (tick() takes them otherwise; a
        lost session is opened again by `_watch`; NotAuthorised fails closed in tick()), and the latest answer, which
        updates seen_at (hand-to-mouth events are only marked as seen: `_forward_extra_events`)."""
        error = self.stream.take_error()
        if isinstance(error, NotAuthorised):
            raise error
        if isinstance(error, SessionLost):
            log.warning("slot %s: the companion's camera session was lost (%s)", self.task_id, error.detail)
            self._drop_session()
        elif error is not None:
            log.warning("slot %s: a companion camera frame failed (%s)", self.task_id, error)
        await self._watch()
        await self._forward_extra_events()

    def _note_heard(self) -> None:
        """Today's chat: the patient's voice for the rest rule, while the listener hears them speak or holds words
        they said that are not handed over yet."""
        if self.voice.hears_speech() or self.voice.heard_pause() is not None:
            self.heard_at = self.clock.monotonic()

    def _listener_idle(self) -> bool:
        """Today's chat: the listener neither hears the patient speak nor holds words they said that are not handed
        over yet (being decoded, or decoded and waiting)."""
        return self.voice is None or (not self.voice.hears_speech() and self.voice.heard_pause() is None)

    def _still_talking(self, voice_at: float | None, now: float) -> bool:
        """The live chat: the patient speaks, or stopped less than MEDICINE_SPEECH_GAP_SECONDS ago (voice_at:
        LiveChat.last_voice_at), and medicine_due came less than MEDICINE_WAIT_SECONDS ago."""
        return (now - self.medicine_at < MEDICINE_WAIT_SECONDS and voice_at is not None
                and now - voice_at < MEDICINE_SPEECH_GAP_SECONDS)

    def _camera_lost(self, now: float) -> bool:
        """The live chat: the companion's camera has been unusable (no camera session, or the stream's camera not
        "ok") continuously for CAMERA_LOST_SECONDS. Whether the patient is in view plays no part."""
        if self.monitor is not None and self.stream.camera.state == "ok":
            self.camera_lost_since = None
            return False
        if self.camera_lost_since is None:
            self.camera_lost_since = now
        return now - self.camera_lost_since >= CAMERA_LOST_SECONDS

    def _resting(self, now: float) -> bool:
        """The rest rule (spec 2026-10-12 companion 4.4): neither the patient's voice nor a verified sighting for
        REST_AFTER_SECONDS, counted from the latest of the companion session's start, the patient's last speech and
        the last sighting (seen_at)."""
        marks = [mark for mark in (self.companion_since, self.heard_at, self.seen_at) if mark is not None]
        return bool(marks) and now - max(marks) >= REST_AFTER_SECONDS

    def _companion_ending(self, now: float) -> str | None:
        """Why the robot ends the companion session now, if it does: a due medicine first, then the rest rule."""
        if self.medicine_wanted:
            return "medicine"
        return "patient_left" if self._resting(now) else None

    async def _end_companion(self, reason: str, lines: dict) -> None:
        """Today's chat: Reachy says why the companion session ends (REST_LINE, MEDICINE_LINE), then the conversation
        ends with `reason` and Reachy goes to sleep (spec 2026-10-12 companion 4.4, 4.5). The live chat does the same
        through LiveChat.stop (`_run_live`)."""
        log.info("slot %s: the companion session ends (%s)", self.task_id, reason)
        self._gesture(None)
        await self._say(self._line(lines))
        await self._close_chat(reason)

    def _line(self, lines: dict) -> str:
        return lines.get(self.language, lines["zh-TW"])

    async def _answer_pause(self) -> None:
        """The moment the VAD hears the patient pause, while SenseVoice is still decoding their words, Reachy says
        「嗯」 and starts thinking. Once per pause, and only for words said since Reachy last spoke: words said while
        it was answering aren't acknowledged after the answer. Reachy keeps still while the patient speaks.

        The 「嗯」 is handed to the speaker without waiting for it, so the turn goes out as soon as it is decoded.
        """
        ended = self.voice.heard_pause()
        if ended is None or ended < self.last_spoke:
            self._gesture(None)
            return
        if self.ack and ended != self.acked:
            self.acked = ended
            try:
                await self.run_blocking(self._start_ack)
            except Exception:   # a courtesy: never worth losing what the patient said
                log.exception("slot %s: the check-in acknowledgement failed", self.task_id)
        self._gesture(None if self.voice.hears_speech() else "think")

    def _start_ack(self) -> None:
        seconds = self.clips.start(ACK_CLIP)   # None without the clip (no English clips yet)
        if seconds:
            self.voice.note_ack(seconds)       # the listener must not hear it as the patient

    def _gesture(self, mode: str | None) -> None:
        """Reachy's check-in body language (robot.gesture never blocks); a failure is logged, never fatal."""
        if not self.gestures or mode == self.gesture_mode:
            return
        self.gesture_mode = mode
        try:
            self.robot.gesture(mode)
        except Exception:
            log.exception("slot %s: gesture %s failed", self.task_id, mode)

    async def _send_turn(self, text: str, metrics: dict | None) -> dict:
        """The patient's words, with their timings. Timings the server refuses (422) are dropped and the words sent
        again on their own: losing timings must never lose what the patient said. A 422 then is about the words."""
        try:
            return await self.app.conversation_turn(self.chat_id, text, metrics)
        except RequestRejected as exc:
            if exc.status != 422 or metrics is None:
                raise
            log.warning("slot %s: check-in timings refused (%s); sending the words without them", self.task_id, exc)
        return await self.app.conversation_turn(self.chat_id, text, None)

    async def _send_playback(self, turn_id, stats: dict) -> None:
        """Best effort: timings never break the conversation (the server may also refuse them), and a failed post
        doesn't count towards an outage (`conversation_turn_metrics` doesn't track one)."""
        if self.chat_id is None or turn_id is None or not stats:
            return
        try:
            await self.app.conversation_turn_metrics(self.chat_id, turn_id, stats)
        except BridgeError as exc:
            log.warning("slot %s: check-in timings not stored (%s)", self.task_id, exc)

    async def _close_chat(self, reason: str, outcome: str | None = None) -> None:
        """End today's chat with `reason` (POST /end); outcomes["checkin"] says `outcome`, else the reason."""
        chat_id, self.chat_id = self.chat_id, None
        if chat_id:
            try:
                await self.app.conversation_end(chat_id, reason)
            except BridgeError as exc:
                log.warning("slot %s: could not close the check-in (%s)", self.task_id, exc)
        self.outcomes["checkin"] = outcome or reason
        self._after_checkin()

    def _after_checkin(self) -> None:
        self._end_chat_events()
        self.companion_since = None   # the companion session is over: observing on is not companion time
        self._gesture(None)
        self.stream.set_max_fps(None)   # the camera's own rate again (POST_SLOT_OBSERVE looks for an extra dose)
        rest = self._is_checkin_task() or self.outcomes.get("checkin") in COMPANION_ENDINGS
        self._go(WIND_DOWN if rest else POST_SLOT_OBSERVE)

    async def _post_slot(self):
        await self._ensure_session("observe")
        if not self.entered:
            self.entered = True
            await self._robot("hold_head")
        if self.clock.monotonic() - self.since >= POST_SLOT_SECONDS:
            await self._end_session()
            self._go(WIND_DOWN)

    async def _already_taken(self):
        await self._play("already_taken")
        self._go(WIND_DOWN)

    async def _wind_down(self):
        if not self.entered:
            await self._end_session()
            if self.outcomes.get("checkin") not in COMPANION_ENDINGS:   # else Reachy has just said why it rests
                await self._play("wind_down")
            self.entered = True
        await self._advance("completed", {"outcomes": self._outcome_detail()})
        await self._robot("sleep")
        self._finish("completed")

    async def _waiting_other_client(self):
        if not self.entered:
            self.entered = True
            if not self.busy_announced:
                self.busy_announced = True
                await self._play("waiting_for_tablet")
        expires = _parse_time(self.task.get("expires_at"))
        if expires is not None and self.clock.wall() >= expires:
            await self._shutdown("busy_other_client", status="aborted")
            return
        if self.clock.monotonic() - self.since >= BUSY_RETRY_SECONDS:
            self._go(self.resume_state or SEARCHING)

    # ── helpers ──────────────────────────────────────────────────────────
    def _go(self, state: str) -> None:
        log.info("slot %s: %s -> %s", self.task_id, self.state, state)
        self._sync_mood(state)
        self._sync_follow(state)
        self.state = state
        self.since = self.clock.monotonic()
        self.entered = False
        self.wrong_since = None
        self.wrong_announced = False
        self.done_at = None
        self.done_word = None
        self.away_since = None

    def _next_dose(self) -> None:
        self.index += 1
        self.help_at = None
        if self.index < len(self.doses):
            self._go(MED_PROMPT)
        else:
            self._go(CHECKIN if self._may_chat() else POST_SLOT_OBSERVE)

    def _finish(self, result: str) -> None:
        log.info("slot %s finished: %s %s", self.task_id, result, self.outcomes)
        self._sync_mood(SLEEP)
        self._sync_follow(SLEEP)
        self.result = result
        self.done = True
        self.state = SLEEP
        self.gesture_mode = None   # robot.sleep() stops any gesture before it moves
        self.stream.set_max_fps(None)   # a companion session cut short leaves no slow camera for the next task
        self._sync_voice()

    def _enter_waiting_other_client(self) -> None:
        self._drop_session()
        if self.state == WAITING_OTHER_CLIENT:
            self.since = self.clock.monotonic()
            return
        self.resume_state = self.state
        self._go(WAITING_OTHER_CLIENT)

    @staticmethod
    def _unresolved(dose: dict) -> bool:
        return dose.get("intake_stats") not in RESOLVED

    def _confirm_source(self, dose: dict, latest: dict) -> str:
        if not dose.get("supported", False):
            return "unsupported_dose"
        if not self.task.get("auto_record", False) or latest.get("auto_commit") is False:
            return "auto_record_off"
        if latest.get("degraded"):
            return "degraded"
        return "uncertain_detection"

    def _outcome_detail(self) -> dict:
        return {str(intk_id): outcome for intk_id, outcome in self.outcomes.items()}

    def _latest(self) -> dict:
        """The attached session's latest answer. A verified one is a sighting when that answer came
        (MonitorStream.latest_at), never later: after the camera stops, the last answer is not read as the patient
        still in view (the rest rule, and who a refused dose may be explained to, _explain)."""
        latest = self.stream.latest
        if not self.monitor or not latest or latest.get("session_id") != self.monitor.get("session_id"):
            return {}
        if latest.get("identity_status") == "verified" and self.stream.latest_at is not None:
            self.seen_at = self.stream.latest_at
        return latest

    async def _robot(self, name: str, *args):
        return await self.run_blocking(getattr(self.robot, name), *args)

    async def _play(self, clip_id: str) -> bool:
        return await self._speak(self.clips.play, clip_id)

    async def _speak(self, play, *args):
        """Play a clip with the listener muted, so the robot never hears its own prompts."""
        if self.voice is not None:
            self.voice.hold()
        try:
            return await self.run_blocking(play, *args)
        finally:
            if self.voice is not None:
                self.voice.release()

    # ── listening for "I finished" ───────────────────────────────────────
    def _may_listen(self) -> bool:
        return self.voice is not None and self.voice.available and bool(self.task.get("microphone"))

    def _is_checkin_task(self) -> bool:
        return self.task.get("reason") == "checkin"

    def _may_chat(self) -> bool:
        """A check-in needs every check-in consent (task["checkin"]), the patient's microphone consent as the
        heartbeat last said it (task["microphone"]), ears and a voice."""
        return (self.voice is not None and self.voice.available and self.speaker is not None
                and self.speaker.available and bool(self.task.get("checkin")) and bool(self.task.get("microphone")))

    def _may_go_live(self) -> bool:
        """A live chat needs the server's go-ahead (task["live"]: its test switch, the robot's live voice switch,
        check-in consent and a usable live model), the patient's microphone consent, and Reachy's own voice: the
        server has Reachy say the help line itself, never GPT-Live. The server starts GPT-Live only when the robot
        hears the patient begin to speak while its camera recognises them (camera first), so it also needs the
        listener's voice detector (speech_models_dir) and the companion's camera session: without either the patient
        would talk to a Reachy that never answers, and today's chat runs instead."""
        return (bool(self.task.get("live")) and bool(self.task.get("microphone")) and self.speaker is not None
                and self.speaker.available and speech_models_dir(self.voice) is not None and self.monitor is not None)

    async def _say(self, text: str | None, stats: dict | None = None, lively: bool = True) -> bool:
        """Speak a check-in line; `stats` receives the speaker's timings. In the conversation Reachy's antennas
        move from its first sound until it has finished (unless not `lively`: the help line after a risk is said
        still), then it is still again to listen."""
        if self.speaker is None or not text:
            return False
        if self.voice is not None:
            self.voice.note_spoken(text)   # hearing it back through the microphone isn't the patient answering
        if self.state != CHECKIN or self.chat_id is None:
            return bool(await self._speak(self.speaker.say, text, stats))
        if not lively:
            self._gesture(None)
        try:
            return bool(await self._speak(self.speaker.say, text, stats,
                                          (lambda: self._gesture("speak")) if lively else None))
        finally:
            self._gesture(None)

    def _sync_mood(self, state: str) -> None:
        """Emotion samples count towards the dose only from the prompt until it is resolved or left."""
        if self.mood is None:
            return
        try:
            if state == WATCHING and self.state != WATCHING:
                self.mood.begin(self.doses[self.index]["intk_id"] if self.index < len(self.doses) else None)
            elif state != WATCHING and self.state == WATCHING:
                self.mood.end()
        except Exception:   # a soft cue, never the dose's flow
            log.exception("mood tracking failed")

    def _sync_follow(self, state: str) -> None:
        if self.follow is None:
            return
        if state == WATCHING and self.state != WATCHING:
            self.follow.start()
        elif state != WATCHING and self.state == WATCHING:
            self.follow.stop()

    async def _follow_face(self, latest: dict, now: float) -> None:
        """Keep the head on the verified face (the server's target_box); nothing while a candidate event is being
        verified, so the hand-to-mouth detector works on a still camera."""
        if self.follow is None or latest.get("candidate"):
            return
        box = latest.get("target_box") if latest.get("identity_status") == "verified" else None
        await self.run_blocking(self.follow.update, box, now)

    def _mood_evidence(self) -> dict:
        if self.mood is None:
            return {}
        try:
            summary = self.mood.summary()
        except Exception:
            log.exception("mood summary failed")
            return {}
        compute = getattr(self.stream, "emotion_compute", None)
        if callable(compute):
            summary = {**summary, "compute": compute()}
        return {"emotion": summary}

    def _sync_voice(self) -> None:
        if self.voice is None:
            return
        if not self.done and self.state == CHECKIN:
            self.voice.set_active(self.chat_id is not None and self._may_chat(), mode="chat")
        else:
            self.voice.set_active(not self.done and self.state == WATCHING and self._may_listen(), mode="done")

    def _note_done(self, now: float) -> None:
        if self.done_at is not None or not self._may_listen():
            return
        word = self.voice.heard_since(self.since)
        if word:
            log.info("slot %s: patient says the dose is finished", self.task_id)
            self.done_at, self.done_word = now, word

    def _walked_away(self, identity, now: float) -> bool:
        if identity == "verified":
            self.away_since = None
            return False
        if self.away_since is None:
            self.away_since = now
        return now - self.away_since >= AWAY_TIMEOUT

    async def _check_wrong_person(self, identity) -> None:
        if identity not in ("mismatch", "ambiguous"):
            self.wrong_since = None
            self.wrong_announced = False
            return
        now = self.clock.monotonic()
        if self.wrong_since is None:
            self.wrong_since = now
        elif not self.wrong_announced and now - self.wrong_since > WRONG_PERSON_SECONDS:
            self.wrong_announced = True
            await self._play("waiting_for_patient")

    async def _refresh_doses(self) -> None:
        current = await self.app.tasks_current()
        if not current or current.get("task_id") != self.task_id:
            raise TaskGone(self.task_id)
        fresh = {dose["intk_id"]: dose for dose in current.get("doses", [])}
        for dose in self.doses:
            dose.update(fresh.get(dose["intk_id"], {}))
        for key in ("auto_record", "microphone", "checkin", "live"):
            if key in current:
                self.task[key] = current[key]
        if current.get("expires_at"):
            self.task["expires_at"] = current["expires_at"]

    async def _recover(self) -> None:
        """After a 409 on the monitor stream: re-read the doses, then the state handler restarts the session."""
        await self._refresh_doses()
        self.needs_recovery = False
        if self.state in (MED_PROMPT, WATCHING):
            dose = self.doses[self.index]
            if not self._unresolved(dose):
                self.outcomes[dose["intk_id"]] = {"taken": "recorded", "pending_confirmation": "needs_confirm"}.get(
                    dose.get("intake_stats"), "already_resolved")
                self._next_dose()

    async def _advance(self, target: str, detail: dict | None = None) -> None:
        if target == "completed":
            await self._advance("in_progress")
        if target in OPEN_ORDER:
            current = OPEN_ORDER.index(self.status) if self.status in OPEN_ORDER else 0
            steps = OPEN_ORDER[current + 1:OPEN_ORDER.index(target) + 1]
        else:
            steps = (target,)
        for status in steps:
            try:
                await self.app.task_status(self.task_id, status, detail if status == target else None)
            except SessionLost as exc:
                log.warning("task %s: status %s refused (%s)", self.task_id, status, exc.detail)
            self.status = status

    async def _ensure_session(self, mode: str, intk_id: int | None = None) -> None:
        companion = self.companion_since is not None and mode == "observe"
        if self.monitor is not None and self.monitor_key == (mode, intk_id) and self.monitor_companion == companion:
            return
        await self._end_session()
        self.monitor = await self.app.monitor_start(mode, intk_id, self.task_id, companion=companion)
        self.monitor_key = (mode, intk_id)
        self.monitor_companion = companion
        self.stream.attach(self.monitor)

    def _drop_session(self) -> None:
        self.stream.detach()
        self.monitor = None
        self.monitor_key = None

    async def _end_session(self) -> None:
        monitor = self.monitor
        self._drop_session()
        if monitor:
            try:
                await self.app.monitor_end(monitor["session_id"], monitor["generation"])
            except (SessionLost, AppUnreachable):
                pass   # already gone, or the server evicts it once landmarks stop

    def _end_chat_events(self) -> None:
        """The chat is over: every hand-to-mouth event its camera session has reported so far happened during it
        (Reachy's last line included), and so did one the server reports later for a frame already streamed (an
        answer still on its way). `_forward_extra_events` sends none of them to family."""
        for event in self._latest().get("extra_events") or []:
            if event.get("event_id"):
                self.sent_extra.add(event["event_id"])
        if self.monitor is not None:
            self.chat_end = (self.monitor["session_id"], self.stream.frame_seq)

    async def _forward_extra_events(self) -> None:
        latest = self._latest()
        chat_end = self.chat_end if self.chat_end and self.chat_end[0] == latest.get("session_id") else None
        for event in latest.get("extra_events") or []:
            event_id = event.get("event_id")
            if not event_id or event_id in self.sent_extra:
                continue
            seq = event.get("frame_seq")
            if self.state == CHECKIN or (chat_end and seq is not None and int(seq) <= chat_end[1]):
                # The companion's camera only tells who is there: a hand at the mouth while chatting (a drink, a
                # snack) is no news for family, and at COMPANION_CAPTURE_FPS the detector isn't calibrated.
                self.sent_extra.add(event_id)
                continue
            try:
                await self.app.extra_event(self.task_id, event_id, event.get("decision"),
                                           float(event.get("confidence") or 0))
            except (SessionLost, RequestRejected) as exc:
                log.warning("extra event %s not stored: %s", event_id, exc)   # never worth aborting the slot
            self.sent_extra.add(event_id)

    async def _fail_closed(self, reason: str) -> None:
        """App lost or authorisation withdrawn: stop streaming and put the robot to sleep, no app calls."""
        self._drop_session()
        try:
            await self._robot("sleep")
        except Exception:
            log.exception("robot sleep failed")
        self._finish(reason)

    async def _shutdown(self, reason: str, status: str | None) -> None:
        """Best-effort: end the monitor session and any check-in, report the task status, sleep the robot."""
        monitor = self.monitor
        self._drop_session()
        if self.chat_id:
            chat_id, self.chat_id = self.chat_id, None
            try:
                await self.app.conversation_end(chat_id, "stopped")
            except BridgeError:
                pass
        try:
            if monitor:
                await self.app.monitor_end(monitor["session_id"], monitor["generation"])
        except BridgeError:
            pass
        if status:
            try:
                await self._advance(status, {"reason": reason, "outcomes": self._outcome_detail()})
            except BridgeError as exc:
                log.warning("could not report %s: %s", status, exc)
        try:
            await self._robot("sleep")
        except Exception:
            log.exception("robot sleep failed")
        self._finish(reason)
