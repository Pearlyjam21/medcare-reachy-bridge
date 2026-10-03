"""Pure slot state machine (spec 01 §7).

`SlotSession.tick()` is called a few times per second by the runner. It never blocks on
hardware directly: robot and clip calls go through `run_blocking` (inline in tests,
`asyncio.to_thread` in the runner). The frame pipeline (`stream`) posts frames to whatever
monitor session is attached and exposes the latest `public()` state it got back.

While a dose is watched, and only while the task says the patient's microphone consent is
current, `voice` listens for the patient saying they have finished. Saying so is a claim, not
evidence: if the camera does not resolve the dose within DONE_GRACE, it goes to caregiver
confirmation as `patient_claim`.

In a check-in conversation Reachy says 「嗯」 the moment the patient pauses (`ack`) and moves
while it prepares and speaks its reply (`gestures`); either can be turned off in the settings.

Every observed result is advisory: the server decides whether anything is recorded. When the
patient's overdose protection refuses a dose (not due yet, too soon after the last one, the
day's maximum reached, or missed too long ago), Reachy says the server's one-sentence reason
once and moves on to the next dose. It files nothing more: after "I finished" or a camera
event, the server itself alerts family to a possible double dose.
"""

import logging
import time
from datetime import datetime, timezone

from medcare_reachy.bridge.app_client import (
    AppUnreachable, BridgeError, BusyOtherClient, DoseRefused, NotAuthorised, RequestRejected, ServiceUnavailable,
    SessionLost)

log = logging.getLogger(__name__)

RESOLVED = frozenset({"taken", "pending_confirmation", "skipped"})
SEARCH_TIMEOUT = 600.0          # 10 min to find the patient (also: patient gone before a later prompt)
WATCH_TIMEOUT = 180.0           # 3 min without an event -> help clip
HELP_TIMEOUT = 180.0            # 3 more min -> leave the dose pending
DONE_GRACE = 8.0                # after "I finished": time for the camera to resolve the dose first
AWAY_TIMEOUT = 90.0             # patient out of sight this long while watched -> leave the dose pending
CHECKIN_SILENCE = 25.0          # check-in: no answer this long after Reachy spoke -> say goodbye
CHECKIN_MAX = 300.0             # check-in: at most 5 minutes
# Spoken (not prerecorded) greeting for a conversation-only task, already in the voice's Simplified characters.
CHECKIN_GREETING = {"zh-TW": "您好！我来找您聊聊天。", "en": "Hello! I came to have a chat with you."}
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


class SlotSession:
    def __init__(self, task: dict, *, app, robot, clips, stream, voice=None, speaker=None, language="zh-TW",
                 clock=None, run_blocking=_inline, ack=True, gestures=True):
        self.task = task
        self.task_id = task["task_id"]
        self.status = task.get("status") or "leased"
        self.doses = [dict(dose) for dose in task.get("doses", [])]
        self.app, self.robot, self.clips, self.stream, self.voice = app, robot, clips, stream, voice
        self.speaker, self.language = speaker, language
        self.ack, self.gestures = ack, gestures
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
        self.sent_extra: set[str] = set()
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
        await self._speak(self.clips.play_med_prompt, dose.get("med_id"))
        if self._may_listen():
            await self._play("say_when_done")
        self._go(WATCHING)

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
            self._next_dose()
            return
        if self.help_at is None and now - self.since >= WATCH_TIMEOUT:
            self.help_at = now
            await self._play("help")
        elif self.help_at is not None and now - self.help_at >= HELP_TIMEOUT:
            self.outcomes[dose["intk_id"]] = "left_pending"
            self._next_dose()

    async def _needs_confirm(self, dose: dict, candidate: dict, latest: dict):
        source = self._confirm_source(dose, latest)
        evidence = {"event_id": candidate.get("event_id"), "decision": candidate.get("decision"),
                    "confidence": candidate.get("confidence"), "frame_seq": candidate.get("frame_seq"),
                    "degraded": bool(latest.get("degraded")), "landmark_fps": latest.get("landmark_fps"),
                    "said_done": self.done_at is not None}
        await self._confirm(dose, source, evidence)

    async def _patient_claim(self, dose: dict, latest: dict):
        """Heard "I finished" but the camera resolved nothing: a caregiver checks the pill box."""
        evidence = {"said_done": True, "phrase": self.done_word, "camera": "no_event",
                    "degraded": bool(latest.get("degraded")), "landmark_fps": latest.get("landmark_fps")}
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
        """A short conversation: the patient's words become text on the robot; the server answers.

        Each turn carries how long hearing it took, and each of Reachy's lines is followed by how long it took to
        arrive and to speak (milliseconds, for the patient's conversation history).
        """
        if not self.entered:
            self.entered = True
            await self._end_session()   # the camera isn't needed to chat
            try:
                opened = await self.app.conversation_start(self.task_id, self.language)
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
            return
        if self.chat_id is None:
            self._after_checkin()
            return
        heard = None
        if self.voice is not None:
            await self._answer_pause()
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
        if now - self.last_spoke >= CHECKIN_SILENCE or now - self.since >= CHECKIN_MAX:
            await self._close_chat("silence")

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

    async def _close_chat(self, reason: str) -> None:
        chat_id, self.chat_id = self.chat_id, None
        if chat_id:
            try:
                await self.app.conversation_end(chat_id, reason)
            except BridgeError as exc:
                log.warning("slot %s: could not close the check-in (%s)", self.task_id, exc)
        self.outcomes["checkin"] = reason
        self._after_checkin()

    def _after_checkin(self) -> None:
        self._gesture(None)
        self._go(WIND_DOWN if self._is_checkin_task() else POST_SLOT_OBSERVE)

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
        self.result = result
        self.done = True
        self.state = SLEEP
        self.gesture_mode = None   # robot.sleep() stops any gesture before it moves
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
        latest = self.stream.latest
        if not self.monitor or not latest or latest.get("session_id") != self.monitor.get("session_id"):
            return {}
        if latest.get("identity_status") == "verified":
            self.seen_at = self.clock.monotonic()   # who a refused dose may be explained to (_explain)
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
        """A check-in needs every check-in consent (task["checkin"]), ears and a voice."""
        return (self.voice is not None and self.voice.available and self.speaker is not None
                and self.speaker.available and bool(self.task.get("checkin")))

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
        for key in ("auto_record", "microphone", "checkin"):
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
        if self.monitor is not None and self.monitor_key == (mode, intk_id):
            return
        await self._end_session()
        self.monitor = await self.app.monitor_start(mode, intk_id, self.task_id)
        self.monitor_key = (mode, intk_id)
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

    async def _forward_extra_events(self) -> None:
        for event in self._latest().get("extra_events") or []:
            event_id = event.get("event_id")
            if not event_id or event_id in self.sent_extra:
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
