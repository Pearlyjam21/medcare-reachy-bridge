"""Pure slot state machine (spec 01 §7, Phase 2: no microphone).

`SlotSession.tick()` is called a few times per second by the runner. It never blocks on
hardware directly: robot and clip calls go through `run_blocking` (inline in tests,
`asyncio.to_thread` in the runner). The frame pipeline (`stream`) posts landmarks/JPEGs to
whatever monitor session is attached and exposes the latest `public()` state it got back.

Every observed result is advisory: the server decides whether anything is recorded.
"""

import logging
import time
from datetime import datetime, timezone

from reachy_bridge.app_client import (
    AppUnreachable, BridgeError, BusyOtherClient, NotAuthorised, RequestRejected, ServiceUnavailable, SessionLost)

log = logging.getLogger(__name__)

RESOLVED = frozenset({"taken", "pending_confirmation", "skipped"})
SEARCH_TIMEOUT = 600.0          # 10 min to find the patient (also: patient gone before a later prompt)
WATCH_TIMEOUT = 180.0           # 3 min without an event -> help clip
HELP_TIMEOUT = 180.0            # 3 more min -> leave the dose pending
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


class TaskGone(Exception):
    """The leased task is no longer this device's current task (expired, aborted server-side)."""


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def wall(self) -> datetime:
        return datetime.now(timezone.utc)


async def _inline(fn, *args):
    return fn(*args)


def _parse_time(value) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class SlotSession:
    def __init__(self, task: dict, *, app, robot, clips, stream, clock=None, run_blocking=_inline):
        self.task = task
        self.task_id = task["task_id"]
        self.status = task.get("status") or "leased"
        self.doses = [dict(dose) for dose in task.get("doses", [])]
        self.app, self.robot, self.clips, self.stream = app, robot, clips, stream
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
        self.handlers = {
            WAKE: self._wake, ANNOUNCE: self._announce, SEARCHING: self._searching,
            MED_PROMPT: self._med_prompt, WATCHING: self._watching, POST_SLOT_OBSERVE: self._post_slot,
            ALREADY_TAKEN: self._already_taken, WIND_DOWN: self._wind_down,
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

    # ── states ───────────────────────────────────────────────────────────
    async def _wake(self):
        if not await self._robot("is_reachable"):
            await self._shutdown("robot_offline", status="aborted")
            return
        await self._robot("wake")
        await self._advance("searching")
        self._go(ANNOUNCE)

    async def _announce(self):
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
        await self.run_blocking(self.clips.play_med_prompt, dose.get("med_id"))
        self._go(WATCHING)

    async def _watching(self):
        dose = self.doses[self.index]
        await self._ensure_session("dose", dose["intk_id"])   # re-created after a lost session
        latest = self._latest()
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
        now = self.clock.monotonic()
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
                    "degraded": bool(latest.get("degraded")), "landmark_fps": latest.get("landmark_fps")}
        try:
            await self.app.confirmation(self.task_id, dose["intk_id"], source, evidence)
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

    def _next_dose(self) -> None:
        self.index += 1
        self.help_at = None
        self._go(MED_PROMPT if self.index < len(self.doses) else POST_SLOT_OBSERVE)

    def _finish(self, result: str) -> None:
        log.info("slot %s finished: %s %s", self.task_id, result, self.outcomes)
        self.result = result
        self.done = True
        self.state = SLEEP

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
        return latest

    async def _robot(self, name: str, *args):
        return await self.run_blocking(getattr(self.robot, name), *args)

    async def _play(self, clip_id: str) -> bool:
        return await self.run_blocking(self.clips.play, clip_id)

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
        if "auto_record" in current:
            self.task["auto_record"] = current["auto_record"]
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
        """Best-effort: end the monitor session, report the task status, sleep the robot."""
        monitor = self.monitor
        self._drop_session()
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
