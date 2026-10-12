"""Async client for the app's private device API (/api/device/*, port 8001)."""

import json
import logging
import time

import httpx

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0
METRICS_TIMEOUT = 3.0   # timings only: never keep a conversation waiting on them


class BridgeError(Exception):
    """`detail` is the error body's "detail"; `body` the whole JSON body (a refusal carries more fields)."""

    def __init__(self, message: str = "", status: int | None = None, detail=None, body: dict | None = None):
        super().__init__(message or str(detail or status or ""))
        self.status = status
        self.detail = detail
        self.body = body


class AppUnreachable(BridgeError):
    """Network failure or a 5xx other than 503: the app is effectively gone."""


class NotAuthorised(BridgeError):
    """401/403: token revoked or robot consent withdrawn."""


class SessionLost(BridgeError):
    """409: the monitor session (or the dose/task it referred to) no longer exists server-side."""


class BusyOtherClient(SessionLost):
    """409 busy_other_client: a browser session holds this dose."""


# Overdose protection: why the server won't start or record a dose now.
DOSE_REFUSALS = frozenset({"dose_not_due_yet", "dose_too_soon", "daily_max_reached", "dose_expired"})


class DoseRefused(SessionLost):
    """409 from overdose protection: the dose isn't due yet, comes too soon after the last one, would pass the day's
    maximum, or was missed too long ago. `detail` is that code and `body` the server's fields, among them
    `speech_text`, one sentence for Reachy to say why (Simplified Chinese for its voice, or English)."""

    @property
    def intk_id(self) -> int | None:
        value = (self.body or {}).get("intk_id")
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @property
    def speech_text(self) -> str | None:
        text = (self.body or {}).get("speech_text")
        return text.strip() if isinstance(text, str) and text.strip() else None


class ServiceUnavailable(BridgeError):
    """503: identity/emotion models are not ready."""


class RequestRejected(BridgeError):
    """Any other 4xx."""


def _body(response: httpx.Response) -> tuple:
    """(detail, body) of an error response: FastAPI's {"detail": ...}, or a refusal's {"detail": code, ...fields}."""
    try:
        body = response.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return response.text or None, None
    return body.get("detail"), body


def _refusal(detail, body) -> dict | None:
    """The fields of an overdose-protection refusal: flat, as main.py sends them, or nested in "detail" (an
    HTTPException with a dict detail). None for any other 409."""
    for fields in (body, detail):
        if not isinstance(fields, dict):
            continue
        code = fields.get("detail") if isinstance(fields.get("detail"), str) else fields.get("code")
        if code in DOSE_REFUSALS:
            return {**fields, "detail": code}
    return None


def _is_busy(detail) -> bool:
    if isinstance(detail, dict):
        return "busy_other_client" in (detail.get("code"), detail.get("error"), detail.get("detail"))
    return isinstance(detail, str) and "busy_other_client" in detail


class AppClient:
    def __init__(self, base_url: str, token: str, *, transport: httpx.AsyncBaseTransport | None = None,
                 clock=time.monotonic, timeout: float = DEFAULT_TIMEOUT):
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"), transport=transport, timeout=timeout,
                                       headers={"Authorization": f"Bearer {token}"})
        self._clock = clock
        # First failure time of the current outage; the slot session fails closed on it.
        self.unreachable_since: float | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _request(self, method: str, path: str, *, track_outage: bool = True, **kwargs):
        """`track_outage=False`: a failure of this best-effort request still raises AppUnreachable, but doesn't
        start an outage (`unreachable_since`, which makes the slot session fail closed)."""
        try:
            response = await self._http.request(method, path, **kwargs)
        except httpx.TransportError as exc:
            if track_outage:
                self._unreachable()
            raise AppUnreachable(f"{method} {path}: {exc!r}") from exc
        status = response.status_code
        if status >= 500 and status != 503:
            if track_outage:
                self._unreachable()
            raise AppUnreachable(f"{method} {path}: HTTP {status}", status)
        self.unreachable_since = None
        if status == 204:
            return None
        if status < 300:
            return response.json() if response.content else None
        detail, body = _body(response)
        if status in (401, 403):
            raise NotAuthorised(status=status, detail=detail, body=body)
        if status == 409:
            refusal = _refusal(detail, body)
            if refusal is not None:
                raise DoseRefused(status=status, detail=refusal["detail"], body=refusal)
            raise (BusyOtherClient if _is_busy(detail) else SessionLost)(status=status, detail=detail, body=body)
        if status == 503:
            raise ServiceUnavailable(status=status, detail=detail, body=body)
        raise RequestRejected(f"{method} {path}: HTTP {status} {detail}", status, detail, body)

    def _unreachable(self) -> None:
        if self.unreachable_since is None:
            self.unreachable_since = self._clock()

    # ── tasks ────────────────────────────────────────────────────────────
    async def tasks_next(self, wait: float = 25) -> dict | None:
        wait = max(0, min(25, int(wait)))
        return await self._request("GET", "/api/device/tasks/next", params={"wait": wait},
                                   timeout=wait + DEFAULT_TIMEOUT)

    async def tasks_current(self) -> dict | None:
        return await self._request("GET", "/api/device/tasks/current")

    async def task_status(self, task_id: str, status: str, detail: dict | None = None) -> dict | None:
        return await self._request("POST", f"/api/device/tasks/{task_id}/status",
                                   json={"status": status, "detail": detail})

    async def confirmation(self, task_id: str, intk_id: int, source: str, evidence: dict | None) -> dict | None:
        return await self._request("POST", f"/api/device/tasks/{task_id}/confirmation",
                                   json={"intk_id": intk_id, "source": source, "evidence": evidence})

    async def extra_event(self, task_id: str, event_id: str, decision: str, confidence: float) -> dict | None:
        return await self._request("POST", f"/api/device/tasks/{task_id}/extra-event",
                                   json={"event_id": event_id, "decision": decision, "confidence": confidence})

    async def heartbeat(self, payload: dict, camera: dict | None = None) -> dict:
        """`camera`: the camera watchdog's snapshot (state, fps, since, media), sent as "camera" when given. Servers
        before it ignore the key."""
        body = payload if camera is None else {**payload, "camera": camera}
        return await self._request("POST", "/api/device/heartbeat", json=body) or {}

    # ── monitor ──────────────────────────────────────────────────────────
    async def monitor_start(self, mode: str, intk_id: int | None, task_id: str) -> dict:
        body = {"mode": mode, **({} if intk_id is None else {"intk_id": intk_id}), "task_id": task_id}
        return await self._request("POST", "/api/device/monitor/start", json=body)

    async def monitor_landmarks(self, packet: dict) -> dict:
        return await self._request("POST", "/api/device/monitor/landmarks", json=packet)

    async def monitor_vision(self, session_id: str, generation: str, frame_seq: int, jpeg: bytes) -> dict:
        return await self._request(
            "POST", "/api/device/monitor/vision",
            data={"session_id": session_id, "generation": generation, "frame_seq": str(frame_seq)},
            files={"file": ("frame.jpg", jpeg, "image/jpeg")})

    async def monitor_frame(self, session_id: str, generation: str, frame_seq: int, timestamp: float,
                            jpeg: bytes, emotion: dict | None = None) -> dict:
        """A camera frame for the server to compute the landmarks of (`timestamp`: capture time, seconds).

        `emotion`: the robot's own multimodal score of the verified face on a recent frame ({"probabilities",
        "target_box", "timestamp"}), as a JSON form field. Servers before the multimodal update ignore it.
        """
        data = {"session_id": session_id, "generation": generation, "frame_seq": str(frame_seq),
                "timestamp": repr(float(timestamp))}
        if emotion is not None:
            data["emotion"] = json.dumps(emotion)
        return await self._request(
            "POST", "/api/device/monitor/frame", data=data,
            files={"file": ("frame.jpg", jpeg, "image/jpeg")})

    # ── check-in conversation (text only: speech is turned into text on the robot) ──
    async def conversation_start(self, task_id: str, language: str) -> dict:
        return await self._request("POST", "/api/device/conversations",
                                   json={"task_id": task_id, "language": language})

    async def conversation_turn(self, conversation_id: str, text: str, metrics: dict | None = None) -> dict:
        """One thing the patient said, with how long hearing it took (`metrics`: milliseconds, no audio).

        The answer carries `reply_turn_id`, the server's id of Reachy's reply, for `conversation_turn_metrics`.
        """
        # The server gives the model LLM_DEADLINE_SECONDS (8 s by default); older servers waited up to 2 x 25 s.
        return await self._request("POST", f"/api/device/conversations/{conversation_id}/turn",
                                   json={"text": text, "metrics": metrics}, timeout=60)

    async def conversation_turn_metrics(self, conversation_id: str, turn_id: int, metrics: dict) -> dict | None:
        """How long one of Reachy's lines took to arrive and to speak (milliseconds); merged into that turn.

        Timings only: a slow or failing metrics endpoint must never make the slot session fail closed.
        """
        return await self._request("POST", f"/api/device/conversations/{conversation_id}/turns/{turn_id}/metrics",
                                   json={"metrics": metrics}, timeout=METRICS_TIMEOUT, track_outage=False)

    async def conversation_end(self, conversation_id: str, reason: str) -> dict | None:
        return await self._request("POST", f"/api/device/conversations/{conversation_id}/end",
                                   json={"reason": reason})

    async def monitor_end(self, session_id: str, generation: str) -> dict | None:
        return await self._request("POST", "/api/device/monitor/end",
                                   json={"session_id": session_id, "generation": generation})
