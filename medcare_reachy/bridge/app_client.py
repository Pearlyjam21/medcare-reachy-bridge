"""Async client for the app's private device API (/api/device/*, port 8001)."""

import logging
import time

import httpx

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10.0


class BridgeError(Exception):
    def __init__(self, message: str = "", status: int | None = None, detail=None):
        super().__init__(message or str(detail or status or ""))
        self.status = status
        self.detail = detail


class AppUnreachable(BridgeError):
    """Network failure or a 5xx other than 503: the app is effectively gone."""


class NotAuthorised(BridgeError):
    """401/403: token revoked or robot consent withdrawn."""


class SessionLost(BridgeError):
    """409: the monitor session (or the dose/task it referred to) no longer exists server-side."""


class BusyOtherClient(SessionLost):
    """409 busy_other_client: a browser session holds this dose."""


class ServiceUnavailable(BridgeError):
    """503: identity/emotion models are not ready."""


class RequestRejected(BridgeError):
    """Any other 4xx."""


def _detail(response: httpx.Response):
    try:
        return response.json().get("detail")
    except (ValueError, AttributeError):
        return response.text or None


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

    async def _request(self, method: str, path: str, **kwargs):
        try:
            response = await self._http.request(method, path, **kwargs)
        except httpx.TransportError as exc:
            self._unreachable()
            raise AppUnreachable(f"{method} {path}: {exc!r}") from exc
        status = response.status_code
        if status >= 500 and status != 503:
            self._unreachable()
            raise AppUnreachable(f"{method} {path}: HTTP {status}", status)
        self.unreachable_since = None
        if status == 204:
            return None
        if status < 300:
            return response.json() if response.content else None
        detail = _detail(response)
        if status in (401, 403):
            raise NotAuthorised(status=status, detail=detail)
        if status == 409:
            raise (BusyOtherClient if _is_busy(detail) else SessionLost)(status=status, detail=detail)
        if status == 503:
            raise ServiceUnavailable(status=status, detail=detail)
        raise RequestRejected(f"{method} {path}: HTTP {status} {detail}", status, detail)

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

    async def heartbeat(self, payload: dict) -> dict:
        return await self._request("POST", "/api/device/heartbeat", json=payload) or {}

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

    async def monitor_end(self, session_id: str, generation: str) -> dict | None:
        return await self._request("POST", "/api/device/monitor/end",
                                   json={"session_id": session_id, "generation": generation})
