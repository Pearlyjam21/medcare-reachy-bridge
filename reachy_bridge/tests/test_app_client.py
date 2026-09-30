import asyncio
import json

import httpx
import pytest

from reachy_bridge.app_client import (
    AppClient, AppUnreachable, BusyOtherClient, NotAuthorised, RequestRejected, ServiceUnavailable, SessionLost)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make_client(handler, clock=None):
    return AppClient("http://app:8001", "rdv1.token", transport=httpx.MockTransport(handler),
                     clock=clock or Clock())


def run(coro):
    return asyncio.run(coro)


def test_tasks_next_sends_bearer_and_wait_and_returns_task():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"task_id": "t1", "doses": []})

    async def scenario():
        async with make_client(handler) as client:
            return await client.tasks_next(wait=25)

    assert run(scenario()) == {"task_id": "t1", "doses": []}
    assert seen["auth"] == "Bearer rdv1.token"
    assert seen["url"] == "http://app:8001/api/device/tasks/next?wait=25"


def test_204_returns_none():
    async def scenario():
        async with make_client(lambda request: httpx.Response(204)) as client:
            return await client.tasks_next(wait=0), await client.tasks_current()

    assert run(scenario()) == (None, None)


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_raise_not_authorised(status):
    async def scenario():
        async with make_client(lambda request: httpx.Response(status, json={"detail": "consent_required"})) as client:
            await client.heartbeat({"robot_reachable": True})

    with pytest.raises(NotAuthorised) as info:
        run(scenario())
    assert info.value.status == status and info.value.detail == "consent_required"


def test_409_raises_session_lost_and_busy_other_client_is_distinguished():
    async def scenario(detail):
        async with make_client(lambda request: httpx.Response(409, json={"detail": detail})) as client:
            await client.monitor_start("dose", intk_id=4, task_id="t1")

    with pytest.raises(SessionLost) as lost:
        run(scenario("Session is no longer active"))
    assert not isinstance(lost.value, BusyOtherClient)
    with pytest.raises(BusyOtherClient):
        run(scenario("busy_other_client"))
    with pytest.raises(BusyOtherClient):
        run(scenario({"code": "busy_other_client", "message": "tablet"}))


def test_503_and_other_4xx():
    async def scenario(status):
        async with make_client(lambda request: httpx.Response(status, json={"detail": "x"})) as client:
            await client.monitor_start("observe", intk_id=None, task_id="t1")

    with pytest.raises(ServiceUnavailable):
        run(scenario(503))
    with pytest.raises(RequestRejected) as info:
        run(scenario(422))
    assert info.value.status == 422


def test_network_errors_raise_app_unreachable_and_track_since():
    clock = Clock()
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"stop_all": False, "server_time": "now"})

    async def scenario():
        async with make_client(handler, clock) as client:
            with pytest.raises(AppUnreachable):
                await client.heartbeat({})
            assert client.unreachable_since == 100.0
            clock.t = 105.0
            with pytest.raises(AppUnreachable):
                await client.heartbeat({})
            assert client.unreachable_since == 100.0   # first failure time is kept
            result = await client.heartbeat({})
            assert client.unreachable_since is None
            return result

    assert run(scenario()) == {"stop_all": False, "server_time": "now"}


def test_server_errors_count_as_unreachable():
    async def scenario():
        async with make_client(lambda request: httpx.Response(502)) as client:
            with pytest.raises(AppUnreachable):
                await client.tasks_current()
            return client.unreachable_since

    assert run(scenario()) == 100.0


def test_request_bodies_for_every_route():
    bodies = []

    def handler(request):
        body = request.content.decode("latin-1")
        bodies.append((request.method, request.url.path, request.headers.get("content-type", ""), body))
        if request.url.path.endswith("/start"):
            return httpx.Response(200, json={"session_id": "s", "generation": "g"})
        return httpx.Response(200, json={"ok": True})

    async def scenario():
        async with make_client(handler) as client:
            await client.task_status("t1", "searching", None)
            await client.confirmation("t1", 5, "degraded", {"confidence": 0.3})
            await client.extra_event("t1", "e1", "uncertain", 0.31)
            await client.monitor_start("dose", intk_id=5, task_id="t1")
            await client.monitor_start("observe", intk_id=None, task_id="t1")
            await client.monitor_landmarks({"session_id": "s", "generation": "g", "frame_seq": 1})
            await client.monitor_vision("s", "g", 3, b"\xff\xd8jpeg")
            await client.monitor_end("s", "g")

    run(scenario())
    paths = [(method, path) for method, path, *_ in bodies]
    assert paths == [
        ("POST", "/api/device/tasks/t1/status"), ("POST", "/api/device/tasks/t1/confirmation"),
        ("POST", "/api/device/tasks/t1/extra-event"), ("POST", "/api/device/monitor/start"),
        ("POST", "/api/device/monitor/start"), ("POST", "/api/device/monitor/landmarks"),
        ("POST", "/api/device/monitor/vision"), ("POST", "/api/device/monitor/end")]
    assert json.loads(bodies[0][3]) == {"status": "searching", "detail": None}
    assert json.loads(bodies[1][3]) == {"intk_id": 5, "source": "degraded", "evidence": {"confidence": 0.3}}
    assert json.loads(bodies[2][3]) == {"event_id": "e1", "decision": "uncertain", "confidence": 0.31}
    assert json.loads(bodies[3][3]) == {"mode": "dose", "intk_id": 5, "task_id": "t1"}
    assert json.loads(bodies[4][3]) == {"mode": "observe", "task_id": "t1"}
    assert json.loads(bodies[7][3]) == {"session_id": "s", "generation": "g"}
    method, path, content_type, body = bodies[6]
    assert content_type.startswith("multipart/form-data")
    for field in ('name="session_id"', 'name="generation"', 'name="frame_seq"', 'name="file"; filename="frame.jpg"'):
        assert field in body
    assert "image/jpeg" in body
