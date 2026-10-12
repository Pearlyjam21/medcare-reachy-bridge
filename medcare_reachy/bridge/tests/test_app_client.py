import asyncio
import json

import httpx
import pytest

from medcare_reachy.bridge.app_client import (
    DOSE_REFUSALS, AppClient, AppUnreachable, BusyOtherClient, DoseRefused, NotAuthorised, RequestRejected,
    ServiceUnavailable, SessionLost)


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


TOO_SOON = {"detail": "dose_too_soon", "intk_id": 4, "med_name": "allegra",
            "scheduled_time": "2026-10-03T08:00:00+08:00", "last_taken_at": "2026-10-03T00:05:00+08:00",
            "next_allowed_at": "2026-10-03T06:05:00+08:00", "reply": "這個藥您凌晨12點05分已經吃過了，請先不要再吃。",
            "speech_text": "这个药您凌晨12点05分已经吃过了，请先不要再吃。"}


@pytest.mark.parametrize("call", ["start", "confirmation"])
def test_overdose_refusal_keeps_the_whole_409_body(call):
    async def scenario():
        async with make_client(lambda request: httpx.Response(409, json=TOO_SOON)) as client:
            if call == "start":
                await client.monitor_start("dose", intk_id=4, task_id="t1")
            else:
                await client.confirmation("t1", 4, "patient_claim", {"said_done": True})

    with pytest.raises(DoseRefused) as info:
        run(scenario())
    refused = info.value
    assert isinstance(refused, SessionLost) and not isinstance(refused, BusyOtherClient)
    assert refused.status == 409 and refused.detail == "dose_too_soon" and refused.body == TOO_SOON
    assert refused.intk_id == 4 and refused.speech_text == TOO_SOON["speech_text"]


@pytest.mark.parametrize("code", sorted(DOSE_REFUSALS))
def test_every_refusal_code_and_a_nested_body_raise_dose_refused(code):
    async def scenario(body):
        async with make_client(lambda request: httpx.Response(409, json=body)) as client:
            await client.monitor_start("dose", intk_id=4, task_id="t1")

    with pytest.raises(DoseRefused) as flat:
        run(scenario({**TOO_SOON, "detail": code}))
    assert flat.value.detail == code and flat.value.body["last_taken_at"] == TOO_SOON["last_taken_at"]
    with pytest.raises(DoseRefused) as nested:   # an HTTPException with a dict detail
        run(scenario({"detail": {**TOO_SOON, "detail": code}}))
    assert nested.value.detail == code and nested.value.speech_text == TOO_SOON["speech_text"]


def test_refusal_without_a_sentence_and_other_409s():
    async def scenario(body):
        async with make_client(lambda request: httpx.Response(409, json=body)) as client:
            await client.monitor_start("dose", intk_id=4, task_id="t1")

    older = {"detail": "dose_not_due_yet", "intk_id": 4, "scheduled_time": "2026-10-03T20:00:00+08:00",
             "due_from": "2026-10-03T18:00:00+08:00"}
    with pytest.raises(DoseRefused) as info:
        run(scenario(older))
    assert info.value.speech_text is None and info.value.body == older
    with pytest.raises(DoseRefused) as blank:
        run(scenario({**TOO_SOON, "speech_text": "  ", "intk_id": "4"}))
    assert blank.value.speech_text is None and blank.value.intk_id is None
    with pytest.raises(SessionLost) as plain:
        run(scenario({"detail": "Dose is unavailable or does not belong to this account"}))
    assert not isinstance(plain.value, DoseRefused)
    assert plain.value.body == {"detail": "Dose is unavailable or does not belong to this account"}


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


def test_conversation_turns_carry_timings_and_reachys_timings_go_to_its_turn():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, json.loads(request.content),
                     request.extensions["timeout"]["read"]))
        if request.url.path.endswith("/turn"):
            return httpx.Response(200, json={"reply": "好", "speech_text": "好", "end": False, "risk": False,
                                             "reply_turn_id": 7, "server_ms": 900})
        return httpx.Response(200, json={"ok": True})

    async def scenario():
        async with make_client(handler) as client:
            answer = await client.conversation_turn("c1", "你好", {"handover_ms": 350, "listen_mode": "chat"})
            await client.conversation_turn("c1", "再見")
            ok = await client.conversation_turn_metrics("c1", 7, {"round_trip_ms": 1800, "tts_chunks": 2})
            return answer, ok

    answer, ok = run(scenario())
    assert answer["reply_turn_id"] == 7 and ok == {"ok": True}
    assert seen == [
        ("POST", "/api/device/conversations/c1/turn", {"text": "你好", "metrics": {"handover_ms": 350,
                                                                                 "listen_mode": "chat"}}, 60),
        ("POST", "/api/device/conversations/c1/turn", {"text": "再見", "metrics": None}, 60),
        ("POST", "/api/device/conversations/c1/turns/7/metrics", {"metrics": {"round_trip_ms": 1800,
                                                                             "tts_chunks": 2}}, 3.0)]


def test_refused_timings_raise_like_any_other_request():
    async def scenario():
        async with make_client(lambda request: httpx.Response(404, json={"detail": "turn_not_found"})) as client:
            await client.conversation_turn_metrics("c1", 99, {"round_trip_ms": 1})

    with pytest.raises(RequestRejected) as info:
        run(scenario())
    assert info.value.status == 404


@pytest.mark.parametrize("failure", ["timeout", "server_error"])
def test_failed_timings_raise_but_never_start_an_outage(failure):
    def handler(request):
        if request.url.path.endswith("/metrics"):
            if failure == "timeout":
                raise httpx.ReadTimeout("slow", request=request)
            return httpx.Response(502)
        raise httpx.ConnectError("refused", request=request)

    async def scenario():
        async with make_client(handler) as client:
            with pytest.raises(AppUnreachable):
                await client.conversation_turn_metrics("c1", 7, {"round_trip_ms": 1800})
            assert client.unreachable_since is None      # timings only: the slot must not fail closed on them
            with pytest.raises(AppUnreachable):
                await client.conversation_turn("c1", "你好")
            return client.unreachable_since              # the conversation itself still counts

    assert run(scenario()) == 100.0


def test_heartbeat_sends_the_camera_only_when_given():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"stop_all": False})

    camera = {"state": "stalled", "fps": 0.0, "since": 1760000000.0,
              "media": {"available": False, "released": True, "no_media": False}}

    async def scenario():
        async with make_client(handler) as client:
            await client.heartbeat({"robot_reachable": True})
            await client.heartbeat({"robot_reachable": True}, camera=camera)

    run(scenario())
    assert bodies == [{"robot_reachable": True}, {"robot_reachable": True, "camera": camera}]


@pytest.mark.parametrize("base_url, expected", [
    ("http://app:8001", "ws://app:8001/api/device/conversations/live"),
    ("http://192.168.137.1:8001/", "ws://192.168.137.1:8001/api/device/conversations/live"),
    ("HTTPS://medcare.example:8443", "wss://medcare.example:8443/api/device/conversations/live"),
])
def test_the_live_chat_websocket_is_on_the_same_server(base_url, expected):
    client = AppClient(base_url, "rdv1.token", transport=httpx.MockTransport(lambda request: httpx.Response(204)))
    assert client.live_url() == expected
    run(client.aclose())


def test_the_live_chat_websocket_carries_the_robot_key():
    client = make_client(lambda request: httpx.Response(204))
    assert client.auth_header() == {"Authorization": "Bearer rdv1.token"}
    run(client.aclose())
