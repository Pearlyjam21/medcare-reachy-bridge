"""Reachy names the medicine it asks for, and tells the server when a watch ends without a result (14 Oct 2026):
the server's spoken prompt clip fetched per medicine and position (bridge/session.py _fetch_med_clip,
bridge/clips.py play_med_prompt, bridge/app_client.py med_clip), the generic prompt when there is none, and the
watch outcomes filed for family's clip (session _file_event, app_client task_event)."""

import asyncio
import json
import wave
from pathlib import Path

import httpx

from medcare_reachy.bridge.app_client import AppClient
from medcare_reachy.bridge.clips import ClipPlayer
from medcare_reachy.bridge.tests.test_session import Harness, dose, make_task

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"


def test_the_prompt_names_each_medicine_by_its_place_in_the_slot_and_keeps_the_clip():
    h = Harness(make_task([dose(1), dose(2)]))
    h.app.med_clips[(10, 1)] = (b"first", "k1")
    h.app.med_clips[(20, 2)] = (b"second", "k2")
    h.find_patient()
    h.verify_and_prompt()
    assert h.clips.played[-1] == "med_prompt:10" and h.clips.prompt_positions == [1]
    assert h.clips.saved[(10, 1)] == (b"first", "k1")
    assert h.app.named("med_clip")[0] == ("med_clip", 10, 1, "zh-TW", None)
    h.see(identity_status="verified", recorded={"status": "taken", "detection_method": "auto"})
    h.tick()
    h.verify_and_prompt()
    assert h.clips.played[-1] == "med_prompt:20" and h.clips.prompt_positions == [1, 2]
    assert h.clips.saved[(20, 2)] == (b"second", "k2")


def test_a_kept_clip_is_checked_by_its_key_and_a_missing_or_slow_one_leaves_the_generic_prompt():
    h = Harness(make_task([dose(1)]))
    h.clips.saved[(10, 1)] = (b"old", "k1")
    h.app.med_clips[(10, 1)] = (b"old", "k1")
    h.find_patient()
    h.verify_and_prompt()
    assert h.app.named("med_clip") == [("med_clip", 10, 1, "zh-TW", "k1")]   # asked with the key: unchanged
    assert h.clips.saved[(10, 1)] == (b"old", "k1") and h.clips.played[-1] == "med_prompt:10"

    h = Harness(make_task([dose(1)]))                                         # the server has no clip
    h.find_patient()
    h.verify_and_prompt()
    assert h.clips.played[-1] == "med_prompt:10" and h.clips.saved == {}

    h = Harness(make_task([dose(1)]))                                         # the request fails
    h.app.fail["med_clip"] = [RuntimeError("down")]
    h.app.med_clip = _failing(h.app)
    h.find_patient()
    h.verify_and_prompt()
    assert h.clips.played[-1] == "med_prompt:10" and h.slot.state == "WATCHING"


def _failing(app):
    async def med_clip(med_id, position, language, etag=None):
        raise RuntimeError("down")
    return med_clip


def test_a_watch_that_sees_nothing_or_loses_the_patient_is_filed_for_familys_clip():
    h = Harness(make_task([dose(1), dose(2)]))
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="verified")
    h.tick(180.5)                                   # help
    h.tick(180.5)                                   # ... then left pending
    assert h.slot.outcomes[1] == "left_pending"
    filed = h.app.named("task_event")
    assert len(filed) == 1 and filed[0][1:3] == (1, "not_seen")
    assert filed[0][3]["said_done"] is False and filed[0][3]["seconds_watched"] >= 360
    h.verify_and_prompt()
    h.see(identity_status="verified")
    h.tick(1)
    h.see(identity_status="searching")              # the patient walks away
    h.tick(1)                                       # ... noticed now
    h.tick(91)                                      # ... and gone for AWAY_TIMEOUT
    assert h.slot.outcomes[2] == "left_pending"
    assert h.app.named("task_event")[-1][1:3] == (2, "patient_left")

    h = Harness(make_task([dose(1)]))               # a server that refuses it never stops the slot
    h.app.fail["task_event"] = [RuntimeError("500")]
    h.find_patient()
    h.verify_and_prompt()
    h.see(identity_status="verified")
    h.tick(180.5)
    h.tick(180.5)
    assert h.slot.outcomes[1] == "left_pending" and len(h.app.named("task_event")) == 1


def test_the_clip_player_prefers_the_positions_clip_then_the_medicines_then_the_generic(tmp_path):
    played = []
    robot = type("Robot", (), {"play_clip": lambda self, path, wait=True: played.append(Path(path).stem) or True})()
    player = ClipPlayer(robot, tmp_path, "zh-TW")
    (tmp_path / "zh-TW").mkdir()
    (tmp_path / "zh-TW" / "med_prompt_generic.wav").write_bytes(WAV)
    player.play_med_prompt(10, 1)
    player.save_med_clip(10, 1, WAV, "k1")
    (tmp_path / "zh-TW" / "med_10.wav").write_bytes(WAV)
    player.play_med_prompt(10, 1)
    player.play_med_prompt(10, 2)
    player.play_med_prompt(11, 1)
    assert played == ["med_prompt_generic", "med_10_1", "med_10", "med_prompt_generic"]
    assert player.med_clip_key(10, 1) == "k1" and player.med_clip_key(10, 2) is None
    assert json.loads((tmp_path / "zh-TW" / "med_10_1.json").read_text()) == {"key": "k1"}
    with wave.open(str(tmp_path / "zh-TW" / "med_10_1.wav")) as w:
        assert w.getframerate() == 16000


def test_the_app_client_fetches_a_clip_with_its_key_and_treats_everything_else_as_none():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, dict(request.url.params), request.headers.get("If-None-Match")))
        if request.url.path.endswith("/event"):
            return httpx.Response(200, json={"event": "robot:t:1:not_seen", "queued": True})
        if request.headers.get("If-None-Match") == '"k1"':
            return httpx.Response(304)
        if request.url.params.get("position") == "2":
            return httpx.Response(404)
        return httpx.Response(200, content=WAV, headers={"ETag": '"k1"', "Content-Type": "audio/wav"})

    async def run():
        async with AppClient("http://app:8001", "rdv1.token", transport=httpx.MockTransport(handler)) as client:
            fresh = await client.med_clip(10, 1, "zh-TW")
            same = await client.med_clip(10, 1, "zh-TW", etag="k1")
            none = await client.med_clip(10, 2, "zh-TW")
            event = await client.task_event("t", 1, "not_seen", {"said_done": False})
            return fresh, same, none, event

    fresh, same, none, event = asyncio.run(run())
    assert fresh == (WAV, "k1") and same == "unchanged" and none is None
    assert event == {"event": "robot:t:1:not_seen", "queued": True}
    assert seen[0][1:3] == ("/api/device/clips/med/10", {"position": "1", "language": "zh-TW"}) and seen[0][3] is None
    assert seen[1][3] == '"k1"' and seen[3][1] == "/api/device/tasks/t/event"
