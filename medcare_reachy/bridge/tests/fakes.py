"""Hardware-free, network-free fakes for the slot state machine."""

import asyncio
import copy
import json
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np

from medcare_reachy.bridge.app_client import DoseRefused, SessionLost
from medcare_reachy.bridge.session import SlotSession

BASE_WALL = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def wall(self):
        return BASE_WALL + timedelta(seconds=self.t - 1000.0)


class FakeRobot:
    def __init__(self, reachable=True):
        self.reachable = reachable
        self.calls = []
        self.gesture_error = None   # raised by gesture() when set
        # The live chat's audio (bridge/live.py):
        self.mic = deque()          # (samples, rate) chunks get_audio() hands out, oldest first
        self.live_audio = True      # what start_live_audio() answers
        self.push_error = None      # raised by push_audio() when set
        self.clear_error = None     # raised by clear_live_audio() when set
        self.blocks = 0             # blocks push_audio() got, silent ones too
        self.pushed = []            # the blocks with sound (float32 copies) ...
        self.pushed_at = []         # ... and the value of `blocks` once each was pushed
        self.cleared_at = None      # `blocks` when clear_live_audio() was last called

    def get_audio(self):
        return self.mic.popleft() if self.mic else None

    def start_live_audio(self):
        self.calls.append("start_live_audio")
        return self.live_audio

    def push_audio(self, samples):
        if self.push_error is not None:
            raise self.push_error
        self.blocks += 1
        if np.any(samples):
            self.pushed.append(np.array(samples, dtype=np.float32))
            self.pushed_at.append(self.blocks)
        return True

    def clear_live_audio(self):
        self.calls.append("clear_live_audio")
        self.cleared_at = self.blocks
        if self.clear_error is not None:
            raise self.clear_error

    def stop_live_audio(self):
        self.calls.append("stop_live_audio")

    def heard(self):
        """Every sample with sound the speaker got, as PCM16 values."""
        if not self.pushed:
            return np.zeros(0, np.int32)
        return np.round(np.concatenate(self.pushed) * 32768).astype(np.int32)

    def gesture(self, mode):
        self.calls.append(f"gesture:{mode}")
        if self.gesture_error is not None:
            raise self.gesture_error

    def is_reachable(self):
        self.calls.append("is_reachable")
        return self.reachable

    def restore_media(self):
        self.calls.append("restore_media")
        return True

    def wake(self):
        self.calls.append("wake")

    def sleep(self):
        self.calls.append("sleep")

    def hold_head(self):
        self.calls.append("hold_head")

    def look_at(self, u, v, duration=0.35):
        self.calls.append(f"look_at:{u:.2f},{v:.2f}")

    def look_around(self, step):
        self.calls.append(f"look_around:{step}")


class FakeClips:
    ACK_SECONDS = 0.4

    def __init__(self):
        self.played = []
        self.missing = set()

    def play(self, clip_id):
        self.played.append(clip_id)
        return True

    def start(self, clip_id):
        """Handed to the speaker without waiting: its length, or None when the clip is missing."""
        if clip_id in self.missing:
            return None
        self.played.append(clip_id)
        return self.ACK_SECONDS

    def play_med_prompt(self, med_id):
        self.played.append(f"med_prompt:{med_id}")
        return True


class FakeVoice:
    """DoneListener stand-in: tests call say() to make the patient say they finished."""

    def __init__(self, clock, available=True):
        self.clock, self.available = clock, available
        self.active = False
        self.mode = "done"
        self.activity = []
        self.holds = 0
        self.max_holds = 0
        self.heard = None
        self.utterances = []
        self.spoken = []
        self.pause = None       # chat: when the VAD released what the patient said (heard_pause)
        self.speaking = False   # chat: hears_speech()
        self.acks = []          # chat: (when, seconds) of each note_ack
        self.suspended = False  # a live chat reads the microphone meanwhile (suspend / resume)
        self.handovers = []     # "suspend" / "resume", in order

    def set_active(self, active, mode="done"):
        if active != self.active:
            self.activity.append(active)
        self.active, self.mode = active, mode

    def hold(self):
        self.holds += 1
        self.max_holds = max(self.max_holds, self.holds)

    def release(self):
        self.holds -= 1

    def suspend(self):
        self.suspended = True
        self.handovers.append("suspend")

    def resume(self):
        self.suspended = False
        self.handovers.append("resume")

    def say(self, word="吃完"):
        assert self.active and self.holds == 0, "the listener is off"
        self.heard = (self.clock.monotonic(), word)

    def speak(self, text, metrics=None, ended=None):
        """Chat mode: the patient says something (`metrics`: what the listener measured while hearing it); the VAD
        released it at `ended` (default now), and it is decoded at once."""
        assert self.active and self.mode == "chat" and self.holds == 0, "not listening for a conversation"
        assert not self.suspended, "the live chat has the microphone"
        self.pauses(ended)
        self.utterances.append((text, metrics or {"handover_ms": 400, "segments": 1, "listen_mode": "chat"}))

    def pauses(self, ended=None):
        """Chat mode: the VAD released what the patient said; SenseVoice is still decoding it."""
        assert self.active and self.mode == "chat" and self.holds == 0, "not listening for a conversation"
        assert not self.suspended, "the live chat has the microphone"
        if self.pause is None:
            self.pause = self.clock.monotonic() if ended is None else ended

    def heard_pause(self):
        return self.pause

    def hears_speech(self):
        return self.speaking

    def note_ack(self, seconds):
        self.acks.append((self.clock.monotonic(), seconds))

    def take_utterance_with_metrics(self):
        if not self.utterances:
            return None
        heard = self.utterances.pop(0)
        if not self.utterances:
            self.pause = None
        return heard

    def take_utterance(self):
        heard = self.take_utterance_with_metrics()
        return heard[0] if heard else None

    def note_spoken(self, text):
        self.spoken.append(text)

    def heard_since(self, since):
        return self.heard[1] if self.heard and self.heard[0] >= since else None


class FakeSpeaker:
    STATS = {"tts_first_audio_ms": 1500, "tts_total_ms": 4200, "tts_chunks": 2, "tts_synth_ms": 2900}

    SYNTH_LEVEL = 0.25   # synth(): every sample, 10 ms per character, at 16 kHz
    SYNTH_RATE = 16000

    def __init__(self, available=True, robot=None):
        self.available = available
        self.said = []
        self.synthesised = []   # synth()'s texts (the live chat)
        self.robot = robot   # gets "say:<text>" in its calls, to check what moved when
        self.error = None    # raised by say() and synth() when set (e.g. synthesis failed)

    def synth(self, text):
        if self.error is not None:
            raise self.error
        self.synthesised.append(text)
        return np.full(len(text) * self.SYNTH_RATE // 100, self.SYNTH_LEVEL, np.float32), self.SYNTH_RATE

    def say(self, text, stats=None, on_audio=None):
        if self.error is not None:
            raise self.error
        if on_audio is not None:
            on_audio()   # the first chunk is about to play
        self.said.append(text)
        if self.robot is not None:
            self.robot.calls.append(f"say:{text}")
        if stats is not None:
            stats.update(self.STATS)
        return True


class FakeCamera:
    """CameraWatchdog stand-in: tests set `state`."""

    def __init__(self):
        self.state = "idle"

    def snapshot(self):
        return {"state": self.state, "fps": 0.0, "since": None, "media": None}


class FakeStream:
    def __init__(self):
        self.attached = None
        self.latest = None
        self.latest_at = None   # when `latest` came (MonitorStream's; Harness.see() sets it)
        self.errors = []
        self.history = []
        self.camera = FakeCamera()
        self.max_fps = None     # set_max_fps(): the companion session's camera rate, None for the stream's own
        self.frame_seq = 0      # the attached session's last streamed frame (MonitorStream's; tests set it)

    def set_max_fps(self, fps):
        self.max_fps = fps

    def attach(self, monitor):
        self.attached = monitor
        self.latest = self.latest_at = None
        self.frame_seq = 0
        self.camera.state = "ok"   # as the watchdog's attach(): a fresh try
        self.history.append(("attach", monitor["session_id"]))

    def detach(self):
        if self.attached:
            self.history.append(("detach", self.attached["session_id"]))
        self.attached = None
        self.latest = self.latest_at = None
        if self.camera.state != "unavailable":   # as the watchdog's detach(): "unavailable" stays
            self.camera.state = "idle"

    def take_error(self):
        return self.errors.pop(0) if self.errors else None

    def emotion_compute(self):
        return None


class FakeApp:
    """Records every call; `fail` maps a method name to a list of exceptions raised on successive calls."""

    def __init__(self, task, clock):
        self.task = copy.deepcopy(task)
        self.clock = clock
        self.current_task = copy.deepcopy(task)
        self.calls = []
        self.fail = {}
        self.unreachable_since = None
        self.sessions = []
        self.companion_starts = []   # conversation_start's `companion`, per call
        self.monitor_companion = []   # monitor_start's `companion`, per call

    def _maybe_fail(self, name):
        queue = self.fail.get(name)
        if queue:
            exc = queue.pop(0)
            if exc is not None:
                raise exc

    def dose(self, intk_id):
        return next(d for d in self.current_task["doses"] if d["intk_id"] == intk_id)

    async def tasks_current(self):
        self.calls.append(("tasks_current",))
        self._maybe_fail("tasks_current")
        return copy.deepcopy(self.current_task)

    async def task_status(self, task_id, status, detail=None):
        self.calls.append(("task_status", status, detail))
        self._maybe_fail("task_status")
        self.current_task["status"] = status
        return {"status": status}

    async def confirmation(self, task_id, intk_id, source, evidence):
        self.calls.append(("confirmation", intk_id, source, evidence))
        self._maybe_fail("confirmation")
        self.dose(intk_id)["intake_stats"] = "pending_confirmation"
        return {"confirmation_id": "c1"}

    async def extra_event(self, task_id, event_id, decision, confidence):
        self.calls.append(("extra_event", event_id, decision, confidence))
        self._maybe_fail("extra_event")
        return {"ok": True}

    async def monitor_start(self, mode, intk_id, task_id, companion=False):
        self.calls.append(("monitor_start", mode, intk_id))
        self.monitor_companion.append(companion)
        self._maybe_fail("monitor_start")
        n = len(self.sessions) + 1
        session = {"session_id": f"s{n}", "generation": f"g{n}", "mode": mode, "intk_id": intk_id,
                   "identity_status": "searching"}
        self.sessions.append(session)
        return dict(session)

    async def conversation_start(self, task_id, language, companion=False):
        self.calls.append(("conversation_start", task_id, language))
        self.companion_starts.append(companion)
        self._maybe_fail("conversation_start")
        self.turn_ids = 1
        return {"conversation_id": "c1", "reply": "今天感覺怎麼樣？", "speech_text": "今天感觉怎么样？", "end": False,
                "reply_turn_id": 1}

    async def conversation_turn(self, conversation_id, text, metrics=None):
        self.calls.append(("conversation_turn", conversation_id, text, metrics))
        self._maybe_fail("conversation_turn")
        self.clock.t += getattr(self, "turn_seconds", 0.0)   # the server thinking
        self.turn_ids += 2   # the patient's turn, then Reachy's reply
        answer = self.replies.pop(0) if getattr(self, "replies", None) else {}
        return {"reply": "真好", "speech_text": "真好", "end": False, "risk": False, "reply_turn_id": self.turn_ids,
                "server_ms": 1200, **answer}

    async def conversation_turn_metrics(self, conversation_id, turn_id, metrics):
        self.calls.append(("conversation_turn_metrics", conversation_id, turn_id, metrics))
        self._maybe_fail("conversation_turn_metrics")
        return {"ok": True}

    async def conversation_end(self, conversation_id, reason):
        self.calls.append(("conversation_end", conversation_id, reason))
        self._maybe_fail("conversation_end")
        return {"ended": True}

    async def monitor_end(self, session_id, generation):
        self.calls.append(("monitor_end", session_id))
        self._maybe_fail("monitor_end")
        return {"success": True}

    def named(self, name):
        return [call for call in self.calls if call[0] == name]

    def live_url(self):
        return "ws://app:8001/api/device/conversations/live"

    def auth_header(self):
        return {"Authorization": "Bearer rdv1.token"}


class FakeSocket:
    """The server's end of the live chat's WebSocket, in bridge/live.py's socket shape: tests `serve()` what the
    server sends (a dict goes as JSON text, bytes as a binary frame), read what the robot sent, and `drop()` it.
    `stamp()`, when given, is read at every send (e.g. the robot's block count), into `stamps`."""

    def __init__(self, stamp=None):
        self.inbox = asyncio.Queue()
        self.sent = []
        self.stamps = []
        self.stamp = stamp
        self.closed = False
        self.close_code = None    # the code the server closed with
        self.closed_with = None   # the code the robot closed with

    async def send(self, data):
        if self.closed:
            return False
        self.sent.append(data)
        if self.stamp is not None:
            self.stamps.append(self.stamp())
        return True

    async def receive(self):
        if self.closed:
            return None
        message = await self.inbox.get()
        if message is None:
            self.closed = True
        return message

    async def close(self, code=1000):
        if not self.closed:
            self.closed, self.closed_with = True, code
            self.inbox.put_nowait(None)

    def serve(self, message):
        self.inbox.put_nowait(json.dumps(message) if isinstance(message, dict) else message)

    def drop(self, code=1006):
        self.close_code = code
        self.inbox.put_nowait(None)

    def json_sent(self):
        return [json.loads(data) for data in self.sent if isinstance(data, str)]

    def frames_sent(self):
        return [data for data in self.sent if isinstance(data, bytes)]

    def stamp_of(self, message):
        """`stamp()` when the robot first sent this JSON message."""
        return self.stamps[self.sent.index(json.dumps(message))]


def dose(intk_id, med_id=None, stats="pending", supported=True):
    return {"intk_id": intk_id, "med_id": med_id or intk_id * 10, "med_name": f"Med{intk_id}",
            "pill_description": None, "dose_form": "solid_oral" if supported else "liquid",
            "units_per_dose": 1, "intake_stats": stats, "supported": supported}


def make_task(doses, *, status="leased", reason="upcoming", auto_record=True, expires_in=3600, microphone=None,
              checkin=False):
    if microphone is None:   # as the server: check-in consent includes the microphone (CHECKIN_SCOPES)
        microphone = checkin
    return {"task_id": "task-1", "slot_time": "2026-10-01T00:00:00+00:00", "reason": reason, "attempt": 1,
            "status": status, "expires_at": (BASE_WALL + timedelta(seconds=expires_in)).isoformat(),
            "patient_name": "Amy", "auto_record": auto_record, "microphone": microphone, "checkin": checkin,
            "doses": doses}


class Harness:
    def __init__(self, task, robot=None, voice=False, speaker=False, **session):
        self.clock = FakeClock()
        self.app = FakeApp(task, self.clock)
        self.robot = robot or FakeRobot()
        self.clips = FakeClips()
        self.stream = FakeStream()
        self.voice = FakeVoice(self.clock) if voice else None
        self.speaker = FakeSpeaker(robot=self.robot) if speaker else None
        self.slot = SlotSession(copy.deepcopy(task), app=self.app, robot=self.robot, clips=self.clips,
                                stream=self.stream, voice=self.voice, speaker=self.speaker, clock=self.clock,
                                **session)

    def tick(self, advance=0.0):
        self.clock.t += advance
        asyncio.run(self.slot.tick())

    def see(self, **state):
        """Pretend the attached monitor session answered with this public() state."""
        assert self.stream.attached, "no monitor session attached"
        self.stream.latest = {"session_id": self.stream.attached["session_id"], **state}
        self.stream.latest_at = self.clock.t   # the answer came now

    def run_until(self, predicate, step=0.5, limit=2000):
        for _ in range(limit):
            if predicate():
                return
            self.tick(step)
        raise AssertionError(f"condition not reached; state={self.slot.state} done={self.slot.done}")

    def to_state(self, state, **kw):
        self.run_until(lambda: self.slot.state == state, **kw)

    def find_patient(self):
        self.to_state("SEARCHING")
        self.tick()
        self.see(identity_status="verified")
        self.tick()

    def verify_and_prompt(self):
        """In MED_PROMPT: the dose session verifies the patient, the prompt plays."""
        self.tick()   # starts the dose session
        self.see(identity_status="verified")
        self.tick()
        assert self.slot.state == "WATCHING"

    def statuses(self):
        return [call[1] for call in self.app.named("task_status")]


def lost(detail="Session is no longer active"):
    return SessionLost(status=409, detail=detail)


TOO_SOON_SPEECH = "这个药您凌晨12点05分已经吃过了，请先不要再吃。"


def refused(detail="dose_too_soon", intk_id=None, speech_text=TOO_SOON_SPEECH, **fields):
    """An overdose-protection 409, as AppClient raises it (`speech_text=None`: a server that sends no sentence)."""
    body = {"detail": detail, "intk_id": intk_id, "med_name": "Med", "scheduled_time": "2026-10-03T08:00:00+08:00",
            "reply": "這個藥您凌晨12點05分已經吃過了，請先不要再吃。", "speech_text": speech_text, **fields}
    return DoseRefused(status=409, detail=detail, body={key: value for key, value in body.items() if value is not None})
