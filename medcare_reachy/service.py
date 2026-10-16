"""Runs the bridge (task polling, heartbeat, camera loop, slot state machine) on its own thread and event loop.

The Reachy Mini app runtime owns the robot connection and calls run() on a thread; the bridge is asyncio, so
it gets a private loop here. Saving new settings restarts it.
"""

import asyncio
import logging
import threading
import time
from pathlib import Path

from medcare_reachy import models
from medcare_reachy.settings_store import SettingsStore

log = logging.getLogger("medcare_reachy")


class ServiceBusy(RuntimeError):
    """A request the bridge cannot take now (the body-language test while a session runs)."""
CLIPS_DIR = Path(__file__).resolve().parent / "bridge" / "clips"
# status(): "running" with no reminder in progress says why once the server has not answered for this long (a
# heartbeat every 10 s, a task poll every ~30 s: one missed answer of either is not an outage).
SERVER_LOST_SECONDS = 30.0
NO_SERVER = {"url": None, "reachable": None, "last_ok_seconds_ago": None, "unreachable_for_seconds": None,
             "last_error": None}   # status()["server"] while no runner is up
STEP_TEXT = {
    "WAKE": "Waking up for a medication reminder.",
    "ANNOUNCE": "Announcing the medication reminder.",
    "SEARCHING": "Looking for the patient.",
    "MED_PROMPT": "Asking the patient to take a medicine.",
    "WATCHING": "Watching the patient take a medicine.",
    "POST_SLOT_OBSERVE": "Watching for a few more minutes after the last medicine.",
    "CHECKIN": "Chatting with the patient.",
    "WIND_DOWN": "Finishing the reminder.",
}


def face_follower(robot, settings: dict):
    """The head that keeps the verified patient's face centred while a dose is watched (bridge/follow.py), or None
    when the settings page's "Follow the patient's face" is off. Recentring goes back to the neutral pose."""
    if not settings.get("follow_face", True):
        return None
    from medcare_reachy.bridge.follow import FaceFollower

    return FaceFollower(robot.look_at, robot.hold_head)


async def default_bridge(reachy_mini, settings: dict, models_dir: Path, on_runner) -> None:
    # Deferred so the settings page works even if a dependency is still installing or broken.
    from medcare_reachy.bridge.app_client import AppClient
    from medcare_reachy.bridge.clips import ClipPlayer
    from medcare_reachy.bridge.media import ReachyRobot
    from medcare_reachy.bridge.runner import MonitorStream, Runner
    from medcare_reachy.bridge.speech import Speaker
    from medcare_reachy.bridge.voice import DoneListener

    from medcare_reachy.bridge.multimodal import ComputeGovernor, load_multimodal

    engine = emotion = governor = None
    if not settings.get("vision_on_server", True):
        # On-robot landmarks (~3 fps on a Pi 4): sessions stay below the server's recording rate.
        from medcare_reachy.bridge.vision import VisionEngine   # ONNX Runtime; no MediaPipe, no AES needed

        await asyncio.to_thread(models.verify_models, models_dir)
        engine = await asyncio.to_thread(VisionEngine, models_dir)
        log.info("[STARTUP] vision: landmarks on the robot (ONNX models in %s verified)", models_dir)
    else:
        log.info("[STARTUP] vision: landmarks on the server (the camera's frames are streamed to it)")
    if settings.get("multimodal_emotion", True):
        # Face from the bundled model, voice and words from the Hub (the machine's Hugging Face login), fused on
        # the robot while the governor keeps it there; otherwise the laptop scores the streamed frames.
        try:
            emotion = await asyncio.to_thread(load_multimodal, models_dir, settings["language"])
            log.info("[STARTUP] emotion: multimodal on, computed %s (face%s%s)", settings.get("emotion_compute", "auto"),
                     ", voice" if emotion.speech is not None else "", ", words" if emotion.text is not None else "")
        except Exception as exc:   # the dose flow doesn't depend on it
            log.exception("multimodal emotion is off: its models could not be loaded")
            log.warning("[STARTUP] emotion: multimodal off (%s: %s)", type(exc).__name__, exc)
        governor = ComputeGovernor(settings.get("emotion_compute", "auto"))
    elif engine is not None:
        from medcare_reachy.bridge.emotion import EmotionEngine   # face only, as 0.3.0 did

        emotion = await asyncio.to_thread(EmotionEngine, models_dir / models.EMOTION_MODEL)
        governor = ComputeGovernor("robot")
        log.info("[STARTUP] emotion: face only, on the robot (the Sense emotion setting is off)")
    else:
        log.info("[STARTUP] emotion: off (the Sense emotion setting is off); the server scores the streamed frames")
    robot = ReachyRobot.attach(reachy_mini)
    log.info("[STARTUP] robot: connection handed over by the app runtime; media %s",
             "released by the SDK: camera, speaker and microphone are off until the daemon gives them back"
             if robot.media_released() else "held: camera, speaker and microphone")
    clips = ClipPlayer(robot, CLIPS_DIR, settings["language"])
    clips.audit()
    # A check-in turn Reachy answered with its 「嗯」 is followed, once handed over, by a thinking phrase.
    hear = getattr(emotion, "hear", None)
    voice = DoneListener(robot, settings["language"], think_aloud=clips.think_aloud, on_segment=hear)
    voice.start()
    speaker = Speaker(robot)
    speaker.preload()
    log.info("[STARTUP] voice: check-in voice (Matcha-TTS) %s; 'I finished' listener (Whisper + silero VAD) %s",
             "on" if speaker.available else "off (model files missing; see the warning above)",
             "on" if voice.available else "off (model files missing; see the warning above)")
    try:
        async with AppClient(settings["app_url"], settings["device_token"]) as app:
            runner = Runner(app=app, robot=robot, clips=clips, voice=voice, speaker=speaker,
                            language=settings["language"],
                            stream=MonitorStream(app, robot, engine, fps=float(settings["capture_fps"]),
                                                 emotion=emotion, governor=governor),
                            ack=settings.get("checkin_ack", True), gestures=settings.get("checkin_gestures", True),
                            mood=emotion if hear is not None else None, follow=face_follower(robot, settings),
                            server_url=settings["app_url"])
            on_runner(runner)
            await runner.run()
    finally:
        voice.stop()
        robot.close()   # stops any gesture; the app runtime keeps the connection
        if engine is not None:
            engine.close()


class BridgeService:
    def __init__(self, reachy_mini, store: SettingsStore, models_dir: Path = models.DEFAULT_DIR,
                 bridge=default_bridge):
        self.reachy_mini = reachy_mini
        self.store = store
        self.models_dir = models_dir
        self.bridge = bridge
        self._lock = threading.Lock()
        # The settings page's requests run on a thread pool, so two restarts can overlap (a double click, or Save
        # right after Restart). Interleaved, they once left two bridges driving the same robot.
        self._restart_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_requested = False   # stop() was called for the tracked thread, maybe before it had a runner
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner = None
        self._state = "connecting_robot" if reachy_mini is None else "stopped"
        self._detail = "Connecting to the robot…" if reachy_mini is None else ""
        self._since = time.time()

    # ── lifecycle ──
    def attach(self, reachy_mini) -> None:
        """The runtime hands over the robot connection after the settings page is already up."""
        with self._lock:
            self.reachy_mini = reachy_mini
            if self._state == "connecting_robot":
                self._set("stopped", "")

    def start(self) -> None:
        with self._lock:
            if self.reachy_mini is None:
                return   # settings are saved; run() starts the bridge once the robot is attached
            if self._thread and self._thread.is_alive():
                # Never two bridges on one robot: the tracked one is still running, or still stopping.
                log.warning("a bridge thread is still running; not starting a second one")
                return
            settings = self.store.load()
            log.info("[STARTUP] settings: %s (server %s, robot key %s)", self.store.path, settings["app_url"] or "-",
                     "yes" if settings["device_token"] else "no")
            if not SettingsStore.configured(settings):
                self._set("not_configured", "Enter the server address and robot key on this page.")
                return
            self._set("starting", "Connecting to the server…")
            self._stop_requested = False
            self._thread = threading.Thread(target=self._thread_main, args=(settings,),
                                            name="medcare-bridge", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 15.0) -> None:
        with self._lock:
            thread, loop, runner = self._thread, self._loop, self._runner
            self._stop_requested = True   # a bridge still starting has no runner to signal yet: _on_runner stops it
        if loop and runner:
            loop.call_soon_threadsafe(runner.request_shutdown)
        if thread:
            thread.join(timeout)
        with self._lock:
            if thread is not None and thread.is_alive():
                # Keep tracking it: a forgotten live bridge once kept running beside its replacement. start()
                # refuses until it has ended, and its own exit then sets the state.
                log.warning("bridge thread did not stop within %.0f s; a new one starts only after it ends", timeout)
                self._set("stopping", "The previous connection is still closing. Press Restart in a moment.")
                return
            if self._thread is not thread:
                return   # start() ran meanwhile (the app starting up): the new bridge is the tracked one
            self._thread = None
            if self._state not in ("error", "not_configured", "connecting_robot"):
                self._set("stopped", "")

    def restart(self) -> None:
        with self._restart_lock:   # one restart at a time (see __init__)
            self.stop()
            self.start()

    def _current(self) -> bool:
        """Whether the calling bridge thread is the service's one. stop() keeps tracking a bridge thread until it
        has ended, so this guards against a thread the service no longer tracks."""
        return self._thread is threading.current_thread()

    def _thread_main(self, settings: dict) -> None:
        loop = asyncio.new_event_loop()
        with self._lock:
            self._loop = loop
        try:
            loop.run_until_complete(self.bridge(self.reachy_mini, settings, self.models_dir, self._on_runner))
            with self._lock:
                if self._current():
                    self._set("stopped", "")
        except models.ModelError as exc:
            log.error("%s", exc)
            with self._lock:
                if self._current():
                    self._set("error", str(exc))
        except Exception as exc:   # surfaced on the settings page; the app itself keeps running
            log.exception("bridge stopped with an error")
            with self._lock:
                if self._current():
                    self._set("error", f"{type(exc).__name__}: {exc}")
        finally:
            loop.close()
            with self._lock:
                if self._current():
                    self._loop = None
                    self._runner = None

    def _on_runner(self, runner) -> None:
        with self._lock:
            if not self._current():
                return
            self._runner = runner
            stop_now = self._stop_requested
            if not stop_now:
                self._set("running", "Waiting for a medication reminder from the server.")
        if stop_now:
            runner.request_shutdown()   # stop() came while this bridge was still starting: it had no runner then

    def _set(self, state: str, detail: str) -> None:
        self._state, self._detail, self._since = state, detail, time.time()

    # ── a test of the body language on the real robot (web.py POST /api/express) ──
    def express(self, affect: str | None, move: str | None, seconds: float = 6.0) -> dict:
        """Show a posture and one recorded emotion move on the idle robot, the way the live chat's `express` does
        (bridge/live.py), on a thread of its own: wake, the neutral pose, the posture with the speaking gesture, the
        move, `seconds` of the posture, then back to neutral and to sleep. ServiceBusy while a session runs."""
        with self._lock:
            runner = self._runner
        if runner is None or runner.robot is None:
            raise ServiceBusy("the bridge is not running")
        if runner.slot is not None:
            raise ServiceBusy("a reminder or a chat is running; try again when the robot is idle")
        if affect not in (None, "", "concern", "joy"):
            raise ServiceBusy(f"unknown affect {affect!r}: concern, joy or null")
        hold = max(1.0, min(20.0, float(seconds)))
        robot = runner.robot

        def show() -> None:
            log.info("[EXPRESS] test: affect=%s move=%s for %.0f s", affect or "-", move or "-", hold)
            try:
                robot.wake()
                robot.hold_head()
                robot.express(affect or None)
                robot.gesture("speak")
                if move:
                    time.sleep(1.5)
                    outcome = robot.play_move("emotion", move, lambda: False)
                    log.info("[EXPRESS] test: move %s %s", move, outcome)
                    robot.gesture("speak")
                time.sleep(hold)
                robot.gesture(None)
                robot.express(None)
                time.sleep(2.5)
                robot.sleep()
            except Exception:
                log.exception("[EXPRESS] test failed")

        threading.Thread(target=show, name="medcare-express-test", daemon=True).start()
        return {"status": "started", "affect": affect or None, "move": move, "seconds": hold}

    # ── status for the settings page ──
    def status(self) -> dict:
        with self._lock:
            runner = self._runner
            status = {"state": self._state, "detail": self._detail, "since": self._since}
        if runner is not None:
            slot = runner.slot
            server = runner.server_status()
            status.update({
                "robot_reachable": runner.robot_reachable,
                "camera_fps": round(runner.stream.camera_fps(), 1),
                "landmark_fps": round(runner.stream.landmark_fps(), 1),
                "vision_fps": round(runner.stream.vision_fps(), 1),
                "emotion": getattr(runner.stream, "emotion_status", lambda: None)(),
                "missing_clips": runner.clips.missing_count,
                "slot_state": getattr(slot, "state", None) if slot else None,
                "task_id": slot.task.get("task_id") if slot else None,
                "server": server,   # robot_reachable is the robot daemon; this is the MedAiCarePlus server
            })
            lost = server["unreachable_for_seconds"]
            if slot is not None:
                status["detail"] = STEP_TEXT.get(getattr(slot, "state", ""), "Working on a medication reminder.")
            elif status["state"] == "running" and lost is not None and lost >= SERVER_LOST_SECONDS:
                # Computed here, never stored: the waiting line is back as soon as the server answers again.
                status["detail"] = (f"Cannot reach the server at {server['url']} ({server['last_error']}) for "
                                    f"{lost:.0f} s.")
        else:
            status["server"] = dict(NO_SERVER)
        return status
