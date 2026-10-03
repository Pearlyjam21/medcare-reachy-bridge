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
CLIPS_DIR = Path(__file__).resolve().parent / "bridge" / "clips"
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


async def default_bridge(reachy_mini, settings: dict, models_dir: Path, on_runner) -> None:
    # Deferred so the settings page works even if a dependency is still installing or broken.
    from medcare_reachy.bridge.app_client import AppClient
    from medcare_reachy.bridge.clips import ClipPlayer
    from medcare_reachy.bridge.media import ReachyRobot
    from medcare_reachy.bridge.runner import MonitorStream, Runner
    from medcare_reachy.bridge.speech import Speaker
    from medcare_reachy.bridge.voice import DoneListener

    engine = emotion = None
    if not settings.get("vision_on_server", True):
        # On-robot landmarks (~3 fps on a Pi 4): sessions stay below the server's recording rate.
        from medcare_reachy.bridge.emotion import EmotionEngine
        from medcare_reachy.bridge.vision import VisionEngine   # ONNX Runtime; no MediaPipe, no AES needed

        await asyncio.to_thread(models.verify_models, models_dir)
        engine = await asyncio.to_thread(VisionEngine, models_dir)
        emotion = await asyncio.to_thread(EmotionEngine, models_dir / models.EMOTION_MODEL)
    robot = ReachyRobot.attach(reachy_mini)
    clips = ClipPlayer(robot, CLIPS_DIR, settings["language"])
    clips.audit()
    # A check-in turn Reachy answered with its 「嗯」 is followed, once handed over, by a thinking phrase.
    voice = DoneListener(robot, settings["language"], think_aloud=clips.think_aloud)
    voice.start()
    speaker = Speaker(robot)
    speaker.preload()
    try:
        async with AppClient(settings["app_url"], settings["device_token"]) as app:
            runner = Runner(app=app, robot=robot, clips=clips, voice=voice, speaker=speaker,
                            language=settings["language"],
                            stream=MonitorStream(app, robot, engine, fps=float(settings["capture_fps"]),
                                                 emotion=emotion),
                            ack=settings.get("checkin_ack", True), gestures=settings.get("checkin_gestures", True))
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
        self._thread: threading.Thread | None = None
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
                return
            settings = self.store.load()
            if not SettingsStore.configured(settings):
                self._set("not_configured", "Enter the server address and robot key on this page.")
                return
            self._set("starting", "Connecting to the server…")
            self._thread = threading.Thread(target=self._thread_main, args=(settings,),
                                            name="medcare-bridge", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 15.0) -> None:
        with self._lock:
            thread, loop, runner = self._thread, self._loop, self._runner
        if loop and runner:
            loop.call_soon_threadsafe(runner.request_shutdown)
        if thread:
            thread.join(timeout)
            if thread.is_alive():
                log.warning("bridge thread did not stop within %.0f s", timeout)
        with self._lock:
            self._thread = None
            if self._state not in ("error", "not_configured", "connecting_robot"):
                self._set("stopped", "")

    def restart(self) -> None:
        self.stop()
        self.start()

    def _current(self) -> bool:
        """Whether the calling bridge thread is still the service's one: a thread that outlived stop()'s
        timeout must not overwrite the state of the bridge that replaced it."""
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
            if self._current():
                self._runner = runner
                self._set("running", "Waiting for a medication reminder from the server.")

    def _set(self, state: str, detail: str) -> None:
        self._state, self._detail, self._since = state, detail, time.time()

    # ── status for the settings page ──
    def status(self) -> dict:
        with self._lock:
            runner = self._runner
            status = {"state": self._state, "detail": self._detail, "since": self._since}
        if runner is not None:
            slot = runner.slot
            status.update({
                "robot_reachable": runner.robot_reachable,
                "camera_fps": round(runner.stream.camera_fps(), 1),
                "landmark_fps": round(runner.stream.landmark_fps(), 1),
                "vision_fps": round(runner.stream.vision_fps(), 1),
                "missing_clips": runner.clips.missing_count,
                "slot_state": getattr(slot, "state", None) if slot else None,
                "task_id": slot.task.get("task_id") if slot else None,
            })
            if slot is not None:
                status["detail"] = STEP_TEXT.get(getattr(slot, "state", ""), "Working on a medication reminder.")
        return status
