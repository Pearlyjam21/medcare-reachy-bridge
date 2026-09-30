"""MedCare Reachy: medication-time companion app for Reachy Mini.

At dose time the MedAiCarePlus server hands this robot a task; the app wakes, finds the patient, prompts each
medicine with prerecorded clips, and streams camera landmarks to the server, which decides what is recorded.
Configure it on the app's settings page (server address + the robot key shown once when pairing).
"""

import logging
import threading

from reachy_mini import ReachyMini, ReachyMiniApp

from medcare_reachy.service import BridgeService
from medcare_reachy.settings_store import SettingsStore
from medcare_reachy.web import register_routes


class MedcareReachy(ReachyMiniApp):
    custom_app_url: str | None = "http://0.0.0.0:8042"
    # The app streams camera frames, so it needs the video-capable media backend.
    request_media_backend: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register the settings API now: the runtime serves the page before it has finished connecting to
        # the robot and calling run(), and the page must not get 404s in that window.
        self.service = BridgeService(None, SettingsStore())
        register_routes(self.settings_app, self.service)

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        self.service.attach(reachy_mini)
        self.service.start()   # does nothing until the settings page has a server address and robot key
        try:
            stop_event.wait()
        finally:
            self.service.stop()


if __name__ == "__main__":
    app = MedcareReachy()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
