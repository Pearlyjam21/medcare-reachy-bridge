def patch(path, pairs):
    s = open(path, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in s else "\n"
    s = s.replace("\r\n", "\n")
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:60], s.count(a))
        s = s.replace(a, b)
    open(path, "w", encoding="utf-8", newline="").write(s.replace("\n", nl))
    print("patched", path)


patch("medcare_reachy/bridge/media.py", [
('''    def look_around(self, step: int) -> None: ...
    def wake(self) -> None: ...''',
'''    def look_around(self, step: int) -> None: ...
    def look_at(self, u: float, v: float, duration: float) -> None: ...   # face centre in the streamed frame (0-1)
    def wake(self) -> None: ...'''),
('''    def look_around(self, step: int) -> None:
        self.set_head(head_pose(SCAN_YAWS_DEG[step % len(SCAN_YAWS_DEG)], pitch_deg=-5), duration=1.5)

    def wake(self) -> None:
        if self._mini is None:
            return
        self._gestures.halt()''',
'''    def look_around(self, step: int) -> None:
        self.set_head(head_pose(SCAN_YAWS_DEG[step % len(SCAN_YAWS_DEG)], pitch_deg=-5), duration=1.5)

    def look_at(self, u: float, v: float, duration: float = 0.35) -> None:
        """Turn the head towards a point of the streamed frame (normalised 0-1, see runner.encode_stream_jpeg):
        the server's face box is in that 4:3 centre crop, the SDK's look_at_image wants a pixel of the camera's
        own image."""
        media = self._media()
        if self._mini is None or media is None:
            return
        camera = getattr(media, "camera", None)
        width, height = (getattr(camera, "resolution", None) or (1280, 720))[:2]
        crop_width = min(width, int(height * 4 / 3))
        x0 = (width - crop_width) // 2
        px = int(round(x0 + min(1.0, max(0.0, u)) * crop_width))
        py = int(round(min(1.0, max(0.0, v)) * height))
        px, py = min(width - 1, max(1, px)), min(height - 1, max(1, py))
        self._gestures.halt()
        self._mini.look_at_image(px, py, duration=duration)

    def wake(self) -> None:
        if self._mini is None:
            return
        self._gestures.halt()'''),
('''    def look_around(self, step: int) -> None:
        self.actions.append(("look_around", step))
''',
'''    def look_around(self, step: int) -> None:
        self.actions.append(("look_around", step))

    def look_at(self, u: float, v: float, duration: float = 0.35) -> None:
        self.actions.append(("look_at", round(u, 2), round(v, 2)))
'''),
])

patch("medcare_reachy/bridge/session.py", [
('''                 clock=None, run_blocking=_inline, ack=True, gestures=True, mood=None):
        self.task = task
        self.mood = mood   # multimodal emotion for the dose being watched (begin/end/summary), or None
''',
'''                 clock=None, run_blocking=_inline, ack=True, gestures=True, mood=None, follow=None):
        self.task = task
        self.mood = mood   # multimodal emotion for the dose being watched (begin/end/summary), or None
        self.follow = follow   # follow.FaceFollower: the head keeps the verified face centred while a dose is watched
'''),
('''        latest = self._latest()
        now = self.clock.monotonic()
        self._note_done(now)
        recorded = latest.get("recorded")''',
'''        latest = self._latest()
        now = self.clock.monotonic()
        self._note_done(now)
        await self._follow_face(latest, now)
        recorded = latest.get("recorded")'''),
('''    def _go(self, state: str) -> None:
        log.info("slot %s: %s -> %s", self.task_id, self.state, state)
        self._sync_mood(state)
        self.state = state''',
'''    def _go(self, state: str) -> None:
        log.info("slot %s: %s -> %s", self.task_id, self.state, state)
        self._sync_mood(state)
        self._sync_follow(state)
        self.state = state'''),
('''    def _mood_evidence(self) -> dict:''',
'''    def _sync_follow(self, state: str) -> None:
        if self.follow is None:
            return
        if state == WATCHING and self.state != WATCHING:
            self.follow.start()
        elif state != WATCHING and self.state == WATCHING:
            self.follow.stop()

    async def _follow_face(self, latest: dict, now: float) -> None:
        """Keep the head on the verified face (the server's target_box); nothing while a candidate event is being
        verified, so the hand-to-mouth detector works on a still camera."""
        if self.follow is None or latest.get("candidate"):
            return
        box = latest.get("target_box") if latest.get("identity_status") == "verified" else None
        await self.run_blocking(self.follow.update, box, now)

    def _mood_evidence(self) -> dict:'''),
('''        self._sync_mood(SLEEP)
        self.result = result''',
'''        self._sync_mood(SLEEP)
        self._sync_follow(SLEEP)
        self.result = result'''),
])

patch("medcare_reachy/bridge/runner.py", [
('''                 ack=True, gestures=True, mood=None):
        self.app, self.robot, self.clips, self.stream, self.voice = app, robot, clips, stream, voice
        self.speaker, self.language, self.mood = speaker, language, mood
''',
'''                 ack=True, gestures=True, mood=None, follow=None):
        self.app, self.robot, self.clips, self.stream, self.voice = app, robot, clips, stream, voice
        self.speaker, self.language, self.mood, self.follow = speaker, language, mood, follow
'''),
('''ack=self.ack, gestures=self.gestures, mood=self.mood)''',
 '''ack=self.ack, gestures=self.gestures, mood=self.mood,
                           follow=self.follow)'''),
])

patch("medcare_reachy/settings_store.py", [
('''            "checkin_ack": True, "checkin_gestures": True, "multimodal_emotion": True,
            "emotion_compute": "auto"}''',
'''            "checkin_ack": True, "checkin_gestures": True, "multimodal_emotion": True,
            "emotion_compute": "auto", "follow_face": True}'''),
('''SWITCHES = ("vision_on_server", "checkin_ack", "checkin_gestures", "multimodal_emotion")''',
'''# follow_face: while a dose is watched the head keeps the verified patient's face centred (bridge/follow.py).
SWITCHES = ("vision_on_server", "checkin_ack", "checkin_gestures", "multimodal_emotion", "follow_face")'''),
])
patch("medcare_reachy/web.py", [
('''    multimodal_emotion: bool | None = None
''', '''    multimodal_emotion: bool | None = None
    follow_face: bool | None = None
'''),
])
patch("medcare_reachy/static/index.html", [
('''        <label class="check">
          <input name="multimodal_emotion" type="checkbox" checked>''',
'''        <label class="check">
          <input name="follow_face" type="checkbox" checked>
          <span>Follow the patient's face while a medicine is taken
            <small>From the prompt until the dose is resolved, Reachy turns its head to keep the patient's face in the middle of its camera. It holds still while a hand-to-mouth movement is being checked.</small>
          </span>
        </label>
        <label class="check">
          <input name="multimodal_emotion" type="checkbox" checked>'''),
])
patch("medcare_reachy/static/main.js", [
('''  form.multimodal_emotion.checked = config.multimodal_emotion !== false;
''', '''  form.multimodal_emotion.checked = config.multimodal_emotion !== false;
  form.follow_face.checked = config.follow_face !== false;
'''),
('''    multimodal_emotion: form.multimodal_emotion.checked,
''', '''    multimodal_emotion: form.multimodal_emotion.checked,
    follow_face: form.follow_face.checked,
'''),
])
patch("medcare_reachy/bridge/tests/fakes.py", [
('''    def hold_head(self):
        self.calls.append("hold_head")
''', '''    def hold_head(self):
        self.calls.append("hold_head")

    def look_at(self, u, v, duration=0.35):
        self.calls.append(f"look_at:{u:.2f},{v:.2f}")
'''),
])
s = open("medcare_reachy/service.py", encoding="utf-8").read()
print([l.strip() for l in s.split("\n") if "Runner(" in l or "mood=" in l or "gestures=settings" in l or "voice = DoneListener" in l])
