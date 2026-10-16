"""ReachyRobot's speaker for the live chat (bridge/live.py): start_live_audio, push_audio, clear_live_audio and
stop_live_audio on the SDK's media calls as reachy_mini 1.11.0 has them."""

import time

import numpy as np
import pytest

from medcare_reachy.bridge.media import LIVE_TAIL_SECONDS, ReachyRobot, VideoFileRobot


class Audio:
    """GStreamerAudio's barge-in call: the MediaManager itself has no clear_player()."""

    def __init__(self):
        self.cleared = 0

    def clear_player(self):
        self.cleared += 1


class Media:
    """One SDK media manager (MediaManager): what the live chat's speaker calls."""

    def __init__(self, out_rate=16000, audio=True):
        self.calls, self.pushed, self.out_rate = [], [], out_rate
        self.audio = Audio() if audio else None

    def start_playing(self):
        self.calls.append("start_playing")

    def stop_playing(self):
        self.calls.append("stop_playing")

    def push_audio_sample(self, samples):
        self.pushed.append(samples)

    def get_output_audio_samplerate(self):
        return self.out_rate


class Mini:
    def __init__(self, media):
        self.media = media
        self.media_released = False


def test_the_live_chat_starts_the_speaker_once_and_pushes_its_blocks_as_they_are():
    mini = Mini(Media())
    robot = ReachyRobot.attach(mini)
    assert robot.start_live_audio() is True
    block = np.full(320, 0.25, np.float32)
    assert robot.push_audio(block) is True and robot.push_audio(block) is True
    assert mini.media.calls == ["start_playing"]
    assert [len(pushed) for pushed in mini.media.pushed] == [320, 320]
    assert mini.media.pushed[0].dtype == np.float32 and np.all(mini.media.pushed[0] == 0.25)


def test_blocks_are_resampled_for_a_speaker_at_another_rate():
    mini = Mini(Media(out_rate=48000))
    robot = ReachyRobot.attach(mini)
    assert robot.push_audio(np.zeros(320, np.float32)) is True
    assert len(mini.media.pushed[0]) == 960


def test_without_media_there_is_no_live_audio():
    mini = Mini(Media())
    mini.media_released = True
    robot = ReachyRobot.attach(mini)
    assert robot.start_live_audio() is False and robot.push_audio(np.zeros(320, np.float32)) is False
    robot.clear_live_audio()
    assert mini.media.calls == [] and mini.media.audio.cleared == 0


def test_a_flush_clears_the_sdk_player():
    mini = Mini(Media())
    robot = ReachyRobot.attach(mini)
    robot.clear_live_audio()
    assert mini.media.audio.cleared == 1
    ReachyRobot.attach(Mini(Media(audio=False))).clear_live_audio()   # an SDK without the call: nothing to clear


def test_the_end_of_a_live_chat_never_stops_the_shared_pipeline_and_the_next_clip_waits():
    mini = Mini(Media())
    robot = ReachyRobot.attach(mini)
    robot.start_live_audio()
    before = time.monotonic()
    robot.stop_live_audio()
    assert "stop_playing" not in mini.media.calls        # the microphone is in the same GStreamer pipeline
    assert robot._quiet_at >= before + LIVE_TAIL_SECONDS  # play_clip waits for the audio pushed last


def test_a_video_file_robot_has_no_live_audio(tmp_path):
    import cv2

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (64, 48))
    if not writer.isOpened():
        pytest.skip("no MJPG writer in this OpenCV build")
    writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    robot = VideoFileRobot(str(path))
    assert robot.start_live_audio() is False and robot.push_audio(np.zeros(320, np.float32)) is False
    robot.clear_live_audio()
    robot.stop_live_audio()
    assert robot.actions == []


# ── the SDK's continuous timestamps (LIVE_PTS_RESYNC_SECONDS) ──
def test_a_push_that_fell_behind_is_preceded_by_silence_for_the_gap():
    """Measured 13 Oct 2026: blocks 0.1-0.2 s behind the SDK's timeline stay behind and the speaker drops them all
    (the silent chats); fill_live_audio() pushes the gap as silence so the next block is on time."""
    mini = Mini(Media())
    robot = ReachyRobot.attach(mini)
    clock = [100.0]
    robot._now = lambda: clock[0]
    robot.start_live_audio()
    robot.push_audio(np.full(320, 0.25, np.float32))        # the timeline: the next block is due at 100.02
    assert robot.fill_live_audio() == 0.0                    # not behind
    clock[0] = 100.17                                        # 150 ms behind
    assert robot.fill_live_audio() == pytest.approx(0.15, abs=1 / 16000)
    assert len(mini.media.pushed) == 2 and len(mini.media.pushed[1]) == 2400 and not np.any(mini.media.pushed[1])
    robot.push_audio(np.full(320, 0.25, np.float32))        # on time again: the timeline continues
    assert robot._live_next_at == pytest.approx(100.17 + 0.02, abs=1e-3)
    assert robot.fill_live_audio() == 0.0


def test_a_gap_the_sdk_resyncs_by_itself_is_not_filled_and_a_flush_anchors_afresh():
    mini = Mini(Media())
    robot = ReachyRobot.attach(mini)
    clock = [100.0]
    robot._now = lambda: clock[0]
    robot.start_live_audio()
    robot.push_audio(np.zeros(320, np.float32))
    clock[0] = 100.32                                        # 300 ms behind: over LIVE_PTS_RESYNC_SECONDS
    assert robot.fill_live_audio() == 0.0 and len(mini.media.pushed) == 1
    robot.push_audio(np.zeros(320, np.float32))              # the SDK anchors at now; so does the facade
    assert robot._live_next_at == pytest.approx(100.34, abs=1e-6)
    robot.clear_live_audio()                                 # clear_player(): afresh at the next push
    assert robot._live_next_at is None and robot.fill_live_audio() == 0.0
    clock[0] = 200.0
    robot.push_audio(np.zeros(320, np.float32))
    assert robot._live_next_at == pytest.approx(200.02, abs=1e-6)
    assert ReachyRobot.attach(Mini(Media())).fill_live_audio() == 0.0   # nothing pushed yet: nothing to fill
