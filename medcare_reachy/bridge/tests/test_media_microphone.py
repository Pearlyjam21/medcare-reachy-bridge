"""ReachyRobot.get_audio starts the SDK's microphone pipeline itself (Reachy Mini SDK 1.8.3 never does)."""

import numpy as np

from medcare_reachy.bridge.media import RECORDING_RETRY_SECONDS, ReachyRobot


class Media:
    """A media manager as SDK 1.8.3 makes it: no microphone samples until start_recording()."""

    def __init__(self, fail=0):
        self.recording, self.starts, self.fail = False, 0, fail

    def start_recording(self):
        self.starts += 1
        if self.fail:
            self.fail -= 1
            raise RuntimeError("pipeline refused")
        self.recording = True

    def get_audio_sample(self):
        return np.ones((160, 2), np.float32) * 0.1 if self.recording else None

    def get_input_audio_samplerate(self):
        return 16000


class Mini:
    def __init__(self, media):
        self.media, self.media_released = media, False


def robot_for(media, clock=None):
    robot = ReachyRobot.attach(Mini(media))
    if clock is not None:
        robot._now = clock
    return robot


def test_the_first_read_starts_the_microphone_and_later_reads_do_not_restart_it():
    media = Media()
    robot = robot_for(media)
    samples, rate = robot.get_audio()
    assert media.starts == 1 and media.recording and rate == 16000 and samples.shape == (160, 2)
    robot.get_audio()
    robot.get_audio()
    assert media.starts == 1


def test_a_fresh_media_manager_gets_its_microphone_started_too():
    first, fresh = Media(), Media()
    robot = robot_for(first)
    robot.get_audio()
    robot._mini.media = fresh            # acquire_media() after a camera recovery builds a new manager
    assert robot.get_audio() is not None and fresh.starts == 1 and first.starts == 1


def test_a_failed_start_is_tried_again_later_not_on_every_read(caplog):
    now = {"t": 100.0}
    media = Media(fail=2)
    robot = robot_for(media, clock=lambda: now["t"])
    assert robot.get_audio() is None and media.starts == 1
    assert robot.get_audio() is None and media.starts == 1          # not before the retry time
    now["t"] += RECORDING_RETRY_SECONDS
    assert robot.get_audio() is None and media.starts == 2          # failed again
    now["t"] += RECORDING_RETRY_SECONDS
    assert robot.get_audio() is not None and media.starts == 3
    assert sum("did not start" in record.getMessage() for record in caplog.records) == 1


def test_no_media_means_no_read_and_no_start():
    media = Media()
    robot = robot_for(media)
    robot._mini.media_released = True     # the daemon has the media back
    assert robot.get_audio() is None and media.starts == 0
