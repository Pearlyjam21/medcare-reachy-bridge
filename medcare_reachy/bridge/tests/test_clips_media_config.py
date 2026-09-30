import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from medcare_reachy.bridge.clips import ClipPlayer, load_manifest
from medcare_reachy.bridge.config import Settings
from medcare_reachy.bridge.media import VideoFileRobot, crop_4_3, head_pose, read_wav, resample

REQUIRED_CLIPS = {"wake_greeting", "reminder", "searching", "waiting_for_patient", "med_prompt_generic",
                  "already_taken", "confirm_with_caregiver", "help", "wind_down", "thanks", "waiting_for_tablet"}


class RecordingRobot:
    def __init__(self):
        self.paths = []

    def play_clip(self, path):
        self.paths.append(Path(path))
        return True


def write_wav(path: Path, samples: np.ndarray, rate=16000, channels=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes((samples * 32767).astype("<i2").tobytes())


def test_manifest_has_every_clip_in_both_languages_pending_review():
    manifest = load_manifest()
    assert set(manifest["clips"]) == REQUIRED_CLIPS
    for clip_id, entry in manifest["clips"].items():
        assert entry["zh-TW"].strip() and entry["en"].strip(), clip_id
        assert entry["review_status"] == "pending_clinician_review", clip_id
    assert manifest["per_medication"]["fallback"] == "med_prompt_generic"


def test_manifest_never_talks_about_other_doses_or_the_rest_of_the_day():
    for entry in load_manifest()["clips"].values():
        text = entry["en"].lower()
        assert "today" not in text and "any more" not in text and "tomorrow" not in text


def test_player_plays_existing_wav_and_counts_missing(tmp_path):
    write_wav(tmp_path / "en" / "thanks.wav", np.zeros(160))
    robot = RecordingRobot()
    player = ClipPlayer(robot, tmp_path, "en")
    assert player.play("thanks") is True
    assert robot.paths == [tmp_path / "en" / "thanks.wav"]
    assert player.play("help") is False and player.play("help") is False
    assert player.missing == {"help"} and player.missing_count == 1
    with pytest.raises(KeyError):
        player.play("not_a_clip")


def test_player_audit_reports_all_missing_manifest_clips(tmp_path):
    write_wav(tmp_path / "zh-TW" / "thanks.wav", np.zeros(160))
    player = ClipPlayer(RecordingRobot(), tmp_path, "zh-TW")
    assert player.audit() == REQUIRED_CLIPS - {"thanks"}


def test_med_prompt_prefers_per_medication_clip(tmp_path):
    write_wav(tmp_path / "en" / "med_prompt_generic.wav", np.zeros(160))
    write_wav(tmp_path / "en" / "med_42.wav", np.zeros(160))
    robot = RecordingRobot()
    player = ClipPlayer(robot, tmp_path, "en")
    player.play_med_prompt(42)
    player.play_med_prompt(7)
    assert [p.name for p in robot.paths] == ["med_42.wav", "med_prompt_generic.wav"]


def test_crop_4_3_centre_crops_16_9_to_640x480():
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[:, 160:1120] = 200          # the 4:3 centre
    out = crop_4_3(frame)
    assert out.shape == (480, 640, 3)
    assert out.min() == 200


def test_crop_4_3_passes_640x480_through():
    frame = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    assert crop_4_3(frame) is frame


def test_head_pose_is_identity_at_rest_and_rotates_about_z():
    assert np.allclose(head_pose(), np.eye(4))
    pose = head_pose(90)
    assert np.allclose(pose[:3, :3] @ [1, 0, 0], [0, 1, 0])


def test_read_wav_downmixes_and_resample_changes_rate(tmp_path):
    stereo = np.tile(np.array([0.5, -0.5], np.float32), 800)
    write_wav(tmp_path / "s.wav", stereo, rate=16000, channels=2)
    samples, rate = read_wav(tmp_path / "s.wav")
    assert rate == 16000 and len(samples) == 800 and abs(samples).max() < 1e-3
    assert len(resample(np.ones(16000, np.float32), 16000, 48000)) == 48000


def test_video_file_robot_replays_frames(tmp_path):
    import cv2

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (1280, 720))
    if not writer.isOpened():
        pytest.skip("no MJPG writer in this OpenCV build")
    for i in range(3):
        writer.write(np.full((720, 1280, 3), i * 60, np.uint8))
    writer.release()
    robot = VideoFileRobot(str(path), fps=1000)
    frames = []
    while len(frames) < 4:
        frame = robot.get_frame()
        if frame is not None:
            frames.append(frame)
    assert all(frame.shape == (480, 640, 3) for frame in frames)   # loops past the end
    assert robot.is_reachable()
    robot.wake()
    robot.look_around(2)
    assert robot.actions == [("wake",), ("look_around", 2)]


def test_reachy_sdk_is_not_imported_by_the_package():
    import medcare_reachy.bridge.media  # noqa: F401
    import medcare_reachy.bridge.runner  # noqa: F401
    assert "reachy_mini" not in sys.modules and "mediapipe" not in sys.modules


def test_settings_from_env_and_validation(tmp_path):
    env = {"REACHY_DEVICE_TOKEN": "rdv1.abc", "REACHY_ROBOT_HOST": "192.168.1.50",
           "APP_INTERNAL_URL": "http://app:8001/", "LANGUAGE": "en", "CLIPS_DIR": str(tmp_path)}
    settings = Settings.from_env(env)
    assert settings.app_url == "http://app:8001" and settings.robot_backend == "reachy"
    assert settings.clips_dir == tmp_path and settings.language == "en"
    assert settings.models_dir.name == "models"
    for bad in ({"REACHY_DEVICE_TOKEN": "nope"}, {"LANGUAGE": "fr"}, {"ROBOT_BACKEND": "sim"},
                {"REACHY_ROBOT_HOST": ""}, {"ROBOT_BACKEND": "video"}):
        with pytest.raises(ValueError):
            Settings.from_env({**env, **bad})
    video = Settings.from_env({**env, "ROBOT_BACKEND": "video", "VIDEO_PATH": "x.mp4", "REACHY_ROBOT_HOST": ""})
    assert video.robot_backend == "video"


class _FakeMini:
    def __init__(self):
        self.exited = False
        self.targets = []

    def __exit__(self, *exc):
        self.exited = True

    def goto_target(self, **kwargs):
        self.targets.append(kwargs)


def test_attached_robot_uses_the_given_connection_and_never_closes_it():
    from medcare_reachy.bridge.media import ReachyRobot

    mini = _FakeMini()
    robot = ReachyRobot.attach(mini)
    assert robot.is_reachable()          # no reconnect attempt: the app owns the connection
    robot.hold_head()
    assert mini.targets and "head" in mini.targets[0]
    robot.close()
    assert mini.exited is False           # the Reachy Mini app runtime closes it, not us


def test_owned_robot_still_closes_its_own_connection():
    from medcare_reachy.bridge.media import ReachyRobot

    mini = _FakeMini()
    robot = ReachyRobot("192.168.1.50")
    robot._mini = mini
    robot.close()
    assert mini.exited is True
