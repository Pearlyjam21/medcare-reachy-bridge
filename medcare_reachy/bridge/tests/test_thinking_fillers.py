"""The check-in's thinking phrase: which clip the player picks for the "thinking" group, in turn.

On 3 Oct 2026 a patient asked about the weather and heard a 0.34 s 「嗯」, then ~6 s of silence: "it said 嗯 for one
millisecond and literally left me in silence". The 「嗯」 still answers the pause (the session's "ack"); once the
patient's words are handed over, Reachy says one of these phrases (voice.DoneListener's think_aloud).
"""

import random
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from medcare_reachy.bridge.clips import THINKING, ClipPlayer, load_manifest

PHRASES = ["thinking_1", "thinking_2", "thinking_3"]


class Robot:
    def __init__(self):
        self.played = []

    def play_clip(self, path, wait=True):
        self.played.append((Path(path).name, wait))
        return True


def write_clip(path: Path, seconds: float, rate=22050):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.zeros(int(seconds * rate), "<i2").tobytes())


def player(tmp_path, present, language="zh-TW", seed=1):
    for index, clip_id in enumerate(present):
        write_clip(tmp_path / language / f"{clip_id}.wav", 1.2 + 0.1 * index)
    robot = Robot()
    return ClipPlayer(robot, tmp_path, language, rng=random.Random(seed)), robot


def test_the_manifest_s_thinking_phrases():
    manifest = load_manifest()
    assert manifest["variants"] == {THINKING: PHRASES}
    assert THINKING not in manifest["clips"]
    assert manifest["clips"]["thinking_1"]["zh-TW"] == "我再想一下喔。"     # the patient's own suggestion
    for clip_id in PHRASES:
        text = manifest["clips"][clip_id]["zh-TW"]
        assert 4 <= len(text) <= 10 and text.endswith("。"), clip_id
        assert not text.startswith(("好", "嗯")), clip_id   # 好 sounds like agreeing; 嗯 was just said
    assert manifest["clips"]["ack"]["zh-TW"] == "嗯"                          # still the answer to the pause


def test_the_ack_is_the_lone_mm_again(tmp_path):
    clips, robot = player(tmp_path, ["ack", *PHRASES])
    assert clips.start("ack") == pytest.approx(1.2, abs=0.001)
    assert clips.played == ["ack"] and robot.played == [("ack.wav", False)]


@pytest.mark.parametrize("seed", range(20))
def test_every_phrase_once_per_round_and_never_the_same_twice_in_a_row(tmp_path, seed):
    clips, robot = player(tmp_path, ["ack", *PHRASES], seed=seed)
    for _ in range(30):
        assert clips.think_aloud() is not None
    picked = clips.played
    assert set(picked) == set(PHRASES)
    assert all(a != b for a, b in zip(picked, picked[1:]))
    for start in range(0, 30, 3):                                       # each round plays each phrase once
        assert sorted(picked[start:start + 3]) == PHRASES
    assert robot.played[0] == (f"{picked[0]}.wav", False)               # handed over without waiting


def test_it_is_not_always_the_same_order(tmp_path):
    orders = set()
    for seed in range(10):
        clips, _ = player(tmp_path / str(seed), PHRASES, seed=seed)
        orders.add(tuple(clips.variant(THINKING) for _ in range(6)))
    assert len(orders) > 1


def test_the_length_returned_is_the_phrase_s_so_its_echo_is_recognised_for_long_enough(tmp_path):
    clips, _ = player(tmp_path, ["ack", *PHRASES])
    seconds = clips.think_aloud()
    lengths = {"thinking_1": 1.3, "thinking_2": 1.4, "thinking_3": 1.5}
    assert seconds == pytest.approx(lengths[clips.played[-1]], abs=0.001)


def test_a_phrase_without_its_wav_is_skipped_and_reported(tmp_path):
    clips, _ = player(tmp_path, ["ack", "thinking_1", "thinking_3"])
    picked = [clips.played[-1] for _ in range(12) if clips.think_aloud()]
    assert set(picked) == {"thinking_1", "thinking_3"}
    assert all(a != b for a, b in zip(picked, picked[1:]))
    assert clips.missing == {"thinking_2"}
    assert clips.audit() == set(load_manifest()["clips"]) - {"ack", "thinking_1", "thinking_3"}


def test_one_phrase_alone_plays_every_time(tmp_path):
    clips, _ = player(tmp_path, ["ack", "thinking_2"])
    for _ in range(3):
        clips.think_aloud()
    assert clips.played == ["thinking_2"] * 3


def test_without_any_phrase_nothing_follows_the_mm(tmp_path):
    clips, robot = player(tmp_path, ["ack"])
    assert clips.think_aloud() is None and clips.play(THINKING) is False
    assert clips.played == [] and robot.played == []
    assert clips.missing == set(PHRASES)


def test_english_has_no_clips_yet_so_nothing_is_said(tmp_path):
    clips, robot = player(tmp_path, [], language="en")
    assert clips.start("ack") is None and clips.think_aloud() is None and robot.played == []


def test_other_clips_are_played_as_named(tmp_path):
    clips, robot = player(tmp_path, ["ack", *PHRASES, "thanks"])
    assert clips.play("thanks") is True and robot.played == [("thanks.wav", True)]
    with pytest.raises(KeyError):
        clips.start("not_a_clip")


def test_the_phrases_come_up_about_equally_often(tmp_path):
    clips, _ = player(tmp_path, ["ack", *PHRASES], seed=7)
    counts = Counter(clips.variant(THINKING) for _ in range(300))
    assert set(counts.values()) == {100}


def test_a_manifest_variant_that_is_no_clip_is_an_error(tmp_path):
    manifest = {**load_manifest(), "variants": {THINKING: ["thinking_9"]}}
    clips = ClipPlayer(Robot(), tmp_path, "zh-TW", manifest=manifest)
    with pytest.raises(KeyError):
        clips.think_aloud()
