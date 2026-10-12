"""DoneListener hands each decoded speech segment to the multimodal emotion engine."""

import numpy as np

from medcare_reachy.bridge.tests.test_voice import VAD_WINDOW, listener, wait_for


def test_each_decoded_segment_is_offered_to_the_segment_listener(tmp_path):
    voice, mic, vad, texts = listener(tmp_path)
    heard = []

    def on_segment(segment, began, ended, text):
        heard.append((len(segment), text, ended >= began))
        if len(heard) == 1:
            raise RuntimeError("the engine's problem, not the listener's")

    voice.on_segment = on_segment
    voice.start()
    try:
        voice.set_active(True)
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32), np.zeros(VAD_WINDOW * 4, np.float32)]
        assert wait_for(lambda: len(heard) == 2)
        assert heard == [(VAD_WINDOW * 2, "我吃完了", True)] * 2
        assert voice.heard_since(0) == "吃完"           # the claim is still recognised after a failing listener
    finally:
        voice.stop()


def test_nothing_is_offered_while_inactive(tmp_path):
    voice, mic, vad, texts = listener(tmp_path)
    heard = []
    voice.on_segment = lambda *args: heard.append(args)
    voice.start()
    try:
        mic.chunks = [np.zeros(VAD_WINDOW * 4, np.float32)]
        assert wait_for(lambda: not mic.chunks)
        assert heard == [] and texts == []
    finally:
        voice.stop()
