"""Multimodal emotion for the dose-watching phase: face + voice + words, fused on the robot.

Ported from the `pearlyjam21/reachy_mini_multimodal_emotion` Space (its `engine` package): `labels`, `fusion`,
`runtime`, `speech` and `text` are its modules unchanged. The face branch is different: this app already knows
which face is the patient's (the server's verified box) and already ships that very FER+ MobileNetV3 as
`emotion_seed43.onnx`, so there is no face detector here and no OpenCV.

    mood = load_multimodal(models_dir, language)     # bundled face model + speech/text from the Hub (optional)
    mood.begin(intk_id)                              # a dose is watched: samples count towards its summary
    mood.score(frame, packet, target_box)            # MonitorStream, at most twice a second -> server report
    mood.hear(segment, began, ended, text)           # DoneListener, each speech segment it decodes
    mood.summary()                                   # what went into the dose's confirmation evidence
    mood.end()

Where it computes is decided by `governor.ComputeGovernor`: on the robot by default, falling back to the laptop
(the server scores the frames the robot streams) when the robot can't keep the target rate or keeps failing.
"""

from .engine import MultimodalEmotion, load_multimodal
from .fusion import BranchResult, FusedResult, fuse
from .governor import LAPTOP, MODES, ROBOT, ComputeGovernor
from .labels import EMOTIONS

__all__ = ["EMOTIONS", "LAPTOP", "MODES", "ROBOT", "BranchResult", "ComputeGovernor", "FusedResult",
           "MultimodalEmotion", "fuse", "load_multimodal"]
