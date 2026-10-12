"""ONNX Runtime settings that leave the CM4's four cores to the robot daemon (motors, camera, audio)."""

from __future__ import annotations


def session_options(threads: int = 1):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # Idle worker threads busy-wait by default, which starves other processes on a small CPU
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return opts
