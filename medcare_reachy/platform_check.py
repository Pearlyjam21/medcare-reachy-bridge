"""Decide whether on-robot vision can run here, BEFORE importing MediaPipe.

MediaPipe 1.x ARM64 wheels are built with ARMv8 AES instructions. The Reachy Mini Wireless's Raspberry Pi 4 CPU has
no crypto extensions, so importing it dies with SIGILL ("compiled with aes enabled … go/sigill-fail-fast"), which
Python can't catch; the app crash-loops. Older 0.10.x ARM64 wheels need numpy < 2, which conflicts with reachy-mini.
So on such robots we refuse to load vision and report why, instead of crashing.
"""

import importlib.util
import platform
from pathlib import Path

ARM_MACHINES = ("aarch64", "arm64", "armv8l")


def _read_cpuinfo() -> str:
    try:
        return Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _arm_has_aes(cpuinfo: str) -> bool:
    for line in cpuinfo.splitlines():
        key, _, value = line.partition(":")
        if key.strip().lower() == "features" and "aes" in value.split():
            return True
    return False


def vision_supported(machine: str | None = None, cpuinfo: str | None = None,
                     mediapipe_present: bool | None = None) -> tuple[bool, str]:
    machine = (machine or platform.machine()).lower()
    if mediapipe_present is None:
        mediapipe_present = importlib.util.find_spec("mediapipe") is not None
    if machine in ARM_MACHINES:
        if not _arm_has_aes(_read_cpuinfo() if cpuinfo is None else cpuinfo):
            return False, ("Camera intake detection can't run on this robot yet: its processor has no AES "
                           "instructions, which the vision library needs. Medication reminders still go out by LINE. "
                           "Next version: vision runs on the MedAiCarePlus server instead.")
    if not mediapipe_present:
        return False, "Camera intake detection needs the MediaPipe library, which isn't installed in this app."
    return True, ""
