"""Copy this working copy's medcare_reachy package onto the robot and switch it to server-side vision.

Interim deployment until the Hugging Face space is updated: reinstalling the app from the dashboard puts the
space's version back. The installed package is backed up first (~/.medcare_reachy/backup-<time>.tgz).

    python tools/deploy_to_robot.py [--host reachy-mini.local] [--user pollen] [--camera-ipc-fps 15]

The SSH password comes from REACHY_SSH_PASSWORD, or is asked for.

Voice clips: every zh-TW clip whose WAV is missing on the robot, or was rendered from other words than the
manifest has now, is rendered again with Matcha-TTS on the robot (~/.medcare_reachy/clips/zh-TW/rendered.json
records what each WAV says). The deploy stops when a planned clip was not written, and when any manifest clip has no
WAV on the robot afterwards; a clip Whisper does not hear exactly as written is kept with a warning to listen to
it. The check-in's 「嗯」 (ack) and thinking phrases (manifest "variants") are kept only when Whisper hears them as
words the app recognises as its own echo; otherwise they are not written and the deploy fails.

Camera: Reachy Mini's daemon (reachy_mini 1.11) shares the camera with apps at IPC_FPS = 10 frames per second,
below the 12 fps the server needs to record a dose. --camera-ipc-fps 15 edits that constant in the daemon (a
backup is kept next to it); it takes effect when the daemon restarts, which this script never does.
"""

import argparse
import getpass
import json
import os
import posixpath
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "medcare_reachy"
MANIFEST = PACKAGE / "bridge" / "clips" / "manifest.json"
REMOTE_PKG = "/venvs/apps_venv/lib/python3.12/site-packages/medcare_reachy"
PYTHON = "/venvs/apps_venv/bin/python"
SKIP_DIRS = {"__pycache__", "vision_models"}
VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
CLIPS_DIR = ".medcare_reachy/clips/zh-TW"     # under the robot user's home; linked into the package
RENDERED = "rendered.json"
DAEMON_MEDIA_SERVER = "/venvs/mini_daemon/lib/python3.12/site-packages/reachy_mini/media/media_server.py"
CAMERA_FPS_NEEDED = 15

# The Matcha zh-baker voice reads Simplified Chinese. For each zh-TW clip: the manifest text it was written for,
# and the words the voice is given. tests/test_deploy_tool.py fails when the manifest changes without this table.
SPOKEN = {
    "wake_greeting": ("您好！吃藥的時間到了。", "您好！吃药的时间到了。"),
    "reminder": ("提醒您，現在是吃藥的時間。", "提醒您，现在是吃药的时间。"),
    "searching": ("我正在找您，請到我面前來。", "我正在找您，请到我面前来。"),
    "waiting_for_patient": ("我在等要吃藥的人，請他到我面前來。", "我在等要吃药的人，请他到我面前来。"),
    "waiting_for_tablet": ("我會等您在平板上完成。", "我会等您在平板上完成。"),
    "say_when_done": ("吃完藥以後，請跟我說「我吃完了」。", "吃完药以后，请跟我说：我吃完了。"),
    "med_prompt_generic": ("請現在吃下一種藥。", "请现在吃下一种药。"),
    "already_taken": ("這次的藥您已經吃過了，做得很好！", "这次的药您已经吃过了，做得很好！"),
    "confirm_with_caregiver": ("我沒辦法確定您吃了沒，會請家人幫忙看看。請先別再吃一次喔。",
                               "我没办法确定您吃了没，会请家人帮忙看看。请先别再吃一次哦。"),
    "help": ("需要幫忙嗎？準備好的時候，請在我面前吃藥。", "需要帮忙吗？准备好的时候，请在我面前吃药。"),
    "wind_down": ("這次就到這裡，請多保重！", "这次就到这里，请多保重！"),
    "thanks": ("謝謝您！", "谢谢您！"),
    # Check-in: said the moment the patient pauses, before anyone knows what they said. 「好的」 or 「嗯，好」 would
    # sound like agreeing, even before a help-line reply.
    "ack": ("嗯", "嗯"),
    # Said once the patient's words are handed over (manifest variants "thinking", one each time): Reachy is thinking
    # about what was said. After the 「嗯」, so none starts with one; on 3 Oct a patient heard the 0.34 s 「嗯」 alone as
    # a blip followed by silence and asked for 「我再想一下」.
    "thinking_1": ("我再想一下喔。", "我再想一下哦。"),
    "thinking_2": ("讓我想一想喔。", "让我想一想哦。"),
    "thinking_3": ("我想想看喔。", "我想想看哦。"),
}
# Speeds a clip is rendered at, best first (the first Whisper hears word for word is kept). Prompts are slow for
# older listeners; the thinking phrases carry no information and are said at the voice's own pace (about 1.1-1.3 s on
# the laptop with the robot's model files; at 0.85 they take 1.4-1.5 s).
DEFAULT_SPEEDS = (0.85, 0.8, 0.9)
RENDER_SPEEDS = {"thinking_1": (1.0, 1.1, 0.9), "thinking_2": (1.0, 1.1, 0.9), "thinking_3": (1.0, 1.1, 0.9)}

# Runs on the robot, after the upload: renders the clips listed in the job file, checks each by reading it back with
# Whisper (a few speeds; the best match is kept), and records what every WAV says. A check-in acknowledgement
# ("filler") is kept only as heard so that the app recognises its echo (voice.without_filler of the package just
# uploaded); when no speed gives that, it is not written ("unrecognised"), so it is never played and the deploy fails.
RENDER = r'''
import difflib
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np

M = Path.home() / ".medcare_reachy" / "models"
T = M / "tts" / "matcha-icefall-zh-baker"
OUT = Path.home() / ".medcare_reachy" / "clips" / "zh-TW"
STAMP = OUT / "rendered.json"

job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rendered = dict(job["rendered"])
OUT.mkdir(parents=True, exist_ok=True)


def save_stamp():
    tmp = OUT / ".rendered.json.tmp"
    tmp.write_text(json.dumps(rendered, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, STAMP)


if job["clips"]:
    import sherpa_onnx
    from faster_whisper import WhisperModel

    if any(clip.get("filler") for clip in job["clips"]):
        from medcare_reachy.bridge.voice import without_filler

    fsts = ",".join(str(T / f) for f in ("phone.fst", "date.fst", "number.fst") if (T / f).is_file())
    tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
            acoustic_model=str(T / "model-steps-3.onnx"), vocoder=str(M / "tts" / "vocos-22khz-univ.onnx"),
            lexicon=str(T / "lexicon.txt"), tokens=str(T / "tokens.txt")), num_threads=2),
        rule_fsts=fsts, max_num_sentences=1))
    asr = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=2,
                       download_root=str(M / "stt" / "whisper"))
    for clip in job["clips"]:
        want = "".join(ch for ch in clip["text"] if ch.isalnum())
        best = None
        for speed in clip.get("speeds") or (0.85, 0.8, 0.9):
            audio = tts.generate(clip["text"], sid=0, speed=speed)
            samples = np.asarray(audio.samples, dtype=np.float32)
            verify_wav = OUT / ".verify.wav"
            with wave.open(str(verify_wav), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(audio.sample_rate)
                w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
            pieces, _ = asr.transcribe(str(verify_wav), language="zh", beam_size=1,
                                       condition_on_previous_text=False)
            heard = "".join(piece.text for piece in pieces).strip()
            verify_wav.unlink(missing_ok=True)
            got = "".join(ch for ch in heard if ch.isalnum())
            usable = not clip.get("filler") or without_filler(heard) == ""
            print(f"render {clip['id']} speed={speed} heard={heard!r} ok={got == want}"
                  + ("" if usable else " echo-not-recognised"))
            score = difflib.SequenceMatcher(None, got, want).ratio()
            if usable and (best is None or score > best[0]):
                best = (score, samples, audio.sample_rate)
            if usable and got == want:
                break
        if best is None:
            print("unrecognised", clip["id"])
            continue
        score, samples, rate = best
        # Matcha starts with ~70 ms of near-silence (the 「嗯」 must come as soon as possible): keep 10 ms of it.
        loud = np.flatnonzero(np.abs(samples) > 0.003)   # ~-50 dBFS
        if len(loud):
            samples = samples[max(0, loud[0] - rate // 100):]
        tmp = OUT / f".{clip['id']}.wav.tmp"
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        os.replace(tmp, OUT / f"{clip['id']}.wav")
        rendered[clip["id"]] = clip["written_for"]
        save_stamp()
        print("wrote", clip["id"], "heard-ok" if score == 1.0 else "not-heard-as-written")
save_stamp()
'''


def spoken_text(clip_id: str, manifest: dict) -> str:
    """The Simplified words the voice says for a clip; refuses a SPOKEN entry written for other manifest text."""
    written_for, spoken = SPOKEN.get(clip_id, (None, None))
    if written_for != manifest["clips"][clip_id]["zh-TW"]:
        raise SystemExit(f"clip {clip_id}: its zh-TW text in the manifest changed; update SPOKEN in {__file__}")
    return spoken


def fillers(manifest: dict) -> set:
    """The check-in's 「嗯」 and thinking phrases (every manifest variant): clips whose echo the app must recognise by
    their words."""
    variants = {clip_id for group in manifest.get("variants", {}).values() for clip_id in group}
    return ({"ack"} | variants) & set(manifest["clips"])


def render_job_clip(clip_id: str, manifest: dict) -> dict:
    return {"id": clip_id, "text": spoken_text(clip_id, manifest), "written_for": manifest["clips"][clip_id]["zh-TW"],
            "speeds": list(RENDER_SPEEDS.get(clip_id, DEFAULT_SPEEDS)), "filler": clip_id in fillers(manifest)}


def plan_clips(manifest: dict, rendered: dict | None, installed: dict | None, on_robot: set) -> tuple[dict, list]:
    """What each WAV on the robot says now, and the clips to render: missing, or made from other words.

    `rendered` is the robot's record of what each WAV says. Robots set up before that record existed have WAVs
    made from the manifest of the app installed there, so `installed` stands in for it.
    """
    if rendered is None:
        rendered = {clip_id: entry["zh-TW"] for clip_id, entry in (installed or {}).get("clips", {}).items()
                    if clip_id in on_robot}
    stale = [clip_id for clip_id, entry in manifest["clips"].items()
             if clip_id not in on_robot or rendered.get(clip_id) != entry["zh-TW"]]
    return rendered, stale


def prepare_clips(sftp, manifest: dict) -> dict:
    """The render job for this deploy; also saves the robot's record of what its WAVs say when it had none.

    The record has to exist before the new manifest is uploaded: a later deploy would otherwise take the new
    manifest's words for what the WAVs say, and a clip whose render failed would never be rendered again.
    A clip without its Simplified words stops the deploy here, before anything on the robot changes.
    """
    record_path = posixpath.join(CLIPS_DIR, RENDERED)
    record = read_json(sftp, record_path)
    rendered, stale = plan_clips(manifest, record, read_json(sftp, f"{REMOTE_PKG}/bridge/clips/manifest.json"),
                                 wav_clips(sftp))
    job = {"rendered": rendered, "clips": [render_job_clip(clip_id, manifest) for clip_id in stale]}
    if record is None and rendered:   # WAVs exist, so their folder does
        temporary = posixpath.join(CLIPS_DIR, f".{RENDERED}.tmp")
        with sftp.file(temporary, "w") as handle:
            handle.write(json.dumps(rendered, ensure_ascii=False, indent=1))
        sftp.posix_rename(temporary, record_path)
    return job


def render_problems(output: str, job: dict) -> tuple[list, list]:
    """From the render script's output: the clips it did not write, and those Whisper heard otherwise."""
    wrote = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "wrote":
            wrote[parts[1]] = parts[2] == "heard-ok"
    if "render-exit=0" not in output.splitlines():
        wrote = {}   # the script itself failed (e.g. killed for memory): trust none of it
    planned = [clip["id"] for clip in job["clips"]]
    return [c for c in planned if c not in wrote], [c for c in planned if wrote.get(c) is False]


def unrecognised(output: str) -> list:
    """Acknowledgements the render script left unwritten: at no speed would the app recognise their echo."""
    words = [line.split() for line in output.splitlines()]
    return [parts[1] for parts in words if len(parts) == 2 and parts[0] == "unrecognised"]


def absent_clips(manifest: dict, on_robot: set) -> list:
    """Manifest clips without a WAV on the robot."""
    return sorted(set(manifest["clips"]) - set(on_robot))


def run(client, command: str, timeout: float = 120) -> str:
    _, out, err = client.exec_command(command, timeout=timeout)
    text = out.read().decode(errors="replace")
    status = out.channel.recv_exit_status()
    error = err.read().decode(errors="replace")
    if status != 0:
        raise RuntimeError(f"`{command}` failed ({status}): {error or text}")
    return text


def read_json(sftp, path: str):
    try:
        with sftp.open(path) as handle:
            return json.loads(handle.read().decode("utf-8"))
    except (OSError, ValueError):
        return None


def wav_clips(sftp) -> set:
    try:
        return {name[:-4] for name in sftp.listdir(CLIPS_DIR) if name.endswith(".wav")}
    except OSError:
        return set()


def upload_package(client) -> int:
    sftp = client.open_sftp()
    count = 0
    try:
        for local in sorted(PACKAGE.rglob("*")):
            relative = local.relative_to(PACKAGE)
            if any(part in SKIP_DIRS for part in relative.parts) or local.is_dir():
                continue
            remote = posixpath.join(REMOTE_PKG, *relative.parts)
            run(client, f"mkdir -p '{posixpath.dirname(remote)}'")
            sftp.put(str(local), remote)
            count += 1
    finally:
        sftp.close()
    return count


def daemon_camera_fps(client) -> int | None:
    line = run(client, f"grep -m1 -E '^IPC_FPS = [0-9]+$' {DAEMON_MEDIA_SERVER} || true").strip()
    return int(line.split("=")[1]) if line else None


def check_camera_rate(client, wanted: int | None, stamp: str) -> None:
    """Report (and with --camera-ipc-fps, change) the rate the daemon shares the camera with apps at."""
    current = daemon_camera_fps(client)
    if current is None:
        print("camera: no IPC_FPS in the daemon; check the app's 'Camera feed' rate on its settings page")
        return
    if wanted is not None and wanted != current:
        run(client, f"sed -i.bak-{stamp} 's/^IPC_FPS = [0-9]*$/IPC_FPS = {wanted}/' {DAEMON_MEDIA_SERVER}")
        print(f"camera: daemon IPC_FPS {current} -> {daemon_camera_fps(client)} "
              f"(backup {DAEMON_MEDIA_SERVER}.bak-{stamp}).\n"
              "  It applies once the daemon restarts. Do that when no reminder is running: "
              "sudo systemctl restart reachy-mini-daemon,\n  then start the app again if it doesn't come back "
              "(curl -X POST localhost:8000/api/apps/start-app/medcare_reachy).")
    elif current < CAMERA_FPS_NEEDED:
        print(f"WARNING camera: the daemon shares the camera with apps at {current} fps (IPC_FPS), and the server "
              f"records a dose only at >= 12 fps.\n  Every robot dose goes to family confirmation until it is raised: "
              f"rerun with --camera-ipc-fps {CAMERA_FPS_NEEDED}.")
    else:
        print(f"camera: daemon shares {current} fps with apps")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="reachy-mini.local")
    parser.add_argument("--user", default="pollen")
    parser.add_argument("--camera-ipc-fps", type=int, choices=range(12, 31), metavar="12-30",
                        help=f"set the daemon's camera rate for apps (the server needs {CAMERA_FPS_NEEDED})")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # the clip check prints Chinese
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    password = os.environ.get("REACHY_SSH_PASSWORD") or getpass.getpass(f"SSH password for {args.user}@{args.host}: ")

    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.user, password=password, timeout=15)
    try:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        sftp = client.open_sftp()
        try:
            job = prepare_clips(sftp, manifest)
        finally:
            sftp.close()
        stale = [clip["id"] for clip in job["clips"]]

        run(client, f"mkdir -p ~/.medcare_reachy && tar -czf ~/.medcare_reachy/backup-{stamp}.tgz "
                    f"-C {posixpath.dirname(REMOTE_PKG)} --exclude=vision_models --exclude=__pycache__ medcare_reachy")
        print(f"backed up the installed app to ~/.medcare_reachy/backup-{stamp}.tgz")

        print(f"uploaded {upload_package(client)} files")
        run(client, f"find {REMOTE_PKG} -name __pycache__ -type d -prune -exec rm -rf {{}} +")
        run(client, f"C={REMOTE_PKG}/bridge/clips; [ -e $C/zh-TW ] || ln -s ~/.medcare_reachy/clips/zh-TW $C/zh-TW")

        run(client, f"{PYTHON} -m pip install 'faster-whisper>=1.1,<2'", timeout=900)
        print("faster-whisper installed")

        run(client, f"mkdir -p ~/.medcare_reachy/models/stt && cd ~/.medcare_reachy/models/stt && "
                    f"([ -s silero_vad.onnx ] || curl -sSL -o silero_vad.onnx {VAD_URL}) && ls -la silero_vad.onnx")
        print("voice activity model ready")

        run(client, f"{PYTHON} -c 'from pathlib import Path; from faster_whisper import WhisperModel; "
                    "WhisperModel(\"base\", device=\"cpu\", compute_type=\"int8\", cpu_threads=2, "
                    "download_root=str(Path.home() / \".medcare_reachy/models/stt/whisper\"))'", timeout=900)
        print("Whisper Base downloaded from Hugging Face")

        print("clips to render:", ", ".join(stale) or "none")
        sftp = client.open_sftp()
        with sftp.file("/tmp/medcare_render_clips.py", "w") as handle:
            handle.write(RENDER)
        with sftp.file("/tmp/medcare_render_clips.json", "w") as handle:
            handle.write(json.dumps(job, ensure_ascii=False))
        sftp.close()
        output = run(client, f"nice -n 15 {PYTHON} /tmp/medcare_render_clips.py /tmp/medcare_render_clips.json 2>&1; "
                             f"echo render-exit=$?", timeout=900)
        lines = output.splitlines()
        print("\n".join(line for line in lines if line.startswith(("render ", "wrote ", "unrecognised ")))
              or "clips up to date")
        missing, misheard = render_problems(output, job)
        if missing or "render-exit=0" not in lines:
            print("\n".join(lines[-15:]))
            print(f"rendering the clips failed ({', '.join(missing) or 'see above'} not written): those keep their "
                  "old WAVs, and the next deploy renders them again")
            if unrecognised(output):
                print(f"ERROR clips: {', '.join(unrecognised(output))} came back from Whisper as words the app "
                      "would not recognise as Reachy's own echo (voice.without_filler), so they were not written and "
                      "are never played. Add what Whisper heard (above) to THINKING_PHRASES/_SOUND_ALIKES in "
                      "voice.py, or change the phrase.")
            return 1
        if misheard:
            print(f"WARNING clips: Whisper did not hear {', '.join(misheard)} exactly as written (the closest of "
                  "three speeds was kept). Listen to them on the robot before relying on them.")
        sftp = client.open_sftp()
        try:
            absent = absent_clips(manifest, wav_clips(sftp))
        finally:
            sftp.close()
        if absent:
            print(f"ERROR clips: no WAV on the robot for {', '.join(absent)}. The app skips a missing thinking phrase "
                  "(and says only its 嗯 when all are missing) and stays silent for any other missing clip. Rerun "
                  "this script.")
            return 1

        check_camera_rate(client, args.camera_ipc_fps, stamp)

        update = '{"capture_fps": 15, "vision_on_server": true}'
        answer = run(client, f"curl -s -m 30 -X POST localhost:8042/api/config "
                             f"-H 'Content-Type: application/json' -d '{update}' || true")
        if not answer.strip():
            print("the medcare_reachy app is not running: start it in Reachy Mini Control, then rerun this script")
            return 1
        print("settings:", answer.strip()[:300])
        time.sleep(10)
        print("status:", run(client, "curl -s -m 10 localhost:8042/api/status").strip())
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
