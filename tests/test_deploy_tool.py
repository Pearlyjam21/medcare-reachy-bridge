"""tools/deploy_to_robot.py: which voice clips a deploy renders again, and the words it gives the voice."""

import copy
import importlib.util
import io
import json
from pathlib import Path

import pytest

from medcare_reachy.bridge.clips import load_manifest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "deploy_to_robot.py"
spec = importlib.util.spec_from_file_location("deploy_to_robot", TOOL)
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def letters(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def test_every_clip_has_simplified_words_written_for_its_current_text():
    manifest = load_manifest()
    assert set(deploy.SPOKEN) == set(manifest["clips"])
    for clip_id, (written_for, spoken) in deploy.SPOKEN.items():
        assert written_for == manifest["clips"][clip_id]["zh-TW"], f"update SPOKEN[{clip_id!r}] in the deploy tool"
        assert len(letters(spoken)) == len(letters(written_for)), clip_id   # the same sentence, character by character
        assert deploy.spoken_text(clip_id, manifest) == spoken


def test_a_changed_clip_is_rendered_on_a_robot_set_up_before_the_render_record():
    manifest = load_manifest()
    installed = copy.deepcopy(manifest)   # the app on the robot before this deploy, with the old wording
    installed["clips"]["confirm_with_caregiver"]["zh-TW"] = "謝謝您。我會請家人確認這種藥。"
    on_robot = set(manifest["clips"])
    rendered, stale = deploy.plan_clips(manifest, None, installed, on_robot)
    assert stale == ["confirm_with_caregiver"]
    assert rendered["confirm_with_caregiver"] == "謝謝您。我會請家人確認這種藥。"
    assert rendered["say_when_done"] == manifest["clips"]["say_when_done"]["zh-TW"]   # not rendered again


def test_the_render_record_decides_once_it_exists():
    manifest = load_manifest()
    record = {clip_id: entry["zh-TW"] for clip_id, entry in manifest["clips"].items()}
    on_robot = set(manifest["clips"])
    assert deploy.plan_clips(manifest, record, None, on_robot)[1] == []
    changed = copy.deepcopy(manifest)
    changed["clips"]["help"]["zh-TW"] = "需要幫忙嗎？"
    assert deploy.plan_clips(changed, record, None, on_robot)[1] == ["help"]
    assert deploy.plan_clips(manifest, record, None, on_robot - {"thanks"})[1] == ["thanks"]   # WAV missing


def test_a_manifest_edit_without_new_simplified_words_stops_the_deploy():
    manifest = copy.deepcopy(load_manifest())
    manifest["clips"]["help"]["zh-TW"] = "需要幫忙嗎？"
    with pytest.raises(SystemExit, match="help"):
        deploy.spoken_text("help", manifest)


def test_render_script_on_the_robot_compiles():
    compile(deploy.RENDER, "render_clips.py", "exec")


class RobotFiles:
    """The SFTP calls the deploy makes, on files kept in a dict (paths as the deploy writes them)."""

    def __init__(self, files: dict):
        self.files = dict(files)

    def open(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return io.BytesIO(self.files[path].encode("utf-8"))

    def file(self, path, mode):
        robot = self

        class Handle(io.StringIO):
            def close(self):
                robot.files[path] = self.getvalue()
                super().close()
        return Handle()

    def posix_rename(self, old, new):
        self.files[new] = self.files.pop(old)

    def listdir(self, folder):
        return [path.rsplit("/", 1)[1] for path in self.files if path.startswith(folder + "/")]


def test_a_render_that_fails_after_the_upload_is_retried_by_the_next_deploy():
    """The robot had no render record; deploy 1 uploaded the new manifest, then its render failed."""
    manifest = load_manifest()
    installed = copy.deepcopy(manifest)
    installed["clips"]["confirm_with_caregiver"]["zh-TW"] = "謝謝您。我會請家人確認這種藥。"
    installed_path = f"{deploy.REMOTE_PKG}/bridge/clips/manifest.json"
    robot = RobotFiles({installed_path: json.dumps(installed, ensure_ascii=False),
                        **{f"{deploy.CLIPS_DIR}/{clip_id}.wav": "" for clip_id in manifest["clips"]}})

    first = deploy.prepare_clips(robot, manifest)
    assert [clip["id"] for clip in first["clips"]] == ["confirm_with_caregiver"]
    record = json.loads(robot.files[f"{deploy.CLIPS_DIR}/{deploy.RENDERED}"])
    assert record["confirm_with_caregiver"] == "謝謝您。我會請家人確認這種藥。"   # what the WAV still says

    robot.files[installed_path] = json.dumps(manifest, ensure_ascii=False)        # the upload; no render followed
    second = deploy.prepare_clips(robot, manifest)
    assert [clip["id"] for clip in second["clips"]] == ["confirm_with_caregiver"]


def test_a_deploy_fails_unless_every_planned_clip_was_written():
    job = {"clips": [{"id": "confirm_with_caregiver"}, {"id": "help"}]}
    good = "render help speed=0.85 heard='x' ok=True\nwrote confirm_with_caregiver heard-ok\nwrote help heard-ok\n"
    assert deploy.render_problems(good + "render-exit=0", job) == ([], [])
    # the second clip never written (the script died), or the script killed after writing both (exit 137)
    assert deploy.render_problems("wrote confirm_with_caregiver heard-ok\nKilled\nrender-exit=137", job)[0] == [
        "confirm_with_caregiver", "help"]
    assert deploy.render_problems(good + "render-exit=137", job)[0] == ["confirm_with_caregiver", "help"]
    assert deploy.render_problems("wrote help heard-ok\nrender-exit=0", job)[0] == ["confirm_with_caregiver"]
    misheard = "wrote confirm_with_caregiver not-heard-as-written\nwrote help heard-ok\nrender-exit=0"
    assert deploy.render_problems(misheard, job) == ([], ["confirm_with_caregiver"])


def test_the_render_script_reports_in_the_words_the_deploy_reads():
    assert '"wrote", clip["id"], "heard-ok"' in deploy.RENDER and '"not-heard-as-written"' in deploy.RENDER
