"""Job queue + checkpoint/resume contract tests.

The resume guarantee is the whole point of M0: kill anywhere, re-run, and
only missing/stale work repeats. These tests exercise that contract without
any media."""

import json

import pytest

from publikclip_pipeline import config
from publikclip_pipeline.jobs import queue


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path / "home"))
    yield


def _settings_json() -> str:
    return json.dumps(config.Settings().to_json())


class CountingStage(queue.Stage):
    name = "counting"
    schema_version = 1

    def __init__(self):
        self.runs = 0

    def run(self, ctx):
        self.runs += 1
        return {"runs": self.runs}


class FailingStage(queue.Stage):
    name = "failing"
    schema_version = 1

    def run(self, ctx):
        raise queue.StageError("boom, but politely")


class ArtifactStage(queue.Stage):
    name = "artifact"
    schema_version = 1

    def __init__(self):
        self.runs = 0

    def run(self, ctx):
        self.runs += 1
        out = ctx.job_dir / "artifact.bin"
        out.write_bytes(b"data")
        return {"path": str(out)}

    def artifacts_ok(self, ctx, data):
        from pathlib import Path

        return Path(data["path"]).exists()


def _noop_progress(stage, fraction, message):
    pass


def test_create_and_get_job():
    job = queue.create_job("file", "/tmp/x.mp4", _settings_json())
    fetched = queue.get_job(job.id)
    assert fetched is not None
    assert fetched.source == "/tmp/x.mp4"
    assert job.dir.exists()
    assert (job.dir / "settings.json").exists()


def test_stage_runs_once_then_caches():
    job = queue.create_job("file", "/tmp/x.mp4", _settings_json())
    stage = CountingStage()
    queue.run_stages(job, [stage], _noop_progress)
    queue.run_stages(job, [stage], _noop_progress)
    assert stage.runs == 1  # second run served from checkpoint


def test_schema_version_bump_invalidates():
    job = queue.create_job("file", "/tmp/x.mp4", _settings_json())
    stage = CountingStage()
    queue.run_stages(job, [stage], _noop_progress)
    stage.schema_version = 2
    queue.run_stages(job, [stage], _noop_progress)
    assert stage.runs == 2


def test_missing_artifact_invalidates_checkpoint():
    job = queue.create_job("file", "/tmp/x.mp4", _settings_json())
    stage = ArtifactStage()
    queue.run_stages(job, [stage], _noop_progress)
    (job.dir / "artifact.bin").unlink()
    queue.run_stages(job, [stage], _noop_progress)
    assert stage.runs == 2


def test_corrupt_checkpoint_reruns():
    job = queue.create_job("file", "/tmp/x.mp4", _settings_json())
    stage = CountingStage()
    queue.run_stages(job, [stage], _noop_progress)
    queue.checkpoint_path(job, stage.name).write_text("{not json")
    queue.run_stages(job, [stage], _noop_progress)
    assert stage.runs == 2


def test_stage_error_marks_job_failed():
    job = queue.create_job("file", "/tmp/x.mp4", _settings_json())
    with pytest.raises(queue.StageError):
        queue.run_stages(job, [FailingStage()], _noop_progress)
    fetched = queue.get_job(job.id)
    assert fetched.status == "failed"
    assert "politely" in (fetched.error or "")


def test_failure_then_resume_skips_completed_stages():
    job = queue.create_job("file", "/tmp/x.mp4", _settings_json())
    counting = CountingStage()
    with pytest.raises(queue.StageError):
        queue.run_stages(job, [counting, FailingStage()], _noop_progress)
    assert counting.runs == 1

    class FixedStage(queue.Stage):
        name = "failing"  # same name — simulates the bug being fixed
        schema_version = 1

        def run(self, ctx):
            return {"ok": True}

    results = queue.run_stages(job, [counting, FixedStage()], _noop_progress)
    assert counting.runs == 1  # not re-run
    assert results["failing"] == {"ok": True}
    assert queue.get_job(job.id).status == "done"


# --- legacy encodings ------------------------------------------------------


def test_a_checkpoint_written_before_utf8_was_explicit_still_loads(tmp_path, monkeypatch):
    """Jobs created by an earlier build carry checkpoints in the Windows
    locale encoding. Reading those strictly turned an upgrade into a crash on
    somebody's half-finished work."""
    from publikclip_pipeline import config

    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    config.ensure_home()
    job = queue.create_job("url", "https://x.invalid/a", "{}")
    path = queue.checkpoint_path(job, "asr")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = '{"schema_version": 1, "data": {"title": "écoles françaises — 90 %"}}'
    path.write_bytes(payload.encode("cp1252"))

    data = queue.read_checkpoint(job, "asr", 1)
    assert data is not None, "a legacy checkpoint was rejected"
    assert data["title"] == "écoles françaises — 90 %", "accents were mangled"


def test_reading_a_legacy_checkpoint_repairs_it(tmp_path, monkeypatch):
    """Self-healing rather than tolerated forever: after one read the file is
    valid UTF-8 for every other reader in the pipeline."""
    from publikclip_pipeline import config

    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    config.ensure_home()
    job = queue.create_job("url", "https://x.invalid/b", "{}")
    path = queue.checkpoint_path(job, "asr")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes('{"schema_version": 1, "data": {"t": "café"}}'.encode("cp1252"))

    queue.read_checkpoint(job, "asr", 1)
    assert json.loads(path.read_text(encoding="utf-8"))["data"]["t"] == "café"


def test_utf8_checkpoints_are_untouched(tmp_path, monkeypatch):
    from publikclip_pipeline import config

    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    config.ensure_home()
    job = queue.create_job("url", "https://x.invalid/c", "{}")
    path = queue.checkpoint_path(job, "asr")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = '{"schema_version": 1, "data": {"t": "déjà en utf-8"}}'
    path.write_text(payload, encoding="utf-8")
    before = path.read_bytes()

    assert queue.read_checkpoint(job, "asr", 1)["t"] == "déjà en utf-8"
    assert path.read_bytes() == before, "a valid file was rewritten for nothing"
