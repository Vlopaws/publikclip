"""Not re-deriving scene changes that are already on disk.

Ten minutes on a seventy-minute video, for a deterministic answer the last
run of this stage already wrote to scenes.json. Re-running the stage — which
is what changing any candidate setting does — paid it again.
"""

from __future__ import annotations

import json

import pytest

from publikclip_pipeline.candidates import stage as candidates_stage


class Ctx:
    """Just enough of StageContext for the scene block."""

    def __init__(self, job_dir, settings=None):
        self.job_dir = job_dir
        self.settings = settings or type("S", (), {"focus": []})()
        self.messages: list[str] = []
        self.prior = {}

    def emit(self, fraction, message, stage=""):
        self.messages.append(message)


def run_scene_block(ctx, detect):
    """The stage's scene resolution, isolated from the rest of the run.

    Mirrors the source so the test exercises the real decision rather than
    a description of it.
    """
    scenes_path = ctx.job_dir / "scenes.json"
    scene_times: list[float] = []
    if scenes_path.exists():
        try:
            cached = json.loads(scenes_path.read_text(encoding="utf-8"))
            if isinstance(cached, list) and cached:
                scene_times = [float(t) for t in cached]
                ctx.emit(-1, f"Scene changes: {len(scene_times)} (cached)")
        except (json.JSONDecodeError, TypeError, ValueError):
            scene_times = []
    if not scene_times:
        ctx.emit(-1, "Detecting scene changes…")
        try:
            scene_times = detect()
        except Exception:  # noqa: BLE001
            scene_times = []
        scenes_path.write_text(json.dumps(scene_times), encoding="utf-8")
    return scene_times


def test_a_previous_result_is_reused(tmp_path):
    (tmp_path / "scenes.json").write_text("[1.0, 2.5, 90.0]")
    ctx = Ctx(tmp_path)
    got = run_scene_block(ctx, lambda: pytest.fail("re-detected a cached answer"))
    assert got == [1.0, 2.5, 90.0]
    assert any("cached" in m for m in ctx.messages)


def test_nothing_cached_means_detect(tmp_path):
    ctx = Ctx(tmp_path)
    assert run_scene_block(ctx, lambda: [4.0]) == [4.0]
    assert json.loads((tmp_path / "scenes.json").read_text()) == [4.0]


def test_an_empty_cache_is_not_reused(tmp_path):
    """An empty list is also what a FAILED detection writes.

    Reusing it would disable the channel for the life of the job, silently
    and with nothing to say why.
    """
    (tmp_path / "scenes.json").write_text("[]")
    ctx = Ctx(tmp_path)
    assert run_scene_block(ctx, lambda: [7.0]) == [7.0]


def test_a_corrupt_cache_falls_back_to_detecting(tmp_path):
    (tmp_path / "scenes.json").write_text("{not json")
    ctx = Ctx(tmp_path)
    assert run_scene_block(ctx, lambda: [3.0]) == [3.0]


def test_a_detection_failure_is_not_fatal(tmp_path):
    """Scenes are one channel of several; losing them degrades, not stops."""
    ctx = Ctx(tmp_path)

    def boom():
        raise RuntimeError("scenedetect exploded")

    assert run_scene_block(ctx, boom) == []


def test_the_stage_still_writes_scenes_json(tmp_path):
    ctx = Ctx(tmp_path)
    run_scene_block(ctx, lambda: [1.5])
    assert (tmp_path / "scenes.json").exists()


def test_the_source_reads_the_cache_before_detecting():
    """Guard against the isolated copy above drifting from the real stage."""
    import inspect

    source = inspect.getsource(candidates_stage.CandidatesStage.run)
    assert "scenes.json" in source
    assert source.index("scenes_path.exists()") < source.index("detect_scenes(")
