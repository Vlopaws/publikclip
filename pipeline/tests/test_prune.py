"""Reclaiming disk from finished jobs.

Two ways to get this wrong, and only one of them is loud. Deleting too
little fills a disk, which at least announces itself. Deleting too much
throws away rendered clips or the checkpoints that explain them, and looks
like success.
"""

from __future__ import annotations

import time

import pytest

from publikclip_pipeline.jobs import prune


def make_job(root, job_id, *, media_mb=2, with_clips=True):
    d = root / job_id
    (d / "clips").mkdir(parents=True)
    (d / "t2frames").mkdir()
    (d / "media.mp4").write_bytes(b"0" * media_mb * 1024 * 1024)
    (d / "audio16k.wav").write_bytes(b"0" * 1024)
    (d / "audio16k.upload.flac").write_bytes(b"0" * 512)
    (d / "diar_embeddings.npy").write_bytes(b"0" * 256)
    (d / "t2frames" / "f0.jpg").write_bytes(b"0" * 64)
    (d / "score.json").write_text("{}")
    (d / "render.json").write_text("{}")
    (d / "trajectory_00.json").write_text("{}")
    if with_clips:
        (d / "clips" / "clip_00.mp4").write_bytes(b"0" * 4096)
        (d / "clips" / "clip_00.ass").write_text("x")
    return d


# --- what is chosen ------------------------------------------------------

def test_the_heavy_regenerable_files_are_chosen(tmp_path):
    d = make_job(tmp_path, "j1")
    names = {p.name for p in prune.disposable_paths(d)}
    assert names == {
        "media.mp4",
        "audio16k.wav",
        "audio16k.upload.flac",
        "diar_embeddings.npy",
        "t2frames",
    }


def test_the_clips_are_never_chosen(tmp_path):
    d = make_job(tmp_path, "j1")
    chosen = prune.disposable_paths(d)
    assert all(p.name != "clips" for p in chosen)
    assert all("clip_00" not in str(p) for p in chosen)


def test_no_checkpoint_is_ever_chosen(tmp_path):
    d = make_job(tmp_path, "j1")
    assert all(p.suffix != ".json" for p in prune.disposable_paths(d))


def test_overlapping_patterns_do_not_double_count(tmp_path):
    """`audio16k.upload.flac` matches both `audio16k.*` and `*.flac`.

    Listing it twice would make the report promise twice the space it can
    actually free, and the report is what the operator decides on.
    """
    d = make_job(tmp_path, "j1")
    chosen = prune.disposable_paths(d)
    assert len(chosen) == len(set(chosen))
    counted = sum(prune._size(p) for p in chosen)
    on_disk = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    assert counted <= on_disk, "reported more bytes than the job occupies"


def test_an_already_pruned_job_offers_nothing(tmp_path):
    d = make_job(tmp_path, "j1")
    for p in prune.disposable_paths(d):
        if p.is_dir():
            import shutil

            shutil.rmtree(p)
        else:
            p.unlink()
    assert prune.disposable_paths(d) == []


# --- what actually happens ----------------------------------------------

def test_applying_removes_the_source_and_keeps_the_product(tmp_path):
    d = make_job(tmp_path, "j1")
    report = prune.PruneReport(applied=False)
    report.jobs.append(
        prune.JobPrune("j1", "t", "done", 9.0, prune.disposable_paths(d), 0)
    )
    prune.apply(report)

    assert not (d / "media.mp4").exists()
    assert not (d / "audio16k.wav").exists()
    assert not (d / "t2frames").exists()
    # The product and the audit trail survive.
    assert (d / "clips" / "clip_00.mp4").exists()
    assert (d / "clips" / "clip_00.ass").exists()
    assert (d / "score.json").exists()
    assert (d / "trajectory_00.json").exists()
    assert report.applied


def test_planning_deletes_nothing(tmp_path, monkeypatch):
    d = make_job(tmp_path, "j1")
    _fake_queue(monkeypatch, tmp_path, [("j1", "done", 9.0)])
    report = prune.plan()
    assert report.applied is False
    assert (d / "media.mp4").exists(), "a dry run removed a file"
    assert report.bytes_freed > 2_000_000


# --- who is left alone ---------------------------------------------------

def test_a_running_job_is_never_swept(tmp_path, monkeypatch):
    # Its media is being read right now; taking it would break the run and
    # look like a pipeline bug.
    make_job(tmp_path, "j1")
    _fake_queue(monkeypatch, tmp_path, [("j1", "running", 30.0)])
    report = prune.plan()
    assert report.jobs == []
    assert any("still running" in s for s in report.skipped)


def test_a_recent_job_is_left_alone(tmp_path, monkeypatch):
    make_job(tmp_path, "j1")
    _fake_queue(monkeypatch, tmp_path, [("j1", "done", 0.5)])
    assert prune.plan(min_age_days=3.0).jobs == []


def test_naming_a_job_overrides_both_guards(tmp_path, monkeypatch):
    # An explicit id is an explicit decision, including "yes, that one".
    make_job(tmp_path, "j1")
    _fake_queue(monkeypatch, tmp_path, [("j1", "running", 0.1)])
    report = prune.plan(job_id="j1")
    assert len(report.jobs) == 1


def test_a_job_directory_that_is_gone_is_not_an_error(tmp_path, monkeypatch):
    _fake_queue(monkeypatch, tmp_path, [("vanished", "done", 30.0)])
    assert prune.plan().jobs == []


def test_a_failed_job_is_swept_too(tmp_path, monkeypatch):
    # A failure leaves the same 900 MB behind as a success.
    make_job(tmp_path, "j1")
    _fake_queue(monkeypatch, tmp_path, [("j1", "failed", 9.0)])
    assert len(prune.plan().jobs) == 1


# --- helper --------------------------------------------------------------

def _fake_queue(monkeypatch, root, specs):
    """Stand in for the job table: (id, status, age_days) each."""
    now = time.time()

    class _Job:
        def __init__(self, job_id, status, age_days):
            self.id = job_id
            self.status = status
            self.created_at = now - age_days * 86400
            self.title = job_id
            self.source = job_id

        @property
        def dir(self):
            return root / self.id

    monkeypatch.setattr(
        prune.queue, "list_jobs", lambda limit=500: [_Job(*s) for s in specs]
    )
