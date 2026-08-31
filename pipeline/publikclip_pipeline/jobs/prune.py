"""Reclaim what a finished job no longer needs.

Nothing in this pipeline ever deleted anything. That is defensible while a
person is driving it — the source is right there when you want to re-cut a
clip — and indefensible the moment it runs unattended, which is what the
autopilot is for. One two-hour interview leaves 905 MB behind: a 720 MB
download plus a 217 MB analysis WAV. A nightly run fills a 20 GB cloud disk
in under three weeks, and the pipeline then fails on something that reads
like a bug in whatever stage happened to be writing at the time.

What goes: the source video, the analysis WAV, the speaker embeddings, the
T2 frame grabs, and any upload temporaries left by a crash. All of it is
derived from a URL that is still recorded in the job.

What stays, always: the rendered clips, every stage checkpoint, and the
trajectories. Those are the output and the audit trail — the evidence for
why a clip was cut where it was — and they are small.

Pruning is deliberate. It reports by default and only deletes when asked,
because "free up space" and "throw away an hour of compute" are the same
command run against the wrong job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import queue

# Heavy, and reproducible from the source URL. Ordered roughly by size.
DISPOSABLE_FILES = ("media.*", "audio16k.*", "*.flac", "diar_embeddings.npy")
DISPOSABLE_DIRS = ("t2frames",)

# Never removed, whatever else matches: the clips are the product, and the
# checkpoints are what makes a job auditable and resumable.
PROTECTED = ("clips", "*.json")

# A job younger than this is probably still interesting to whoever ran it.
DEFAULT_MIN_AGE_DAYS = 3.0


@dataclass
class JobPrune:
    job_id: str
    title: str
    status: str
    age_days: float
    paths: list[Path] = field(default_factory=list)
    bytes_freed: int = 0

    def to_json(self) -> dict:
        return {
            "job_id": self.job_id,
            "title": self.title,
            "status": self.status,
            "age_days": round(self.age_days, 1),
            "files": [p.name for p in self.paths],
            "bytes_freed": self.bytes_freed,
        }


@dataclass
class PruneReport:
    applied: bool
    jobs: list[JobPrune] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def bytes_freed(self) -> int:
        return sum(j.bytes_freed for j in self.jobs)

    def to_json(self) -> dict:
        return {
            "applied": self.applied,
            "bytes_freed": self.bytes_freed,
            "jobs": [j.to_json() for j in self.jobs],
            "skipped": self.skipped,
        }


def _size(path: Path) -> int:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    try:
        return path.stat().st_size
    except OSError:
        return 0


def disposable_paths(job_dir: Path) -> list[Path]:
    """Everything in one job directory that can be regenerated.

    Deduplicated, because the patterns overlap by design: an upload
    temporary is caught by both `audio16k.*` and `*.flac`. Deleting it twice
    is harmless, but counting it twice makes the report claim more space
    than it can free — and the report is the whole basis for deciding
    whether to run this at all.
    """
    found: list[Path] = []
    for pattern in DISPOSABLE_FILES:
        found.extend(p for p in job_dir.glob(pattern) if p.is_file())
    for name in DISPOSABLE_DIRS:
        target = job_dir / name
        if target.is_dir():
            found.append(target)

    # Belt and braces: a pattern that grew to overlap the protected set
    # would silently delete the product, so exclude by name here too.
    unique: list[Path] = []
    seen: set[Path] = set()
    for p in found:
        if p.name == "clips" or p.suffix == ".json" or p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def plan(
    min_age_days: float = DEFAULT_MIN_AGE_DAYS,
    job_id: str | None = None,
    now: float | None = None,
) -> PruneReport:
    """What would be freed, and from which jobs. Deletes nothing."""
    now = time.time() if now is None else now
    report = PruneReport(applied=False)

    for job in queue.list_jobs(limit=500):
        if job_id and job.id != job_id:
            continue
        age_days = (now - job.created_at) / 86400.0
        if not job_id:
            # An explicit job id is an explicit decision; a sweep is not.
            if job.status == "running":
                report.skipped.append(f"{job.id}: still running")
                continue
            if age_days < min_age_days:
                report.skipped.append(f"{job.id}: {age_days:.1f} days old")
                continue
        if not job.dir.exists():
            continue

        paths = disposable_paths(job.dir)
        if not paths:
            continue
        entry = JobPrune(
            job_id=job.id,
            title=job.title or job.source,
            status=job.status,
            age_days=age_days,
            paths=paths,
            bytes_freed=sum(_size(p) for p in paths),
        )
        report.jobs.append(entry)
    return report


def apply(report: PruneReport) -> PruneReport:
    """Delete what `plan` found. Returns the report, marked applied."""
    import shutil

    for entry in report.jobs:
        for path in entry.paths:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
    report.applied = True
    return report
