"""Discover → process → select → publish, unattended.

The loop is deliberately boring; the interesting decisions live in the
modules it calls. What this owns is the failure policy, and the policy is
that one bad source must not take the batch down with it: a video that fails
to ingest, a scoring stage that runs out of quota, a platform that refuses a
post — each is recorded against that candidate and the run continues.

Nothing here is clever about *when* to run. Point cron, a systemd timer or
Task Scheduler at it; the job queue is what stops the same video being
processed twice.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .. import config, stages
from ..jobs import queue
from ..sources import SourceItem, unseen
from . import publish as publish_mod
from . import review
from .select import DEFAULT_MAX_DURATION, DEFAULT_MIN_SCORE, SelectedClip, select


@dataclass
class CandidateOutcome:
    item: SourceItem
    job_id: str | None = None
    selected: list[SelectedClip] = field(default_factory=list)
    held: list[dict] = field(default_factory=list)
    published: list[publish_mod.PublishResult] = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> dict:
        return {
            "source": self.item.to_json(),
            "job_id": self.job_id,
            "selected": [c.to_json() for c in self.selected],
            "held": self.held,
            "published": [p.to_json() for p in self.published],
            "error": self.error,
        }


@dataclass
class RunReport:
    started_at: float
    discovered: int = 0
    outcomes: list[CandidateOutcome] = field(default_factory=list)
    publisher: str = "dry-run"

    @property
    def clips_selected(self) -> int:
        return sum(len(o.selected) for o in self.outcomes)

    @property
    def posts_ok(self) -> int:
        return sum(1 for o in self.outcomes for p in o.published if p.ok)

    @property
    def failures(self) -> int:
        return sum(1 for o in self.outcomes if o.error)

    def to_json(self) -> dict:
        return {
            "started_at": self.started_at,
            "publisher": self.publisher,
            "discovered": self.discovered,
            "clips_selected": self.clips_selected,
            "clips_held": sum(len(o.held) for o in self.outcomes),
            "posts_ok": self.posts_ok,
            "failures": self.failures,
            "candidates": [o.to_json() for o in self.outcomes],
        }


# The selection floor rejects clips after they are rendered, so rendering
# exactly `clips_per_video` would leave nothing to fall back on when one of
# them scores badly. Twice the target, floored at six, keeps a real choice
# without paying for twelve.
FINALIST_HEADROOM = 2
MIN_FINALISTS = 6


def _finalist_cap(clips_per_video: int) -> int:
    return max(MIN_FINALISTS, clips_per_video * FINALIST_HEADROOM)


def _process(
    item: SourceItem,
    llm_mode: str,
    captions: str | None,
    emit,
    max_finalists: int | None = None,
) -> str:
    """Run the full pipeline for one candidate; returns its job id."""
    settings = config.Settings()
    settings.llm_mode = llm_mode
    settings.max_finalists = max_finalists
    if captions:
        settings.caption_preset = captions
    source_type = "url" if item.url.startswith(("http://", "https://")) else "file"
    job = queue.create_job(source_type, item.url, json.dumps(settings.to_json()))
    queue.run_stages(job, stages.default_stages(), emit)
    return job.id


def run(
    candidates: list[SourceItem],
    *,
    publisher=None,
    platforms: list[str] | None = None,
    clips_per_video: int = 3,
    min_score: float = DEFAULT_MIN_SCORE,
    max_clip_duration: float | None = DEFAULT_MAX_DURATION,
    llm_mode: str = "ollama",
    captions: str | None = None,
    skip_seen: bool = True,
    on_event=None,
) -> RunReport:
    """Process every candidate and publish what clears the bar."""
    publisher = publisher or publish_mod.DryRunPublisher()
    platforms = platforms or ["instagram"]
    note = on_event or (lambda kind, message: None)

    # Validate the destination first, and unconditionally — before the
    # expensive pipeline, and even when there is nothing to post. A run that
    # finds no new videos must still surface a dead connection, otherwise a
    # scheduled job reports success every night until the day it matters.
    publisher.check_ready(platforms)

    if skip_seen:
        candidates = unseen(candidates)

    report = RunReport(started_at=time.time(), publisher=publisher.name)
    report.discovered = len(candidates)
    if not candidates:
        return report

    for position, item in enumerate(candidates, start=1):
        note("candidate", f"[{position}/{len(candidates)}] {item.title[:60]}")
        outcome = CandidateOutcome(item=item)
        try:
            def emit(stage: str, fraction: float, message: str) -> None:
                note("stage", f"  {stage}: {message}")

            outcome.job_id = _process(
                item, llm_mode, captions, emit,
                max_finalists=_finalist_cap(clips_per_video),
            )
        except Exception as err:  # noqa: BLE001 - one bad source must not end the batch
            outcome.error = str(err)
            report.outcomes.append(outcome)
            note("error", f"  failed: {err}")
            continue

        job = queue.get_job(outcome.job_id)
        outcome.selected = select(
            outcome.job_id,
            job.dir,
            take=clips_per_video,
            min_score=min_score,
            max_duration=max_clip_duration,
        )
        note("selected", f"  {len(outcome.selected)} clip(s) above {min_score}")

        for clip in outcome.selected:
            # An unattended run has nobody to look at a clip before it goes
            # out. Hold anything that would risk the account rather than
            # publish it; it stays rendered and one command from posting.
            verdict = review.check(clip.transcript, clip.title, clip.description)
            if not verdict.ok:
                outcome.held.append(
                    {"clip": clip.clip, "score": clip.score, **verdict.to_json()}
                )
                note(
                    "hold",
                    f"  HELD clip {clip.clip} ({', '.join(verdict.reasons)}) — "
                    "rendered, not posted",
                )
                continue
            for platform in platforms:
                if publish_mod.already_posted(clip, platform):
                    note("skip", f"  clip {clip.clip} already on {platform}")
                    continue
                result = publisher.publish(clip, platform)
                publish_mod.record(result)
                outcome.published.append(result)
                verb = "would post" if result.dry_run else ("posted" if result.ok else "FAILED")
                note("publish", f"  {verb} clip {clip.clip} → {platform}: {clip.caption(60)}")

        report.outcomes.append(outcome)

    return report
