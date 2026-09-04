"""Which clips out of a finished job are worth posting.

The scoring stage already ranks and audits; this only decides how many of
the top to take and what to refuse outright. Kept separate from the runner
because it is the one piece worth arguing about, and the one that should be
easy to test without running an hour of pipeline.

A floor rather than "always take the top 3" is deliberate: a bad source
video produces three bad clips, and posting them is worse than posting
nothing. The rubric says 8+ on any dimension should be rare, so the default
floor sits below that but above the mediocre middle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Composite score below which a clip is not worth an audience's attention.
#
# The scale is 0-100: rubric.composite averages the normalised subscores,
# the interest curve and the visual pass, then multiplies by 100.
#
# This value has been wrong twice, in opposite directions, and both times
# the argument for it was arithmetic rather than measurement.
#
# It began at 5.5, reading a 0-100 number as though it were 0-10. Nothing
# failed: the floor simply never rejected anything, and an unattended run
# would have published every clip it rendered.
#
# It then became 50 — "half marks on the weighted rubric", which is sound
# reasoning about a scale nobody had looked at. Across 24 clips from two
# creators the composite actually runs 22.9 to 52.9, median 35.8. A floor
# at 50 keeps 8% of everything ever rendered here, and on a two-hour
# Thinkerview interview it kept nothing at all: the autopilot ran the whole
# pipeline and published zero clips, which is the same practical outcome as
# not running.
#
# 40 is the third quartile of what has actually been measured. It rejects
# three clips in four, and it makes the floor and `--clips 3` agree instead
# of fight: twelve rendered, about three survive, three get published.
#
# Still a design value. The honest version of this number comes from the
# Instagram feedback loop (decision #13), which fits the cross-validation
# constants from real outcomes and can fit this one the same way. Until a
# batch has actually been posted, treat it as a starting point and watch
# what the first one publishes.
DEFAULT_MIN_SCORE = 40.0

# A caller passing something on the 0-10 scale means the units mistake
# above, not a deliberately permissive run — 8 would be the third percentile
# here, which nobody asks for on purpose.
_SUSPICIOUS_SCALE = 10.0

# Vertical clips much past a minute lose completion rate on every platform.
DEFAULT_MAX_DURATION = 90.0


@dataclass(frozen=True)
class SelectedClip:
    job_id: str
    clip: int
    path: Path
    score: float
    best_platform: str
    duration: float
    summary: str
    confidence: str
    # Written by the render stage from this clip's own transcript: the
    # headline burned onto the video, a caption for the post, and tags.
    # They were being generated and then dropped here, so a published post
    # fell back to the scorer's one-line summary — which describes the clip
    # for an operator reading a report, not for someone scrolling.
    title: str = ""
    description: str = ""
    hashtags: tuple[str, ...] = ()
    # The clip's own words, kept so an unattended run can judge what it is
    # about before publishing it. See autopilot.review.
    transcript: str = ""

    def caption(self, limit: int = 180) -> str:
        """What the post should say, best available first.

        The render stage's description is written to be read by an audience;
        the scorer's summary is written to be read by whoever is deciding
        whether to publish. Prefer the first and fall back to the second.

        No hashtag padding and no invented hype either way: everything here
        was generated from the transcript, so it stays recognisable as a
        description of what actually happens.
        """
        text = " ".join((self.description or self.summary).split())
        if len(text) <= limit:
            return text
        return text[: limit - 1].rsplit(" ", 1)[0] + "…"

    def to_json(self) -> dict:
        return {
            "job_id": self.job_id,
            "clip": self.clip,
            "path": str(self.path),
            "score": self.score,
            "best_platform": self.best_platform,
            "duration": self.duration,
            "summary": self.summary,
            "confidence": self.confidence,
            "title": self.title,
            "description": self.description,
            "hashtags": list(self.hashtags),
        }


def _read(job_dir: Path, stage: str) -> dict:
    path = job_dir / f"{stage}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("data") or {}
    except (json.JSONDecodeError, OSError):
        return {}


def select(
    job_id: str,
    job_dir: Path,
    *,
    take: int = 3,
    min_score: float = DEFAULT_MIN_SCORE,
    max_duration: float | None = DEFAULT_MAX_DURATION,
) -> list[SelectedClip]:
    """The best `take` rendered clips that clear the floor, best first.

    Reads the job's artifacts rather than taking them as arguments — the
    artifacts are the source of truth in this pipeline, and a resumed job
    should select identically to a fresh one.
    """
    if 0 < min_score < _SUSPICIOUS_SCALE:
        raise ValueError(
            f"min_score={min_score} looks like a 0-10 value, but composite "
            f"scores run 0-100 (see rubric.composite). Use ~{min_score * 10:.0f} "
            "to mean the same thing, or 0 to disable the floor."
        )

    render = _read(job_dir, "render")
    score = _read(job_dir, "score")
    # render.stage enumerates score.json's clips, so output["clip"] is the
    # positional index into that same list.
    scored_clips = score.get("clips") or []
    confidence = score.get("confidence") or "unknown"

    chosen: list[SelectedClip] = []
    for output in render.get("outputs") or []:
        path = Path(output["path"])
        if not path.exists():
            continue  # rendered then moved or cleaned up
        if output.get("score") is None or output["score"] < min_score:
            continue
        duration = output.get("duration") or 0.0
        if max_duration is not None and duration > max_duration:
            continue
        index = output.get("clip", 0)
        meta = scored_clips[index] if 0 <= index < len(scored_clips) else {}
        chosen.append(
            SelectedClip(
                job_id=job_id,
                clip=index,
                path=path,
                score=float(output["score"]),
                best_platform=output.get("best_platform") or "reels",
                duration=float(duration),
                summary=meta.get("summary") or "",
                confidence=confidence,
                title=output.get("title") or "",
                description=output.get("description") or "",
                hashtags=tuple(output.get("hashtags") or ()),
                transcript=output.get("transcript") or "",
            )
        )

    chosen.sort(key=lambda c: c.score, reverse=True)
    return chosen[:take]
