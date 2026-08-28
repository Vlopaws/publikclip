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
# the interest curve and the visual pass, then multiplies by 100. So 50 is
# not a round number picked to feel strict — it is literally half marks on
# the weighted rubric, and the rubric itself says 8-out-of-10 on any single
# dimension should be rare.
#
# The first version of this said 5.5, reading a 0-100 number as though it
# were 0-10. Nothing failed: the floor simply never rejected anything, and
# an unattended run would have published every clip it rendered, including
# the ones a human called bad. A filter that silently passes everything is
# worse than no filter, because it looks like one.
#
# Still a design value, to be fitted from real outcomes the way the
# cross-validation constants are.
DEFAULT_MIN_SCORE = 50.0

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

    def caption(self, limit: int = 180) -> str:
        """The clip's own one-line summary, which the scorer already wrote.

        No hashtag padding and no invented hype: whatever ships here was
        generated from the transcript, so it should stay recognisable as a
        description of what actually happens.
        """
        text = " ".join(self.summary.split())
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
            )
        )

    chosen.sort(key=lambda c: c.score, reverse=True)
    return chosen[:take]
