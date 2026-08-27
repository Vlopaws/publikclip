"""What discovery hands to the rest of the pipeline.

Discovery answers "what is worth clipping" and stops there — it never
downloads. One normalised item type so the ingest stage, the CLI and any
automation on top stay indifferent to whether a candidate came from a
YouTube channel listing or a Twitch clip page.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..jobs import queue


@dataclass(frozen=True)
class SourceItem:
    """One candidate video, before anything has been fetched."""

    id: str
    url: str
    title: str
    source: str                     # "youtube" | "twitch"
    duration_sec: float | None = None
    view_count: int | None = None
    channel: str | None = None
    raw: dict = field(default_factory=dict, repr=False)

    def summary(self) -> str:
        bits = [self.title[:60]]
        if self.duration_sec:
            mins, secs = divmod(int(self.duration_sec), 60)
            bits.append(f"{mins}:{secs:02d}")
        if self.view_count is not None:
            bits.append(f"{self.view_count:,} views".replace(",", " "))
        return "  ".join(bits)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "source": self.source,
            "duration_sec": self.duration_sec,
            "view_count": self.view_count,
            "channel": self.channel,
        }


def within_duration(
    items: Iterable[SourceItem],
    min_sec: float | None,
    max_sec: float | None,
) -> list[SourceItem]:
    """Drop what is not worth the compute.

    Both ends matter and for opposite reasons: below the floor there is not
    enough material to cut a clip from, and above the ceiling a single job
    can spend hours of transcription and tens of LLM calls on one source.
    Items with an unknown duration are kept — flat listings occasionally
    omit it, and it is better to let ingest decide than to silently skip.
    """
    kept = []
    for item in items:
        if item.duration_sec is None:
            kept.append(item)
            continue
        if min_sec is not None and item.duration_sec < min_sec:
            continue
        if max_sec is not None and item.duration_sec > max_sec:
            continue
        kept.append(item)
    return kept


def already_processed(items: Iterable[SourceItem], limit: int = 500) -> set[str]:
    """URLs the job queue has already seen.

    Matching is on the queue's stored `source` string, which is the URL the
    job was created with — the same one discovery emits.
    """
    seen = {job.source for job in queue.list_jobs(limit=limit)}
    return {item.url for item in items if item.url in seen}


def unseen(items: Iterable[SourceItem], limit: int = 500) -> list[SourceItem]:
    """Discovery run twice a day would otherwise re-clip yesterday's videos."""
    items = list(items)
    done = already_processed(items, limit=limit)
    return [item for item in items if item.url not in done]
