"""Many channels at once, ranked as one pool.

Built for an event: ZEvent puts fifty-odd streamers live for three days and
the clippable moments are scattered across all of them. Asking for "the top
three clips from Zerator" is the wrong question then — the right one is "the
top three clips from any of these fifty", and the answer changes hour to
hour.

So a roster is a list of channels the operator keeps, and discovery treats
it as a single pool: fetch every channel, merge, rank by view count, hand
back the head. Views are a crowd's judgement made before the pipeline ever
looks, and during a live event they are the freshest signal available.

Deliberately a file rather than a scrape. No public endpoint publishes the
participant list — four candidates were tried and all 404 — and an event
roster that breaks mid-event because someone changed a webpage is worse than
one that is simply typed out. A file is also the honest place for the
judgement calls: who is worth clipping is not a fact to be looked up.

Listing is parallel. Fifty sequential yt-dlp calls at three seconds each is
two and a half minutes before any work starts, which during a live event is
two and a half minutes of staleness.
"""

from __future__ import annotations

import concurrent.futures
import re
from dataclasses import dataclass, field
from pathlib import Path

from .common import SourceItem
from .twitch import DEFAULT_MAX_SEC, DEFAULT_MIN_SEC, channel_clips

# A roster of fifty is normal; more than this is likely a mistake — a whole
# file of something else, or a runaway generator — and fifty yt-dlp calls
# already take a while.
MAX_CHANNELS = 200

# Listing is network-bound, not CPU-bound, so the useful width is much
# wider than the core count. Kept modest anyway: this is somebody else's
# server being asked fifty questions at once.
LIST_WORKERS = 8

# Twitch login names: 4-25 chars, letters, digits and underscore.
_CHANNEL = re.compile(r"^[A-Za-z0-9_]{3,25}$")


@dataclass
class RosterResult:
    """What one sweep of a roster found, and what it could not reach."""

    items: list[SourceItem] = field(default_factory=list)
    reached: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def channels(self) -> int:
        return len(self.reached) + len(self.failed)

    def to_json(self) -> dict:
        return {
            "channels": self.channels,
            "reached": len(self.reached),
            "failed": self.failed,
            "clips": [item.to_json() for item in self.items],
        }


def parse(text: str) -> list[str]:
    """Channel names from a roster file.

    One per line. Blank lines and `#` comments are ignored, a full channel
    URL is reduced to its name, and duplicates are dropped while keeping the
    order the operator wrote — during an event that order is often priority.
    """
    names: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = line.rstrip("/").rsplit("/", 1)[-1].lstrip("@").strip()
        if not _CHANNEL.match(name):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names[:MAX_CHANNELS]


def load(path: str | Path) -> list[str]:
    return parse(Path(path).read_text(encoding="utf-8"))


def sweep(
    channels: list[str],
    per_channel: int = 5,
    *,
    min_duration_sec: float | None = DEFAULT_MIN_SEC,
    max_duration_sec: float | None = DEFAULT_MAX_SEC,
    progress=None,
    workers: int = LIST_WORKERS,
) -> RosterResult:
    """Every channel's recent clips, merged and ranked by view count.

    A channel that cannot be listed — renamed, offline, never streamed — is
    recorded and skipped. Fifty channels means fifty chances to fail, and
    one of them taking down the batch would make the roster useless exactly
    when it is longest.
    """
    emit = progress or (lambda fraction, message: None)
    result = RosterResult()
    if not channels:
        return result

    done = 0

    def one(name: str):
        return name, channel_clips(
            name,
            limit=per_channel,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, name): name for name in channels}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            done += 1
            try:
                _, items = future.result()
            except Exception as err:  # noqa: BLE001 — one channel is not the batch
                result.failed[name] = str(err)[:160]
                emit(done / len(channels), f"{name}: {str(err)[:60]}")
                continue
            result.reached.append(name)
            result.items.extend(items)
            emit(done / len(channels), f"{name}: {len(items)} clip(s)")

    # One pool, one ranking. Ranking per channel and then interleaving would
    # give a quiet channel's best clip the same standing as the event's.
    result.items.sort(key=lambda i: i.view_count or 0, reverse=True)
    return result
