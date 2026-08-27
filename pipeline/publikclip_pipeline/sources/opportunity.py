"""Is a creator already served by clippers, or is there room?

The question "who is worth clipping" ages badly as a list and not at all as
a measurement, so this measures. YouTube search, read through the same
anonymous yt-dlp path as the rest of discovery, is a decent proxy for the
clip scene around a creator: if three dedicated channels are posting their
moments to six figures of views, the niche is taken.

What this is NOT: ground truth. It is a heuristic over search results, and
search ranking is not a census. Treat a low score as "worth a look", never
as "nobody is clipping them". The evidence is returned alongside the number
so the judgement stays yours.

Nothing here downloads anything or needs an API key.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..ingest import ytdlp
from . import youtube as youtube_source

# Channel names that announce themselves as clip channels, across the
# languages this is most likely pointed at.
_CLIP_CHANNEL = re.compile(
    r"clip|best[\s-]?of|bestof|moments|highlight|extrait|zapping|momentos|shorts?$",
    re.IGNORECASE,
)

# Queries kept generic on purpose: adding the creator's language would bias
# the result toward whichever language the caller happened to guess.
_QUERIES = ("{name} clips", "{name} best moments", "{name} best of")

# Mirrors youtube._CHANNEL_ID; kept local so this module reads standalone.
_CHANNEL_ID_RE = re.compile(r"^UC[\w-]{22}$")


@dataclass
class ClipChannel:
    name: str
    videos_found: int = 0
    total_views: int = 0
    top_views: int = 0
    top_title: str = ""

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "videos_found": self.videos_found,
            "total_views": self.total_views,
            "top_views": self.top_views,
            "top_title": self.top_title,
        }


@dataclass
class Saturation:
    creator: str
    clip_channels: list[ClipChannel] = field(default_factory=list)
    results_scanned: int = 0

    @property
    def dedicated_channels(self) -> int:
        return len(self.clip_channels)

    @property
    def top_clip_views(self) -> int:
        """Best single clip found. A hit proves the audience wants clips of
        this creator, which is the opposite of a reason to skip them — but
        combined with several dedicated channels it means the lane is taken."""
        return max((c.top_views for c in self.clip_channels), default=0)

    @property
    def verdict(self) -> str:
        """A blunt three-way read, deliberately coarse — the underlying
        signal is not precise enough to deserve a percentage."""
        if self.dedicated_channels == 0:
            return "open"
        if self.dedicated_channels <= 2 and self.top_clip_views < 100_000:
            return "thin"
        return "crowded"

    def to_json(self) -> dict:
        return {
            "creator": self.creator,
            "verdict": self.verdict,
            "dedicated_channels": self.dedicated_channels,
            "top_clip_views": self.top_clip_views,
            "results_scanned": self.results_scanned,
            "channels": [c.to_json() for c in self.clip_channels],
        }


def _search(binary, query: str, limit: int) -> list[dict]:
    raw = ytdlp._run(
        binary,
        ["--flat-playlist", "-J", "--no-warnings", f"ytsearch{limit}:{query}"],
    )
    try:
        return json.loads(raw).get("entries") or []
    except json.JSONDecodeError:
        return []


def _looks_like_the_creator(channel: str, creator: str) -> bool:
    """A creator's own 'best of' uploads are not competition."""
    return creator.lower().strip("@").replace(" ", "") == channel.lower().replace(" ", "")


def clip_saturation(
    creator: str,
    per_query: int = 12,
    progress: ytdlp.ProgressFn | None = None,
) -> Saturation:
    """How crowded the clip scene around `creator` looks."""
    emit = progress or (lambda fraction, message: None)
    binary = ytdlp.ensure_ytdlp(emit)
    name = creator.strip().lstrip("@")

    channels: dict[str, ClipChannel] = {}
    scanned = 0
    for template in _QUERIES:
        query = template.format(name=name)
        emit(-1, f"Searching “{query}”…")
        for entry in _search(binary, query, per_query):
            scanned += 1
            channel_name = entry.get("channel") or entry.get("uploader")
            if not channel_name or not _CLIP_CHANNEL.search(channel_name):
                continue
            if _looks_like_the_creator(channel_name, name):
                continue
            record = channels.setdefault(channel_name, ClipChannel(name=channel_name))
            record.videos_found += 1
            views = entry.get("view_count") or 0
            record.total_views += views
            if views >= record.top_views:
                record.top_views = views
                record.top_title = (entry.get("title") or "")[:80]

    ranked = sorted(channels.values(), key=lambda c: c.total_views, reverse=True)
    return Saturation(creator=name, clip_channels=ranked, results_scanned=scanned)


# --- demand, not just supply ---------------------------------------------
#
# Saturation alone is ambiguous: "nobody clips them" and "nobody wants clips
# of them" produce the same zero. Pairing it with the creator's own reach
# separates the two, and the pair is what actually says whether to bother.
#
# The thresholds below are design values, not measurements — they say what
# "enough audience to be worth an hour of compute" means, and they are the
# first thing to tune once real outcomes exist (the same way the scoring
# constants get fitted from Instagram results).
MIN_VIABLE_MEDIAN_VIEWS = 20_000
STRONG_MEDIAN_VIEWS = 100_000


@dataclass
class Opportunity:
    creator: str
    saturation: "Saturation"
    median_views: int | None
    uploads_seen: int
    # Which channel the demand figure actually came from. Resolution is a
    # majority vote over search results and can land on a same-named channel
    # or a fan re-upload, so the number is only as good as this name — it is
    # reported rather than assumed correct.
    measured_channel: str | None = None

    @property
    def verdict(self) -> str:
        if self.median_views is None:
            return "unknown"
        if self.median_views < MIN_VIABLE_MEDIAN_VIEWS:
            return "too small"
        if self.saturation.verdict == "crowded":
            return "taken"
        if self.median_views >= STRONG_MEDIAN_VIEWS:
            return "sweet spot"
        return "worth a look"

    def to_json(self) -> dict:
        return {
            "creator": self.creator,
            "verdict": self.verdict,
            "median_views": self.median_views,
            "uploads_seen": self.uploads_seen,
            "measured_channel": self.measured_channel,
            "saturation": self.saturation.to_json(),
        }


def _median(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) // 2


def resolve_channel(
    name: str, per_query: int = 8, progress: ytdlp.ProgressFn | None = None
) -> str | None:
    """The channel id a display name most likely refers to.

    Turning "Sans Permission" into a handle by deleting its spaces is a
    guess, and a bad one — it found a different, tiny channel of the same
    name. Search knows the answer, so ask it and take the majority channel
    across the top results rather than inventing a URL.
    """
    emit = progress or (lambda fraction, message: None)
    binary = ytdlp.ensure_ytdlp(emit)
    counts: dict[str, int] = {}
    for entry in _search(binary, name, per_query):
        channel_id = entry.get("channel_id")
        if channel_id:
            counts[channel_id] = counts.get(channel_id, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def audience(
    channel: str, sample: int = 10, progress: ytdlp.ProgressFn | None = None
) -> tuple[int | None, int, str | None]:
    """Median views across recent uploads, as a proxy for reach.

    Median rather than mean on purpose: one viral video should not make a
    quiet channel look like a big one. Duration filtering is off here — we
    are measuring the audience, not picking material.

    A `UC…` id or an `@handle` is used as given; anything else is resolved
    through search first, because a display name is not a handle.
    """
    ref = channel.strip()
    if not (ref.startswith("@") or ref.startswith("http") or _CHANNEL_ID_RE.match(ref)):
        resolved = resolve_channel(ref, progress=progress)
        if not resolved:
            return None, 0, None
        ref = resolved
    try:
        items = youtube_source.recent_uploads(
            ref,
            limit=sample,
            min_duration_sec=None,
            max_duration_sec=None,
            progress=progress,
        )
    except Exception:  # noqa: BLE001 - an unresolvable channel is a real answer
        return None, 0, None
    views = [i.view_count for i in items if i.view_count]
    measured = items[0].channel if items else None
    return _median(views), len(items), measured


def assess(
    creator: str,
    channel: str | None = None,
    per_query: int = 12,
    progress: ytdlp.ProgressFn | None = None,
) -> Opportunity:
    """Supply and demand together: is this creator worth clipping?

    `channel` defaults to the creator name, which resolves as a handle for
    most channels; pass it explicitly when the searchable name and the
    YouTube handle differ.
    """
    saturation = clip_saturation(creator, per_query=per_query, progress=progress)
    median_views, seen, measured = audience(channel or creator, progress=progress)
    return Opportunity(
        creator=creator,
        saturation=saturation,
        median_views=median_views,
        uploads_seen=seen,
        measured_channel=measured,
    )
