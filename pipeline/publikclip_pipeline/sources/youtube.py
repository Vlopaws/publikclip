"""Discover a YouTube channel's recent uploads.

Deliberately built on the managed yt-dlp binary rather than the YouTube Data
API: no Google Cloud project, no OAuth, no API key, and no 10 000-unit daily
quota to budget against. The same binary the ingest stage already downloads
and checksum-verifies does the listing.

`--flat-playlist` is what keeps this cheap — one request describes the whole
listing instead of one extraction per video. The trade is that flat entries
carry no upload date; the /videos tab returns newest-first, so "the latest N"
is positional rather than date-filtered. Ask for dates only when you need
them (`with_dates=True`), which costs one extraction per video.

The /videos tab excludes Shorts by design, which is what we want — a 30-second
vertical clip is not a clipping source.
"""

from __future__ import annotations

import json
import re

from ..ingest import ytdlp
from .common import SourceItem, within_duration

# A source shorter than this has nothing to cut from; longer than this and a
# single job spends hours of transcription plus tens of LLM calls.
DEFAULT_MIN_SEC = 120.0
DEFAULT_MAX_SEC = 4 * 60 * 60.0

_CHANNEL_ID = re.compile(r"^UC[\w-]{22}$")
_HANDLE = re.compile(r"^@[\w.-]+$")
_TAB = re.compile(r"/(videos|streams|shorts|playlists|featured)/?$")


def channel_url(channel: str, tab: str = "videos") -> str:
    """Normalise the many ways a channel gets referred to into one URL.

    Accepts `@handle`, a bare handle, a `UC…` channel id, or any youtube.com
    channel URL with or without a tab already on it.
    """
    ref = channel.strip().rstrip("/")
    if not ref:
        raise ValueError("empty channel reference")

    if ref.startswith("http://") or ref.startswith("https://"):
        base = _TAB.sub("", ref)
        return f"{base}/{tab}"
    if _CHANNEL_ID.match(ref):
        return f"https://www.youtube.com/channel/{ref}/{tab}"
    if _HANDLE.match(ref):
        return f"https://www.youtube.com/{ref}/{tab}"
    # A bare name: treat it as a handle, which is how YouTube resolves it
    # today. Handles cannot contain whitespace, so a display name like
    # "Sans Permission" has to lose its spaces to stand a chance of
    # resolving — a guess, but a better one than a URL that cannot exist.
    handle = "".join(ref.lstrip("@").split())
    return f"https://www.youtube.com/@{handle}/{tab}"


def _entry_to_item(entry: dict, channel_name: str | None) -> SourceItem | None:
    video_id = entry.get("id")
    if not video_id:
        return None
    url = entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
    duration = entry.get("duration")
    return SourceItem(
        id=video_id,
        url=url,
        title=entry.get("title") or video_id,
        source="youtube",
        duration_sec=float(duration) if duration else None,
        view_count=entry.get("view_count"),
        channel=entry.get("channel") or channel_name,
        raw=entry,
    )


def recent_uploads(
    channel: str,
    limit: int = 10,
    *,
    min_duration_sec: float | None = DEFAULT_MIN_SEC,
    max_duration_sec: float | None = DEFAULT_MAX_SEC,
    tab: str = "videos",
    progress: ytdlp.ProgressFn | None = None,
) -> list[SourceItem]:
    """The channel's most recent uploads, newest first.

    `limit` bounds what yt-dlp is asked for, so duration filtering can leave
    fewer than `limit` items — raise it if a channel posts a lot of material
    outside the duration window.
    """
    emit = progress or (lambda fraction, message: None)
    binary = ytdlp.ensure_ytdlp(emit)
    url = channel_url(channel, tab=tab)
    emit(-1, f"Listing {url}…")

    raw = ytdlp._run(
        binary,
        [
            "--flat-playlist",
            "-J",
            "--playlist-end",
            str(max(1, limit)),
            "--no-warnings",
            url,
        ],
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        raise ytdlp.YtDlpError(f"Could not read the channel listing for {url}: {err}") from err

    channel_name = data.get("channel") or data.get("uploader") or data.get("title")
    items = [
        item
        for item in (_entry_to_item(e, channel_name) for e in data.get("entries") or [])
        if item is not None
    ]
    return within_duration(items, min_duration_sec, max_duration_sec)
