"""Discover Twitch clips worth re-cutting.

Two paths, and they are not equivalent — which one you can use decides
whether you need Twitch credentials at all:

`channel_clips` needs nothing. yt-dlp's twitch:videos:clips extractor reads
a channel's clip page anonymously. The catch, verified rather than assumed:
yt-dlp ignores the `?filter=` query parameter and always returns the page's
default, which is **top of the last 7 days**. There is no knob here, so this
module does not pretend to offer one.

`category_clips` — "the trending clips in Just Chatting right now" — has no
anonymous route. Twitch's Helix API is the only source, and every Helix call
requires a Client-ID plus an app access token, which means registering a
(free) application at dev.twitch.tv. The function is written and ready; it
raises with exactly what is missing until those credentials exist.

Clips are short by nature. They are useful as *sources* mainly when they
still contain a cuttable beat — a 12-second clip is already the clip.
"""

from __future__ import annotations

import json
import time

import httpx

from .. import config
from ..ingest import ytdlp
from .common import SourceItem, within_duration

HELIX = "https://api.twitch.tv/helix"
OAUTH_TOKEN_URL = "https://id.twitch.tv/oauth2/token"

CLIENT_ID_ENV = "PUBLIKCLIP_TWITCH_CLIENT_ID"
CLIENT_SECRET_ENV = "PUBLIKCLIP_TWITCH_CLIENT_SECRET"

# yt-dlp's clip listing is fixed to this window; stated so callers can label
# their own output honestly.
CHANNEL_CLIP_WINDOW = "top 7 days"

# Clips below this have no internal structure left to cut.
DEFAULT_MIN_SEC = 20.0
DEFAULT_MAX_SEC = 600.0


class TwitchError(Exception):
    """User-actionable Twitch failure (missing credentials, unknown game)."""


# --- anonymous path -------------------------------------------------------


def _entry_to_item(entry: dict, channel_name: str | None) -> SourceItem | None:
    clip_id = entry.get("id")
    if not clip_id:
        return None
    url = entry.get("url") or f"https://clips.twitch.tv/{clip_id}"
    duration = entry.get("duration")
    return SourceItem(
        id=str(clip_id),
        url=url,
        title=entry.get("title") or str(clip_id),
        source="twitch",
        duration_sec=float(duration) if duration else None,
        view_count=entry.get("view_count"),
        channel=entry.get("channel") or entry.get("uploader") or channel_name,
        raw=entry,
    )


def channel_clips(
    channel: str,
    limit: int = 10,
    *,
    min_duration_sec: float | None = DEFAULT_MIN_SEC,
    max_duration_sec: float | None = DEFAULT_MAX_SEC,
    progress: ytdlp.ProgressFn | None = None,
) -> list[SourceItem]:
    """A channel's most-viewed clips of the last 7 days. No credentials."""
    emit = progress or (lambda fraction, message: None)
    binary = ytdlp.ensure_ytdlp(emit)
    name = channel.strip().rstrip("/").rsplit("/", 1)[-1].lstrip("@")
    url = f"https://www.twitch.tv/{name}/clips"
    emit(-1, f"Listing {url} ({CHANNEL_CLIP_WINDOW})…")

    raw = ytdlp._run(
        binary,
        ["--flat-playlist", "-J", "--playlist-end", str(max(1, limit)), "--no-warnings", url],
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        raise ytdlp.YtDlpError(f"Could not read the clip listing for {url}: {err}") from err

    channel_name = data.get("uploader") or data.get("channel") or name
    items = [
        item
        for item in (_entry_to_item(e, channel_name) for e in data.get("entries") or [])
        if item is not None
    ]
    return within_duration(items, min_duration_sec, max_duration_sec)


# --- Helix path (credentials required) ------------------------------------


def credentials() -> tuple[str, str] | None:
    client_id = config.secret("twitch_client_id", CLIENT_ID_ENV)
    client_secret = config.secret("twitch_client_secret", CLIENT_SECRET_ENV)
    if client_id and client_secret:
        return client_id, client_secret
    return None


def _require_credentials() -> tuple[str, str]:
    creds = credentials()
    if creds:
        return creds
    raise TwitchError(
        "Browsing clips by category needs Twitch API credentials — there is no "
        "anonymous route for it. Register a free app at dev.twitch.tv/console/apps, "
        f"then set {CLIENT_ID_ENV} and {CLIENT_SECRET_ENV} (or twitch_client_id / "
        "twitch_client_secret in ~/.publikclip/secrets.json). "
        "Clips of a specific channel need none of this — use channel_clips."
    )


_token_cache: dict[str, tuple[str, float]] = {}


def _app_token() -> str:
    """Client-credentials app token, cached until shortly before expiry."""
    client_id, client_secret = _require_credentials()
    cached = _token_cache.get(client_id)
    if cached and cached[1] > time.time():
        return cached[0]
    try:
        res = httpx.post(
            OAUTH_TOKEN_URL,
            params={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
            },
            timeout=config.HTTP_TIMEOUT,
        )
    except httpx.HTTPError as err:
        raise TwitchError(f"Could not reach Twitch to authenticate: {err}") from err
    if res.status_code in (400, 401, 403):
        raise TwitchError("Twitch rejected the client id/secret. Check them in Settings.")
    res.raise_for_status()
    payload = res.json()
    token = payload["access_token"]
    # Renew a minute early rather than discover expiry mid-listing.
    _token_cache[client_id] = (token, time.time() + max(60.0, payload.get("expires_in", 3600) - 60))
    return token


def _headers() -> dict:
    client_id, _ = _require_credentials()
    return {"Client-Id": client_id, "Authorization": f"Bearer {_app_token()}"}


def game_id(name: str) -> str:
    """Resolve a category name ("Just Chatting") to its Twitch game id."""
    res = httpx.get(
        f"{HELIX}/games", params={"name": name}, headers=_headers(), timeout=config.HTTP_TIMEOUT
    )
    res.raise_for_status()
    data = res.json().get("data") or []
    if not data:
        raise TwitchError(
            f"Twitch has no category called {name!r}. The name must match exactly, "
            "as it appears on the directory page."
        )
    return data[0]["id"]


def category_clips(
    category: str,
    limit: int = 20,
    *,
    days: int = 7,
    min_duration_sec: float | None = DEFAULT_MIN_SEC,
    max_duration_sec: float | None = DEFAULT_MAX_SEC,
) -> list[SourceItem]:
    """Most-viewed clips in a category over the last `days`.

    Helix returns clips in descending view order, so "trending" is the head
    of the list rather than a separate parameter.
    """
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - days * 86400))
    res = httpx.get(
        f"{HELIX}/clips",
        params={
            "game_id": game_id(category),
            "first": min(100, max(1, limit)),
            "started_at": started_at,
        },
        headers=_headers(),
        timeout=config.HTTP_TIMEOUT,
    )
    res.raise_for_status()
    items = []
    for clip in res.json().get("data") or []:
        duration = clip.get("duration")
        items.append(
            SourceItem(
                id=clip["id"],
                url=clip.get("url") or f"https://clips.twitch.tv/{clip['id']}",
                title=clip.get("title") or clip["id"],
                source="twitch",
                duration_sec=float(duration) if duration else None,
                view_count=clip.get("view_count"),
                channel=clip.get("broadcaster_name"),
                raw=clip,
            )
        )
    return within_duration(items, min_duration_sec, max_duration_sec)
