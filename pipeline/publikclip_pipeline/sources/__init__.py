"""Discovery: deciding what to clip, before anything is downloaded.

Both backends run on the checksum-verified yt-dlp binary the ingest stage
already manages, so finding material needs no API key, no OAuth and no
quota. The one exception is Twitch's browse-by-category, which has no
anonymous route at all — see twitch.category_clips.
"""

from . import opportunity, twitch, youtube
from .common import SourceItem, already_processed, unseen, within_duration

__all__ = [
    "SourceItem",
    "already_processed",
    "opportunity",
    "twitch",
    "unseen",
    "within_duration",
    "youtube",
]
