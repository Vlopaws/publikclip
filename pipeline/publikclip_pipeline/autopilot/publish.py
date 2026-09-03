"""Where a selected clip goes once it exists.

Publishing is the one irreversible, public step in this pipeline, so the
shape here is deliberately defensive:

- The default backend posts nothing. `DryRunPublisher` reports exactly what
  would be sent, which is what makes an automated run reviewable before it
  is trusted.
- The default *visibility* is private. An unattended poster that defaults to
  public is one bad score away from an audience, and "private then promote"
  is recoverable in a way the reverse is not.
- Every backend records what it did to a ledger, so a re-run cannot post the
  same clip twice, and so the Instagram feedback loop has a clip↔post link
  to calibrate against later.

Composio matters here for a reason that is easy to miss: Meta ingests Reels
by *public URL*, not by upload, which is why the upstream project deferred
auto-publishing entirely. Composio's file parameters do the temporary
hosting hop themselves, which is what makes this possible without running a
bucket.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from .select import SelectedClip

PLATFORMS = ("instagram", "tiktok", "youtube")
VISIBILITIES = ("private", "unlisted", "public")

API_KEY_ENV = "PUBLIKCLIP_COMPOSIO_API_KEY"

# People & Blogs. YouTube requires a category and this is the least wrong
# default for talk-show and podcast clips.
DEFAULT_YOUTUBE_CATEGORY = "22"

# TikTok has no "unlisted"; its narrowest setting is self-only, which is also
# the only one an unaudited app may use.
_TIKTOK_PRIVACY = {
    "private": "SELF_ONLY",
    "unlisted": "SELF_ONLY",
    "public": "PUBLIC_TO_EVERYONE",
}


class PublishError(Exception):
    """User-actionable publishing failure (no key, no connection, refused)."""


@dataclass
class PublishResult:
    clip: SelectedClip
    platform: str
    ok: bool
    dry_run: bool = False
    post_id: str | None = None
    url: str | None = None
    visibility: str | None = None
    error: str | None = None

    def to_json(self) -> dict:
        return {
            "job_id": self.clip.job_id,
            "clip": self.clip.clip,
            "platform": self.platform,
            "ok": self.ok,
            "dry_run": self.dry_run,
            "post_id": self.post_id,
            "url": self.url,
            "visibility": self.visibility,
            "error": self.error,
        }


# --- ledger ---------------------------------------------------------------


def ledger_path() -> Path:
    return config.home_dir() / "autopilot-posts.json"


def _load_ledger() -> list[dict]:
    path = ledger_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def record(result: PublishResult) -> None:
    """Append to the ledger. Dry runs are recorded too, marked as such, so a
    rehearsal is auditable without being mistaken for a real post."""
    entries = _load_ledger()
    entries.append({**result.to_json(), "at": time.time()})
    config.ensure_home()
    ledger_path().write_text(json.dumps(entries, indent=1), encoding="utf-8")


def already_posted(clip: SelectedClip, platform: str) -> bool:
    """A real post of this exact clip to this platform. Dry runs do not count
    — rehearsing must never block the real thing."""
    for entry in _load_ledger():
        if (
            entry.get("job_id") == clip.job_id
            and entry.get("clip") == clip.clip
            and entry.get("platform") == platform
            and entry.get("ok")
            and not entry.get("dry_run")
        ):
            return True
    return False


# --- backends -------------------------------------------------------------


# Scoring and publishing name things differently and always will: a clip is
# scored for "reels" and "shorts" — the formats — but posted to "instagram"
# and "youtube" — the accounts. Someone reading a score report and then
# writing --platforms is going to type the format, so say what to type
# instead of only what was wrong.
_FORMAT_TO_PLATFORM = {"reels": "instagram", "shorts": "youtube", "tiktoks": "tiktok"}


def _reject_unknown(platforms: list[str]) -> None:
    unknown = [p for p in platforms if p not in PLATFORMS]
    if not unknown:
        return
    supported = f"Supported: {', '.join(PLATFORMS)}."
    if all(p in _FORMAT_TO_PLATFORM for p in unknown):
        swaps = ", ".join(f"{p} -> {_FORMAT_TO_PLATFORM[p]}" for p in unknown)
        raise PublishError(
            f"{', '.join(unknown)}: that is a format name, not a platform. "
            f"Use {swaps}. {supported}"
        )
    raise PublishError(
        f"Unsupported platform(s): {', '.join(unknown)}. {supported}"
    )


@dataclass
class DryRunPublisher:
    """Posts nothing. The default, and the thing to run first."""

    name: str = "dry-run"
    visibility: str = "private"
    calls: list[tuple[SelectedClip, str]] = field(default_factory=list)

    def check_ready(self, platforms: list[str]) -> None:
        _reject_unknown(platforms)

    def publish(self, clip: SelectedClip, platform: str) -> PublishResult:
        self.calls.append((clip, platform))
        return PublishResult(
            clip=clip, platform=platform, ok=True, dry_run=True, visibility=self.visibility
        )


class ComposioPublisher:
    """The real one: Composio-managed OAuth, one connected account per
    platform, files uploaded through Composio's temporary hosting."""

    name = "composio"

    def __init__(
        self,
        user_id: str = "publikclip",
        visibility: str = "private",
        youtube_category: str = DEFAULT_YOUTUBE_CATEGORY,
    ):
        if visibility not in VISIBILITIES:
            raise PublishError(f"visibility must be one of {', '.join(VISIBILITIES)}")
        self.user_id = user_id
        self.visibility = visibility
        self.youtube_category = youtube_category
        self._session = None

    # -- plumbing ----------------------------------------------------------

    def _connect(self):
        if self._session is not None:
            return self._session
        api_key = config.secret("composio_api_key", API_KEY_ENV)
        if not api_key:
            raise PublishError(
                "Composio needs an API key to publish unattended. Get one at "
                "dashboard.composio.dev, then set "
                f"{API_KEY_ENV} (or composio_api_key in ~/.publikclip/secrets.json)."
            )
        try:
            from composio import Composio
        except ImportError as err:
            raise PublishError(
                "The composio package is not installed. Add it to the pipeline "
                "environment (uv add composio) to publish."
            ) from err
        client = Composio(api_key=api_key)
        self._session = client.sessions.create(user_id=self.user_id)
        return self._session

    def check_ready(self, platforms: list[str]) -> None:
        """Fail before the batch, not during it.

        A run that transcribes an hour of video and then discovers there is
        no connection has already paid for the expensive part.
        """
        _reject_unknown(platforms)
        if "instagram" in platforms and self.visibility != "public":
            # Instagram has no private publish: a Reel is public the moment
            # it exists. Posting it anyway under a private request would be
            # exactly the surprise this default exists to prevent.
            raise PublishError(
                "Instagram has no private or unlisted publish — a Reel is public "
                f"immediately. Refusing to post it under visibility={self.visibility!r}. "
                "Use --visibility public to accept that, or drop instagram from "
                "--platforms."
            )
        self._connect()

    @staticmethod
    def _file_arg(path: Path) -> str:
        """The value for Composio's `file_uploadable` parameters.

        The SDK resolves a local path into the {name, mimetype, s3key} shape
        the tool schema declares, uploading to its own temporary storage on
        the way. If a deployment ever wants that dict built by hand, this is
        the one place to change.
        """
        return str(path)

    def _execute(self, tool_slug: str, arguments: dict) -> dict:
        session = self._connect()
        try:
            response = session.execute(tool_slug=tool_slug, arguments=arguments)
        except Exception as err:  # noqa: BLE001 - SDK raises provider-specific types
            raise PublishError(f"{tool_slug} failed: {err}") from err
        self._raise_for_nested_error(tool_slug, response)
        return response

    @staticmethod
    def _raise_for_nested_error(tool_slug: str, response: dict) -> None:
        """Some tools report success at the envelope and failure inside it.

        TikTok is the documented case: publish can return successful=true
        with data.error set (url_ownership_unverified, and the 403 an
        unaudited app gets). Trusting the envelope would record a post that
        never happened.
        """
        data = response.get("data") if isinstance(response, dict) else None
        if not isinstance(data, dict):
            return
        nested = data.get("error")
        if not nested:
            return
        code = nested.get("code") if isinstance(nested, dict) else str(nested)
        message = nested.get("message", "") if isinstance(nested, dict) else ""
        if code and "unaudited_client" in str(code):
            raise PublishError(
                "TikTok refused: an unaudited app may only post to private "
                "accounts. Either keep --visibility private, or get the app "
                "audited at developers.tiktok.com."
            )
        raise PublishError(f"{tool_slug} reported success but returned error {code}: {message}")

    # -- platforms ---------------------------------------------------------

    def publish(self, clip: SelectedClip, platform: str) -> PublishResult:
        handlers = {
            "instagram": self._instagram,
            "tiktok": self._tiktok,
            "youtube": self._youtube,
        }
        handler = handlers.get(platform)
        if handler is None:
            return PublishResult(
                clip=clip, platform=platform, ok=False,
                error=f"Unsupported platform {platform!r}",
            )
        try:
            return handler(clip)
        except PublishError as err:
            return PublishResult(
                clip=clip, platform=platform, ok=False,
                visibility=self.visibility, error=str(err),
            )

    def _instagram(self, clip: SelectedClip) -> PublishResult:
        me = self._execute("INSTAGRAM_GET_USER_INFO", {})
        data = me.get("data") or {}
        ig_user_id = data.get("id") or data.get("user_id")
        if not ig_user_id:
            raise PublishError(
                "Could not read the Instagram user id. Publishing requires a "
                "Business or Creator account, not a personal one."
            )
        container = self._execute(
            "INSTAGRAM_POST_IG_USER_MEDIA",
            {
                "ig_user_id": ig_user_id,
                "media_type": "REELS",
                "caption": clip.caption(),
                "video_file": self._file_arg(clip.path),
            },
        )
        creation_id = (container.get("data") or {}).get("id")
        if not creation_id:
            raise PublishError(f"Instagram did not return a container id: {container}")
        published = self._execute(
            "INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH",
            {"ig_user_id": ig_user_id, "creation_id": creation_id},
        )
        media_id = (published.get("data") or {}).get("id")
        if not media_id:
            raise PublishError("Instagram accepted the container but returned no media id.")
        return PublishResult(
            clip=clip, platform="instagram", ok=True, post_id=media_id, visibility="public"
        )

    def _tiktok(self, clip: SelectedClip) -> PublishResult:
        """Upload and publish in one call.

        TIKTOK_UPLOAD_VIDEO takes `publish=True`, which saves handing a
        publish_id to a second tool — one fewer place for an upload to
        succeed and a publish to silently not.
        """
        response = self._execute(
            "TIKTOK_UPLOAD_VIDEO",
            {
                "file_to_upload": self._file_arg(clip.path),
                "caption": clip.caption(2200),
                "privacy_level": _TIKTOK_PRIVACY[self.visibility],
                "publish": True,
            },
        )
        data = response.get("data") or {}
        publish_id = data.get("publish_id")
        if not publish_id:
            raise PublishError(f"TikTok returned no publish_id: {response}")
        return PublishResult(
            clip=clip, platform="tiktok", ok=True, post_id=publish_id,
            visibility=self.visibility,
        )

    def _youtube(self, clip: SelectedClip) -> PublishResult:
        """Metadata and file in one multipart request.

        The title is the clip's own summary, trimmed to something a Shorts
        title can carry; the full summary becomes the description.
        """
        response = self._execute(
            "YOUTUBE_MULTIPART_UPLOAD_VIDEO",
            {
                "title": clip.caption(90),
                "description": clip.summary or clip.caption(),
                "categoryId": self.youtube_category,
                "privacyStatus": self.visibility,
                "videoFile": self._file_arg(clip.path),
            },
        )
        data = response.get("data") or {}
        video_id = data.get("id") or (data.get("video") or {}).get("id")
        if not video_id:
            raise PublishError(f"YouTube returned no video id: {response}")
        return PublishResult(
            clip=clip, platform="youtube", ok=True, post_id=video_id,
            url=f"https://www.youtube.com/watch?v={video_id}",
            visibility=self.visibility,
        )


def make_publisher(mode: str, visibility: str = "private", **kwargs):
    if mode in ("dry-run", "dry_run", "none"):
        return DryRunPublisher(visibility=visibility)
    if mode == "composio":
        return ComposioPublisher(visibility=visibility, **kwargs)
    if mode == "zernio":
        # Imported here for the same reason as postiz below: zernio imports
        # from this module, and a top-level import either way is circular.
        from .zernio import ZernioPublisher

        return ZernioPublisher(visibility=visibility, **kwargs)
    if mode == "postiz":
        # Imported here: postiz imports from this module, and a top-level
        # import either way would be circular.
        from .postiz import PostizPublisher

        return PostizPublisher(visibility=visibility, **kwargs)
    raise PublishError(
        f"Unknown publish mode {mode!r} (dry-run | zernio | composio | postiz)"
    )
