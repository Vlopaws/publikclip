"""Publishing through a self-hosted Postiz instance.

The alternative to Composio, and a different trade. Composio is a managed
service: one call, no infrastructure, someone else's uptime and someone
else's copy of your clips. Postiz is nine containers you run yourself, and
in exchange the credentials, the media and the schedule stay on your
machine — and it is AGPL, like this project.

The reason it earns its footprint is `type: "schedule"`. An automated
poster that fires immediately is trusted or it is not; one that queues a
scheduled post gives a window to look at what the pipeline chose and cancel
it. That is the dry run made permanent, so scheduling is the default here
and posting now has to be asked for.

Not tested against a live instance — no connected channel existed when this
was written. The request shapes follow Postiz's documented public API; the
first real call is the proof.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from .. import config
from .publish import PLATFORMS, PublishError, PublishResult
from .select import SelectedClip

API_KEY_ENV = "PUBLIKCLIP_POSTIZ_API_KEY"
URL_ENV = "PUBLIKCLIP_POSTIZ_URL"
DEFAULT_URL = "http://localhost:4007"

# How far ahead a scheduled post lands. Long enough to actually look at it
# before it goes out, short enough that a daily run still posts that day.
DEFAULT_SCHEDULE_DELAY_MIN = 60

# Postiz names channels by provider; these are the identifiers it uses for
# the three platforms this pipeline renders for.
_PROVIDER_ALIASES = {
    "instagram": {"instagram", "instagram-standalone"},
    "tiktok": {"tiktok"},
    "youtube": {"youtube"},
}

TIMEOUT = 120.0


def api_key() -> str | None:
    return config.secret("postiz_api_key", API_KEY_ENV)


def base_url() -> str:
    return (config.secret("postiz_url", URL_ENV) or DEFAULT_URL).rstrip("/")


def available() -> bool:
    return bool(api_key())


class PostizPublisher:
    """Queues clips into a self-hosted Postiz instance."""

    name = "postiz"

    def __init__(
        self,
        visibility: str = "private",
        post_now: bool = False,
        schedule_delay_min: int = DEFAULT_SCHEDULE_DELAY_MIN,
    ):
        self.visibility = visibility
        self.post_now = post_now
        self.schedule_delay_min = schedule_delay_min
        self._integrations: dict[str, str] | None = None

    # -- plumbing ----------------------------------------------------------

    def _headers(self) -> dict:
        key = api_key()
        if not key:
            raise PublishError(
                "Postiz needs an API key. Open your instance at "
                f"{base_url()}, go to Settings → Public API, then set "
                f"{API_KEY_ENV} (or postiz_api_key in ~/.publikclip/secrets.json)."
            )
        # Postiz takes the raw key, with no Bearer prefix.
        return {"Authorization": key}

    def _get(self, path: str) -> dict | list:
        try:
            res = httpx.get(
                f"{base_url()}/api/public/v1{path}", headers=self._headers(), timeout=TIMEOUT
            )
        except httpx.HTTPError as err:
            raise PublishError(
                f"Could not reach Postiz at {base_url()}: {err}. Is the stack running "
                "(docker compose ps)?"
            ) from err
        if res.status_code in (401, 403):
            raise PublishError("Postiz rejected the API key. Regenerate it in Settings.")
        res.raise_for_status()
        return res.json()

    def integrations(self) -> dict[str, str]:
        """platform name -> Postiz integration id, for what is connected."""
        if self._integrations is not None:
            return self._integrations
        payload = self._get("/integrations")
        rows = payload if isinstance(payload, list) else payload.get("integrations") or []
        found: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            provider = str(
                row.get("providerIdentifier") or row.get("provider") or row.get("identifier") or ""
            ).lower()
            ident = row.get("id")
            if not ident:
                continue
            for platform, aliases in _PROVIDER_ALIASES.items():
                if provider in aliases and platform not in found:
                    found[platform] = str(ident)
        self._integrations = found
        return found

    def check_ready(self, platforms: list[str]) -> None:
        unknown = [p for p in platforms if p not in PLATFORMS]
        if unknown:
            raise PublishError(f"Unsupported platform(s): {', '.join(unknown)}")
        if "instagram" in platforms and self.visibility != "public":
            raise PublishError(
                "Instagram has no private or unlisted publish — a Reel is public "
                f"immediately. Refusing to queue it under visibility={self.visibility!r}. "
                "Use --visibility public, or drop instagram from --platforms."
            )
        connected = self.integrations()
        missing = [p for p in platforms if p not in connected]
        if missing:
            raise PublishError(
                f"No Postiz channel connected for: {', '.join(missing)}. "
                f"Connect them at {base_url()} → Add Channel, then retry."
            )

    # -- publishing --------------------------------------------------------

    def _upload(self, path: Path) -> dict:
        """Postiz stores the file and hands back the reference a post needs."""
        try:
            with open(path, "rb") as fh:
                res = httpx.post(
                    f"{base_url()}/api/public/v1/upload",
                    headers=self._headers(),
                    files={"file": (path.name, fh, "video/mp4")},
                    timeout=TIMEOUT,
                )
        except httpx.HTTPError as err:
            raise PublishError(f"Uploading {path.name} to Postiz failed: {err}") from err
        if res.status_code != 200:
            raise PublishError(f"Postiz refused the upload (HTTP {res.status_code}): {res.text[:200]}")
        payload = res.json()
        if not payload.get("id") and not payload.get("path"):
            raise PublishError(f"Postiz upload returned no media reference: {payload}")
        return payload

    def _when(self) -> tuple[str, str]:
        """(type, ISO date). Scheduling is the default; see module docstring."""
        if self.post_now:
            return "now", datetime.now(timezone.utc).isoformat()
        at = datetime.now(timezone.utc) + timedelta(minutes=self.schedule_delay_min)
        return "schedule", at.isoformat()

    def publish(self, clip: SelectedClip, platform: str) -> PublishResult:
        try:
            integration_id = self.integrations().get(platform)
            if not integration_id:
                raise PublishError(f"No Postiz channel connected for {platform}.")
            media = self._upload(clip.path)
            post_type, when = self._when()
            body = {
                "type": post_type,
                "date": when,
                "shortLink": False,
                "tags": [],
                "posts": [
                    {
                        "integration": {"id": integration_id},
                        "value": [{"content": clip.caption(), "image": [media]}],
                        # Postiz keys its per-platform settings off __type.
                        "settings": {"__type": platform},
                    }
                ],
            }
            try:
                res = httpx.post(
                    f"{base_url()}/api/public/v1/posts",
                    headers=self._headers(),
                    json=body,
                    timeout=TIMEOUT,
                )
            except httpx.HTTPError as err:
                raise PublishError(f"Postiz post creation failed: {err}") from err
            if res.status_code == 429:
                raise PublishError(
                    "Postiz rate limit reached (90 posts/hour). Lower --clips or "
                    "spread the run out."
                )
            if res.status_code not in (200, 201):
                raise PublishError(
                    f"Postiz refused the post (HTTP {res.status_code}): {res.text[:200]}"
                )
            payload = res.json()
            post_id = payload.get("id") or (payload[0].get("id") if isinstance(payload, list) and payload else None)
            return PublishResult(
                clip=clip,
                platform=platform,
                ok=True,
                post_id=str(post_id) if post_id else None,
                url=f"{base_url()}/launches",
                visibility=self.visibility,
            )
        except PublishError as err:
            return PublishResult(
                clip=clip, platform=platform, ok=False,
                visibility=self.visibility, error=str(err),
            )
