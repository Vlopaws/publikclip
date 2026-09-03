"""Publishing through Zernio: one API, one key, many networks.

Chosen over the alternatives for what it removes rather than what it adds.
Composio brokers OAuth but still wants a connection per platform managed
through its own model; Postiz is nine containers to run and feed. Zernio is
a bearer token and three endpoints, and the accounts are connected once in a
browser by a person, which is exactly where that step belongs.

The media flow is a presign, not a multipart upload, and that detail decides
the design: a rendered clip lives on a private VM with no public address, so
it cannot simply be linked. Zernio hands out a short-lived upload URL and
the public URL the post will reference, the bytes go straight to storage,
and the post carries the second URL. Uploads sit in temporary storage for
seven days; publishing copies them to permanent storage.

Nothing here invents a caption. The clip already carries a title, a
description and hashtags written from its own transcript at render time, and
this passes those through. What a post says should be traceable to what was
said in it.
"""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path

import httpx

from .. import config
from .publish import (
    PLATFORMS,
    VISIBILITIES,
    PublishError,
    PublishResult,
    _reject_unknown,
)
from .select import SelectedClip

API_BASE = "https://zernio.com/api/v1"
API_KEY_SECRET = "zernio_api_key"
API_KEY_ENV = "PUBLIKCLIP_ZERNIO_API_KEY"

# A clip is tens of megabytes and the upload is one PUT, so the timeout has
# to cover the whole transfer rather than a request round trip.
UPLOAD_TIMEOUT = 900.0
API_TIMEOUT = 60.0

# The presign response names the two URLs, and the docs quote only one of
# them ("publicUrl"). Rather than guess the other and fail with a KeyError
# months from now, accept what the field is plausibly called and say exactly
# what came back when none of them is there.
_UPLOAD_URL_KEYS = ("uploadUrl", "upload_url", "url", "signedUrl", "presignedUrl")
_PUBLIC_URL_KEYS = ("publicUrl", "public_url", "mediaUrl", "fileUrl")


def api_key() -> str | None:
    return config.secret(API_KEY_SECRET, API_KEY_ENV)


def _first(payload: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    # Some APIs nest the interesting part one level down.
    for container in payload.values():
        if isinstance(container, dict):
            found = _first(container, keys)
            if found:
                return found
    return None


class ZernioPublisher:
    """Posts a rendered clip to the accounts connected in Zernio."""

    name = "zernio"

    def __init__(
        self,
        visibility: str = "private",
        schedule_in_minutes: int | None = None,
    ):
        if visibility not in VISIBILITIES:
            raise PublishError(
                f"Unknown visibility {visibility!r} ({', '.join(VISIBILITIES)})"
            )
        key = api_key()
        if not key:
            raise PublishError(
                "No Zernio API key found. Put it in ~/.publikclip/secrets.json as "
                f'"{API_KEY_SECRET}", or set {API_KEY_ENV}. '
                "Get one at zernio.com — the first two accounts are free."
            )
        self._key = key
        self.visibility = visibility
        self.schedule_in_minutes = schedule_in_minutes
        self._accounts: dict[str, str] | None = None

    # --- plumbing --------------------------------------------------------

    def _headers(self, json_body: bool = True) -> dict:
        headers = {"Authorization": f"Bearer {self._key}"}
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _get(self, path: str) -> dict:
        try:
            res = httpx.get(
                f"{API_BASE}{path}", headers=self._headers(False), timeout=API_TIMEOUT
            )
        except httpx.HTTPError as err:
            raise PublishError(f"Zernio unreachable: {err}") from err
        return self._read(res, path)

    def _post(self, path: str, body: dict) -> dict:
        try:
            res = httpx.post(
                f"{API_BASE}{path}",
                headers=self._headers(),
                json=body,
                timeout=API_TIMEOUT,
            )
        except httpx.HTTPError as err:
            raise PublishError(f"Zernio unreachable: {err}") from err
        return self._read(res, path)

    @staticmethod
    def _read(res, path: str) -> dict:
        if res.status_code in (401, 403):
            raise PublishError(
                "Zernio rejected the API key. Check it in "
                "~/.publikclip/secrets.json."
            )
        if res.status_code >= 400:
            raise PublishError(
                f"Zernio {path} failed (HTTP {res.status_code}): {res.text[:300]}"
            )
        try:
            return res.json()
        except json.JSONDecodeError as err:
            raise PublishError(
                f"Zernio {path} returned something that is not JSON: {res.text[:200]}"
            ) from err

    # --- accounts --------------------------------------------------------

    def accounts(self, refresh: bool = False) -> dict[str, str]:
        """platform -> accountId, for the accounts connected in Zernio.

        Fetched once. A connection is made in a browser by a person and does
        not change during a run.
        """
        if self._accounts is not None and not refresh:
            return self._accounts

        payload = self._get("/accounts")
        rows = payload if isinstance(payload, list) else (
            payload.get("accounts") or payload.get("data") or []
        )
        found: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            platform = (row.get("platform") or row.get("provider") or "").lower()
            account = row.get("accountId") or row.get("id") or row.get("account_id")
            if platform and account:
                found.setdefault(platform, str(account))
        self._accounts = found
        return found

    def check_ready(self, platforms: list[str]) -> None:
        """Fail before the batch, not during it.

        A run that transcribes an hour of video and then discovers that
        TikTok was never connected has already paid for the expensive part.
        """
        _reject_unknown(platforms)
        connected = self.accounts()
        if not connected:
            raise PublishError(
                "Zernio has no connected accounts. Connect them at zernio.com, "
                "then re-run."
            )
        missing = [p for p in platforms if p not in connected]
        if missing:
            raise PublishError(
                f"Not connected in Zernio: {', '.join(missing)}. "
                f"Connected: {', '.join(sorted(connected)) or 'none'}."
            )

    # --- media -----------------------------------------------------------

    def upload(self, path: Path) -> str:
        """Put the clip in Zernio's storage; return the URL a post can use."""
        if not path.exists():
            raise PublishError(f"Clip file is missing: {path}")
        content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"

        presigned = self._post(
            "/media/presign",
            {"filename": path.name, "contentType": content_type},
        )
        upload_url = _first(presigned, _UPLOAD_URL_KEYS)
        public_url = _first(presigned, _PUBLIC_URL_KEYS)
        if not upload_url or not public_url:
            raise PublishError(
                "Zernio's presign response did not carry the URLs this expects. "
                f"It returned: {json.dumps(presigned)[:300]}"
            )

        try:
            with open(path, "rb") as fh:
                res = httpx.put(
                    upload_url,
                    content=fh,
                    headers={"Content-Type": content_type},
                    timeout=UPLOAD_TIMEOUT,
                )
        except httpx.HTTPError as err:
            raise PublishError(f"Uploading {path.name} to Zernio failed: {err}") from err
        if res.status_code >= 400:
            raise PublishError(
                f"Zernio storage refused {path.name} "
                f"(HTTP {res.status_code}): {res.text[:200]}"
            )
        return public_url

    # --- posting ---------------------------------------------------------

    def _caption(self, clip: SelectedClip) -> str:
        """Whatever the render stage wrote, unchanged.

        The description and hashtags were generated from this clip's own
        transcript. Rewriting them here would put words in a post that
        nothing in the video accounts for.
        """
        parts = [clip.caption()]
        tags = getattr(clip, "hashtags", None) or []
        if tags:
            parts.append(" ".join(f"#{t.lstrip('#')}" for t in tags))
        return "\n\n".join(p for p in parts if p)

    def publish(self, clip: SelectedClip, platform: str) -> PublishResult:
        try:
            account = self.accounts().get(platform)
            if not account:
                raise PublishError(f"{platform} is not connected in Zernio")

            media_url = self.upload(Path(clip.path))
            body: dict = {
                "content": self._caption(clip),
                "mediaItems": [{"url": media_url, "type": "video"}],
                "platforms": [{"platform": platform, "accountId": account}],
            }
            if self.schedule_in_minutes:
                import datetime as _dt

                when = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(
                    minutes=self.schedule_in_minutes
                )
                body["scheduledFor"] = when.strftime("%Y-%m-%dT%H:%M:%SZ")
                body["timezone"] = "UTC"
            else:
                body["publishNow"] = True

            created = self._post("/posts", body)
            post_id = _first(created, ("id", "postId", "post_id"))
            return PublishResult(
                clip=clip,
                platform=platform,
                ok=True,
                post_id=post_id,
                url=_first(created, ("url", "permalink", "postUrl")),
                visibility=self.visibility,
            )
        except PublishError as err:
            return PublishResult(
                clip=clip, platform=platform, ok=False, error=str(err)
            )
