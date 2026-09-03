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

# Confirmed against Zernio's own OpenAPI (zernio.com/openapi.json), which
# the API's 404 responses point at: POST /v1/media/presign answers
# {uploadUrl, publicUrl, key, expiresIn}. The alternatives stay as a hedge
# against a rename, and an unrecognised response prints what it actually
# contained rather than raising a KeyError that names nothing.
_UPLOAD_URL_KEYS = ("uploadUrl", "upload_url", "url", "signedUrl", "presignedUrl")
_PUBLIC_URL_KEYS = ("publicUrl", "public_url", "mediaUrl", "fileUrl")

# Visibility is not a Zernio-wide concept. Its post schema carries
# `tiktokSettings.privacyLevel` and nothing equivalent for Instagram or
# YouTube — checked against the OpenAPI, not assumed. So "private" is
# honourable on TikTok and simply not expressible on the other two.
#
# This publisher recorded a visibility and never sent one. A post asked for
# as private went out public, and the setting looked like it worked. A
# control that silently does nothing is worse than no control, so the
# platforms that cannot honour it now refuse the post instead.
_TIKTOK_PRIVACY = {
    "private": "SELF_ONLY",
    # TikTok has no "unlisted"; self-only is its narrowest setting.
    "unlisted": "SELF_ONLY",
    "public": "PUBLIC_TO_EVERYONE",
}
# Platforms whose privacy Zernio's post schema cannot set.
_PUBLIC_ONLY_PLATFORMS = ("instagram", "youtube")

# GET /v1/accounts returns Mongo documents, so the identifier is `_id`.
# Reading only `accountId`/`id` returned no accounts at all against three
# genuinely connected ones — the key was fine and the parse was wrong,
# which looks identical from the outside.
_ACCOUNT_ID_KEYS = ("_id", "accountId", "account_id", "id")


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
        # publish() is called once per clip per platform, and the file does
        # not change between those calls. Three clips across three networks
        # is nine uploads of three files — sixty megabytes of the same bytes
        # sent twice for nothing. Keyed on the path, for the life of the
        # publisher, which is one batch.
        self._uploaded: dict[str, str] = {}

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
            account = next(
                (str(row[k]) for k in _ACCOUNT_ID_KEYS if row.get(k)), None
            )
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

        if self.visibility != "public":
            blind = [p for p in platforms if p in _PUBLIC_ONLY_PLATFORMS]
            if blind:
                raise PublishError(
                    f"Zernio cannot post privately to {', '.join(blind)} — its "
                    "post schema has no privacy field for them, so anything "
                    "sent there goes out at the account's default. Either use "
                    "--visibility public deliberately, or drop those platforms."
                )

    def tiktok_privacy_options(self, account_id: str) -> list[str]:
        """What this TikTok account is actually allowed to post as.

        The valid values come from TikTok's creator info for the account,
        not from a constant — an unaudited app, for instance, may only be
        permitted SELF_ONLY.
        """
        try:
            payload = self._get(f"/accounts/{account_id}/tiktok/creator-info")
        except PublishError:
            return []
        for key in ("privacyLevelOptions", "privacy_level_options", "options"):
            found = payload.get(key)
            if isinstance(found, list):
                return [str(v) for v in found]
        info = payload.get("creatorInfo") or payload.get("data") or {}
        if isinstance(info, dict):
            for key in ("privacyLevelOptions", "privacy_level_options"):
                found = info.get(key)
                if isinstance(found, list):
                    return [str(v) for v in found]
        return []

    # --- media -----------------------------------------------------------

    def upload(self, path: Path) -> str:
        """Put the clip in Zernio's storage; return the URL a post can use.

        Uploaded once per file per batch. Zernio keeps a presigned upload in
        temporary storage for seven days and copies it to permanent storage
        when a post using it publishes, so one URL serves every platform.
        """
        if not path.exists():
            raise PublishError(f"Clip file is missing: {path}")
        cached = self._uploaded.get(str(path))
        if cached:
            return cached
        content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"

        # `size` is optional but pre-validated server-side: a file over the
        # limit is refused before the bytes are sent rather than after.
        presigned = self._post(
            "/media/presign",
            {
                "filename": path.name,
                "contentType": content_type,
                "size": path.stat().st_size,
            },
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
        self._uploaded[str(path)] = public_url
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

            path = Path(clip.path)
            media_url = self.upload(path)
            body: dict = {
                "content": self._caption(clip),
                "mediaItems": [
                    {
                        "type": "video",
                        "url": media_url,
                        "filename": path.name,
                        "size": path.stat().st_size,
                        "mimeType": "video/mp4",
                    }
                ],
                "platforms": [{"platform": platform, "accountId": account}],
            }
            if clip.title:
                # Reference only — Zernio's own docs say this is not used as
                # the caption. The burned-in headline is the one people see.
                body["title"] = clip.title
            if platform == "tiktok":
                wanted = _TIKTOK_PRIVACY[self.visibility]
                allowed = self.tiktok_privacy_options(account)
                if allowed and wanted not in allowed:
                    raise PublishError(
                        f"TikTok will not accept {wanted} for this account. "
                        f"It allows: {', '.join(allowed)}."
                    )
                body["tiktokSettings"] = {"privacyLevel": wanted}

            if clip.hashtags:
                # Also reference only: the spec states hashtags are NOT
                # appended to the content, which is why _caption puts them
                # in the text itself. Sent here too so they survive in
                # Zernio's record of the post.
                body["hashtags"] = [t.lstrip("#") for t in clip.hashtags]
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
            # 201 answers {message, post, warnings}; the identifier is one
            # level down, and it is a Mongo `_id` like the accounts.
            post_id = _first(created, ("_id", "id", "postId", "post_id"))
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
