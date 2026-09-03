"""Publishing through Zernio.

The failure that matters here is not a crash — it is a post that goes out
wrong, or a batch that discovers halfway through that an account was never
connected, after paying for the transcription of everything before it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from publikclip_pipeline.autopilot import zernio
from publikclip_pipeline.autopilot.publish import PublishError
from publikclip_pipeline.autopilot.select import SelectedClip


@pytest.fixture(autouse=True)
def _key(monkeypatch, tmp_path):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    monkeypatch.setenv(zernio.API_KEY_ENV, "zk_test")


def a_clip(tmp_path, **over):
    path = tmp_path / "clip_00.mp4"
    path.write_bytes(b"\x00" * 2048)
    fields = dict(
        job_id="j1", clip=0, path=path, score=52.0, best_platform="reels",
        duration=30.0, summary="ce que dit le scorer", confidence="third-party",
        title="UN TITRE", description="Ce que voit le public.",
        hashtags=("zevent", "twitch"),
    )
    fields.update(over)
    return SelectedClip(**fields)


class Fake:
    """Stands in for the whole API surface, recording what was asked."""

    def __init__(self, accounts=None, presign=None, fail=None, tiktok_options=None):
        self.tiktok_options = (
            tiktok_options
            if tiktok_options is not None
            else ["PUBLIC_TO_EVERYONE", "SELF_ONLY"]
        )
        # All three of the operator's real accounts, so a test about one
        # platform is not silently a test about a missing connection.
        self.accounts = accounts if accounts is not None else [
            {"platform": "tiktok", "accountId": "acc_tt"},
            {"platform": "instagram", "accountId": "acc_ig"},
            {"platform": "youtube", "accountId": "acc_yt"},
        ]
        self.presign = presign if presign is not None else {
            "uploadUrl": "https://upload.example/put",
            "publicUrl": "https://cdn.example/clip.mp4",
        }
        self.fail = fail or {}
        self.posts = []
        self.uploaded = []

    def install(self, monkeypatch):
        outer = self

        class Res:
            def __init__(self, code=200, payload=None, text=""):
                self.status_code = code
                self._payload = payload if payload is not None else {}
                self.text = text or json.dumps(self._payload)

            def json(self):
                return self._payload

        def get(url, **kw):
            if url.endswith("/tiktok/creator-info"):
                return Res(200, {"privacyLevelOptions": outer.tiktok_options})
            if "accounts" in outer.fail:
                return Res(outer.fail["accounts"], text="nope")
            return Res(200, {"accounts": outer.accounts})

        def post(url, **kw):
            if url.endswith("/media/presign"):
                if "presign" in outer.fail:
                    return Res(outer.fail["presign"], text="nope")
                return Res(200, outer.presign)
            if url.endswith("/posts"):
                if "posts" in outer.fail:
                    return Res(outer.fail["posts"], text="refused")
                outer.posts.append(kw.get("json"))
                return Res(200, {"id": "post_1", "url": "https://x/post_1"})
            raise AssertionError(f"unexpected POST {url}")

        def put(url, **kw):
            outer.uploaded.append(url)
            if "upload" in outer.fail:
                return Res(outer.fail["upload"], text="storage said no")
            return Res(200, {})

        monkeypatch.setattr(zernio.httpx, "get", get)
        monkeypatch.setattr(zernio.httpx, "post", post)
        monkeypatch.setattr(zernio.httpx, "put", put)
        return self


# --- the key -------------------------------------------------------------

def test_no_key_says_where_to_put_one(monkeypatch):
    monkeypatch.delenv(zernio.API_KEY_ENV, raising=False)
    with pytest.raises(PublishError) as err:
        zernio.ZernioPublisher()
    message = str(err.value)
    assert zernio.API_KEY_SECRET in message
    assert "secrets.json" in message


def test_a_rejected_key_is_reported_as_a_key_problem(monkeypatch):
    Fake(fail={"accounts": 401}).install(monkeypatch)
    with pytest.raises(PublishError) as err:
        zernio.ZernioPublisher().accounts()
    assert "rejected the API key" in str(err.value)


# --- failing before the batch -------------------------------------------

def test_an_unconnected_platform_is_caught_before_any_work(monkeypatch):
    """The whole point of check_ready.

    Discovering that TikTok was never connected after transcribing an hour
    of video means the expensive part is already paid for.
    """
    Fake(accounts=[{"platform": "instagram", "accountId": "acc_ig"}]).install(monkeypatch)
    with pytest.raises(PublishError) as err:
        zernio.ZernioPublisher().check_ready(["instagram", "tiktok"])
    assert "tiktok" in str(err.value)
    assert "instagram" in str(err.value), "say what IS connected, not only what is not"


def test_no_accounts_at_all_says_to_connect_them(monkeypatch):
    Fake(accounts=[]).install(monkeypatch)
    with pytest.raises(PublishError) as err:
        zernio.ZernioPublisher().check_ready(["tiktok"])
    assert "no connected accounts" in str(err.value)


def test_a_format_name_is_still_rejected(monkeypatch):
    # `reels` is what a clip is scored for, not where it is posted.
    Fake().install(monkeypatch)
    with pytest.raises(PublishError) as err:
        zernio.ZernioPublisher().check_ready(["reels"])
    assert "instagram" in str(err.value)


def test_accounts_are_fetched_once(monkeypatch):
    fake = Fake().install(monkeypatch)
    calls = []
    original = zernio.httpx.get
    monkeypatch.setattr(
        zernio.httpx, "get",
        lambda url, **kw: (calls.append(url), original(url, **kw))[1],
    )
    pub = zernio.ZernioPublisher()
    pub.check_ready(["tiktok"])
    pub.accounts()
    pub.accounts()
    assert len(calls) == 1


# --- the media flow ------------------------------------------------------

def test_the_clip_is_uploaded_then_referenced_by_its_public_url(monkeypatch, tmp_path):
    """A rendered clip lives on a private VM with no public address, so the
    presign URL it is PUT to and the URL the post carries are different."""
    fake = Fake().install(monkeypatch)
    result = zernio.ZernioPublisher().publish(a_clip(tmp_path), "tiktok")

    assert result.ok, result.error
    assert fake.uploaded == ["https://upload.example/put"]

    media = fake.posts[0]["mediaItems"][0]
    # The two URLs are different things and conflating them would only fail
    # against the live API: bytes go to the presigned one, the post carries
    # the public one.
    assert media["url"] == "https://cdn.example/clip.mp4"
    assert media["url"] not in fake.uploaded
    assert media["type"] == "video"
    assert media["filename"] == "clip_00.mp4"
    assert media["size"] == 2048


def test_the_presign_declares_the_size_for_pre_validation(monkeypatch, tmp_path):
    """Optional, and worth sending: an oversized file is refused before the
    bytes go over the wire rather than after."""
    fake = Fake().install(monkeypatch)
    sent = {}
    original = zernio.httpx.post

    def spy(url, **kw):
        if url.endswith("/media/presign"):
            sent.update(kw.get("json") or {})
        return original(url, **kw)

    monkeypatch.setattr(zernio.httpx, "post", spy)
    zernio.ZernioPublisher().publish(a_clip(tmp_path), "tiktok")
    assert sent["size"] == 2048
    assert sent["contentType"] == "video/mp4"


def test_the_account_id_is_read_from_a_mongo_document(monkeypatch):
    """What the live API actually returns.

    GET /v1/accounts answers Mongo documents keyed on `_id`. Reading only
    `accountId`/`id` found nothing against three genuinely connected
    accounts — and an empty result looks exactly like "nothing connected",
    which is the wrong thing to tell someone.
    """
    Fake(accounts=[{"_id": "6a99b54a77555aae01d285e5", "platform": "instagram"}]).install(
        monkeypatch
    )
    assert zernio.ZernioPublisher().accounts() == {
        "instagram": "6a99b54a77555aae01d285e5"
    }


def test_the_reference_only_fields_travel_too(monkeypatch, tmp_path):
    """Zernio stores `title` and `hashtags` but does not put either into the
    caption, which is why the caption carries the tags itself."""
    fake = Fake().install(monkeypatch)
    zernio.ZernioPublisher().publish(a_clip(tmp_path), "tiktok")
    body = fake.posts[0]
    assert body["title"] == "UN TITRE"
    assert body["hashtags"] == ["zevent", "twitch"]
    assert "#zevent" in body["content"], "tags must also be in the visible text"


def test_a_presign_without_the_urls_says_what_came_back(monkeypatch, tmp_path):
    """Guessing a field name and failing with a KeyError months later tells
    nobody anything."""
    Fake(presign={"somethingElse": 1}).install(monkeypatch)
    result = zernio.ZernioPublisher().publish(a_clip(tmp_path), "tiktok")
    assert not result.ok
    assert "somethingElse" in result.error


def test_the_url_fields_are_accepted_under_their_likely_names(monkeypatch, tmp_path):
    Fake(presign={"data": {"signedUrl": "https://u/1", "public_url": "https://p/1"}}).install(
        monkeypatch
    )
    fake_result = zernio.ZernioPublisher().publish(a_clip(tmp_path), "tiktok")
    assert fake_result.ok, fake_result.error


def test_storage_refusing_the_file_is_reported_not_swallowed(monkeypatch, tmp_path):
    Fake(fail={"upload": 403}).install(monkeypatch)
    result = zernio.ZernioPublisher().publish(a_clip(tmp_path), "tiktok")
    assert not result.ok
    assert "403" in result.error


def test_a_missing_clip_file_fails_before_the_network(monkeypatch, tmp_path):
    fake = Fake().install(monkeypatch)
    clip = a_clip(tmp_path)
    Path(clip.path).unlink()
    result = zernio.ZernioPublisher().publish(clip, "tiktok")
    assert not result.ok
    assert "missing" in result.error
    assert fake.uploaded == []


# --- what the post says --------------------------------------------------

def test_the_post_uses_the_copy_written_for_an_audience(monkeypatch, tmp_path):
    """The render stage writes a description from the clip's own transcript.
    The scorer's summary describes the clip for whoever is deciding whether
    to publish it, which is a different job."""
    fake = Fake().install(monkeypatch)
    zernio.ZernioPublisher().publish(a_clip(tmp_path), "tiktok")
    content = fake.posts[0]["content"]
    assert "Ce que voit le public." in content
    assert "ce que dit le scorer" not in content


def test_hashtags_travel_with_the_post(monkeypatch, tmp_path):
    fake = Fake().install(monkeypatch)
    zernio.ZernioPublisher().publish(a_clip(tmp_path), "tiktok")
    assert "#zevent" in fake.posts[0]["content"]
    assert "#twitch" in fake.posts[0]["content"]


def test_a_clip_with_no_generated_copy_falls_back_to_the_summary(monkeypatch, tmp_path):
    # Half the clips get no title: no band free of faces, so no headline.
    fake = Fake().install(monkeypatch)
    clip = a_clip(tmp_path, title="", description="", hashtags=())
    zernio.ZernioPublisher().publish(clip, "tiktok")
    assert fake.posts[0]["content"] == "ce que dit le scorer"


def test_the_post_names_the_right_account_for_the_platform(monkeypatch, tmp_path):
    fake = Fake().install(monkeypatch)
    pub = zernio.ZernioPublisher()
    pub.publish(a_clip(tmp_path), "instagram")
    assert fake.posts[0]["platforms"] == [
        {"platform": "instagram", "accountId": "acc_ig"}
    ]


# --- when it goes out ----------------------------------------------------

def test_it_publishes_now_by_default(monkeypatch, tmp_path):
    fake = Fake().install(monkeypatch)
    zernio.ZernioPublisher().publish(a_clip(tmp_path), "tiktok")
    assert fake.posts[0]["publishNow"] is True
    assert "scheduledFor" not in fake.posts[0]


def test_scheduling_sends_a_timestamp_and_a_timezone(monkeypatch, tmp_path):
    fake = Fake().install(monkeypatch)
    zernio.ZernioPublisher(schedule_in_minutes=60).publish(a_clip(tmp_path), "tiktok")
    body = fake.posts[0]
    assert body["scheduledFor"].endswith("Z")
    assert body["timezone"] == "UTC"
    assert "publishNow" not in body


def test_a_refused_post_comes_back_as_a_result_not_an_exception(monkeypatch, tmp_path):
    """One bad post must not end a batch of ten."""
    Fake(fail={"posts": 422}).install(monkeypatch)
    result = zernio.ZernioPublisher().publish(a_clip(tmp_path), "tiktok")
    assert not result.ok
    assert "422" in result.error


def test_a_successful_post_carries_its_id_back(monkeypatch, tmp_path):
    Fake().install(monkeypatch)
    result = zernio.ZernioPublisher().publish(a_clip(tmp_path), "tiktok")
    assert result.post_id == "post_1"
    assert result.url == "https://x/post_1"


# --- visibility ----------------------------------------------------------

def test_a_private_post_tells_tiktok_it_is_private(monkeypatch, tmp_path):
    """The bug that shipped: the publisher recorded a visibility and never
    sent one, so a post asked for as private went out public and the setting
    looked like it had worked."""
    fake = Fake().install(monkeypatch)
    zernio.ZernioPublisher(visibility="private").publish(a_clip(tmp_path), "tiktok")
    assert fake.posts[0]["tiktokSettings"] == {"privacyLevel": "SELF_ONLY"}


def test_a_public_post_says_so_too(monkeypatch, tmp_path):
    fake = Fake().install(monkeypatch)
    zernio.ZernioPublisher(visibility="public").publish(a_clip(tmp_path), "tiktok")
    assert fake.posts[0]["tiktokSettings"] == {"privacyLevel": "PUBLIC_TO_EVERYONE"}


def test_a_privacy_level_the_account_cannot_use_is_refused(monkeypatch, tmp_path):
    """An unaudited TikTok app may only be allowed SELF_ONLY. Sending
    something else would be rejected by TikTok after the upload."""
    Fake(tiktok_options=["SELF_ONLY"]).install(monkeypatch)
    result = zernio.ZernioPublisher(visibility="public").publish(
        a_clip(tmp_path), "tiktok"
    )
    assert not result.ok
    assert "SELF_ONLY" in result.error


def test_an_account_that_reports_no_options_is_not_blocked(monkeypatch, tmp_path):
    # Absence of information is not a refusal; TikTok still validates.
    fake = Fake(tiktok_options=[]).install(monkeypatch)
    result = zernio.ZernioPublisher(visibility="private").publish(
        a_clip(tmp_path), "tiktok"
    )
    assert result.ok, result.error
    assert fake.posts[0]["tiktokSettings"] == {"privacyLevel": "SELF_ONLY"}


def test_instagram_and_youtube_refuse_a_private_run_before_it_starts(monkeypatch):
    """Zernio's post schema has no privacy field for either, so a "private"
    post there would go out at the account's default. Refusing is honest;
    publishing publicly while reporting private is not.
    """
    Fake().install(monkeypatch)
    for platform in ("instagram", "youtube"):
        with pytest.raises(PublishError) as err:
            zernio.ZernioPublisher(visibility="private").check_ready([platform])
        assert platform in str(err.value)
        assert "public" in str(err.value), "say what the alternative is"


def test_those_platforms_are_fine_when_public_is_asked_for_deliberately(monkeypatch):
    Fake().install(monkeypatch)
    zernio.ZernioPublisher(visibility="public").check_ready(
        ["instagram", "youtube", "tiktok"]
    )


def test_tiktok_alone_can_still_run_privately(monkeypatch):
    Fake().install(monkeypatch)
    zernio.ZernioPublisher(visibility="private").check_ready(["tiktok"])
