"""Unattended clipping.

Two things matter more than the rest here and get the most tests: nothing
posts unless it was asked to, and one broken source does not take the batch
down with it.
"""

import json
from datetime import datetime, timezone

import pytest

from publikclip_pipeline.autopilot import publish as publish_mod
from publikclip_pipeline.autopilot import runner, select
from publikclip_pipeline.autopilot.select import SelectedClip
from publikclip_pipeline.sources.common import SourceItem


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))


def _job_dir(tmp_path, outputs, clips=None):
    job_dir = tmp_path / "job"
    job_dir.mkdir(exist_ok=True)
    for out in outputs:
        clip_file = job_dir / f"clip{out['clip']}.mp4"
        clip_file.write_bytes(b"video")
        out["path"] = str(clip_file)
    (job_dir / "render.json").write_text(json.dumps({"data": {"outputs": outputs}}))
    (job_dir / "score.json").write_text(
        json.dumps({"data": {"clips": clips or [], "confidence": "third-party"}})
    )
    return job_dir


def _out(clip, score, duration=45.0):
    return {"clip": clip, "score": score, "duration": duration, "best_platform": "reels"}


def _clip(**kw):
    base = dict(
        job_id="j1", clip=0, path=None, score=7.0, best_platform="reels",
        duration=30.0, summary="something happens", confidence="standard",
    )
    base.update(kw)
    return SelectedClip(**base)


# --- selection ------------------------------------------------------------


def test_only_clips_above_the_floor_are_selected(tmp_path):
    job_dir = _job_dir(tmp_path, [_out(0, 80.0), _out(1, 30.0), _out(2, 60.0)])
    chosen = select.select("j1", job_dir, take=5, min_score=55.0)
    assert [c.clip for c in chosen] == [0, 2]


def test_selection_is_best_first_and_capped(tmp_path):
    job_dir = _job_dir(tmp_path, [_out(0, 6.0), _out(1, 9.0), _out(2, 7.0)])
    chosen = select.select("j1", job_dir, take=2, min_score=0)
    assert [c.score for c in chosen] == [9.0, 7.0]


def test_overlong_clips_are_dropped(tmp_path):
    """Vertical clips past a minute and a half lose completion rate."""
    job_dir = _job_dir(tmp_path, [_out(0, 9.0, duration=200.0), _out(1, 6.0, duration=40.0)])
    chosen = select.select("j1", job_dir, take=5, min_score=0, max_duration=90.0)
    assert [c.clip for c in chosen] == [1]


def test_a_missing_render_file_is_not_a_crash(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert select.select("j1", empty) == []


def test_a_clip_whose_file_vanished_is_skipped(tmp_path):
    job_dir = _job_dir(tmp_path, [_out(0, 9.0), _out(1, 8.0)])
    (job_dir / "clip0.mp4").unlink()
    assert [c.clip for c in select.select("j1", job_dir, min_score=0)] == [1]


def test_the_summary_comes_from_the_matching_scored_clip(tmp_path):
    job_dir = _job_dir(
        tmp_path,
        [_out(1, 9.0)],
        clips=[{"summary": "first clip"}, {"summary": "second clip"}],
    )
    assert select.select("j1", job_dir, min_score=0)[0].summary == "second clip"


# --- captions -------------------------------------------------------------


def test_caption_is_the_scorers_own_sentence():
    assert _clip(summary="  a  funny   thing  ").caption() == "a funny thing"


def test_caption_truncates_on_a_word_boundary():
    caption = _clip(summary="word " * 60).caption(limit=20)
    assert len(caption) <= 20
    assert caption.endswith("…")
    assert "wor…" not in caption, "cut mid-word"


# --- the ledger -----------------------------------------------------------


def test_a_real_post_is_not_repeated():
    clip = _clip()
    publish_mod.record(
        publish_mod.PublishResult(clip=clip, platform="instagram", ok=True, post_id="1")
    )
    assert publish_mod.already_posted(clip, "instagram")
    assert not publish_mod.already_posted(clip, "tiktok")


def test_a_rehearsal_does_not_block_the_real_post():
    """Dry runs are recorded for auditability, never as a completed post."""
    clip = _clip()
    publish_mod.record(
        publish_mod.PublishResult(clip=clip, platform="instagram", ok=True, dry_run=True)
    )
    assert not publish_mod.already_posted(clip, "instagram")


def test_a_failed_post_can_be_retried():
    clip = _clip()
    publish_mod.record(
        publish_mod.PublishResult(clip=clip, platform="instagram", ok=False, error="429")
    )
    assert not publish_mod.already_posted(clip, "instagram")


def test_a_corrupt_ledger_does_not_stop_a_run():
    publish_mod.ledger_path().parent.mkdir(parents=True, exist_ok=True)
    publish_mod.ledger_path().write_text("{ not json")
    assert publish_mod.already_posted(_clip(), "instagram") is False


# --- publisher selection --------------------------------------------------


def test_the_default_publisher_posts_nothing():
    publisher = publish_mod.make_publisher("dry-run")
    result = publisher.publish(_clip(), "instagram")
    assert result.dry_run and result.ok
    assert result.post_id is None


def test_an_unknown_publish_mode_is_refused():
    with pytest.raises(publish_mod.PublishError):
        publish_mod.make_publisher("yolo")


def test_composio_refuses_up_front_without_a_key(monkeypatch, tmp_path):
    """Discovering there is no key after transcribing an hour of video is
    the expensive way to find out.

    Uses tiktok deliberately: instagram trips the visibility guard first,
    which would mask whether the key check runs at all.
    """
    monkeypatch.delenv(publish_mod.API_KEY_ENV, raising=False)
    with pytest.raises(publish_mod.PublishError) as err:
        publish_mod.ComposioPublisher().check_ready(["tiktok"])
    assert publish_mod.API_KEY_ENV in str(err.value)


def test_an_unsupported_platform_is_refused_before_the_batch(monkeypatch):
    monkeypatch.setenv(publish_mod.API_KEY_ENV, "key")
    with pytest.raises(publish_mod.PublishError) as err:
        publish_mod.ComposioPublisher().check_ready(["myspace"])
    assert "myspace" in str(err.value)


# --- the run loop ---------------------------------------------------------


def _item(url="https://x.invalid/a", title="A video"):
    return SourceItem(id="a", url=url, title=title, source="youtube")


def test_a_broken_source_does_not_end_the_batch(monkeypatch, tmp_path):
    processed = []

    def fake_process(item, llm_mode, captions, emit, **kwargs):
        processed.append(item.url)
        if "bad" in item.url:
            raise RuntimeError("ingest exploded")
        return "job-" + item.id

    monkeypatch.setattr(runner, "_process", fake_process)
    monkeypatch.setattr(runner, "unseen", lambda items: items)
    monkeypatch.setattr(runner, "select", lambda *a, **k: [])
    monkeypatch.setattr(runner.queue, "get_job", lambda job_id: type("J", (), {"dir": tmp_path})())

    report = runner.run(
        [_item(url="https://x.invalid/bad"), _item(url="https://x.invalid/good")]
    )
    assert len(processed) == 2, "the batch stopped at the first failure"
    assert report.failures == 1
    assert report.outcomes[0].error == "ingest exploded"
    assert report.outcomes[1].error is None


def test_nothing_is_processed_when_everything_is_already_seen(monkeypatch):
    monkeypatch.setattr(runner, "unseen", lambda items: [])
    called = []
    monkeypatch.setattr(runner, "_process", lambda *a, **k: called.append(1))
    report = runner.run([_item()])
    assert report.discovered == 0 and called == []


def test_the_destination_is_checked_before_any_processing(monkeypatch, tmp_path):
    """check_ready must run first; the pipeline is the expensive part."""
    order = []

    class _Refusing:
        name = "composio"

        def check_ready(self, platforms):
            order.append("check")
            raise publish_mod.PublishError("no connection")

        def publish(self, clip, platform):  # pragma: no cover
            raise AssertionError("must not be reached")

    monkeypatch.setattr(runner, "unseen", lambda items: items)
    monkeypatch.setattr(runner, "_process", lambda *a, **k: order.append("process") or "j")

    with pytest.raises(publish_mod.PublishError):
        runner.run([_item()], publisher=_Refusing())
    assert order == ["check"]


def test_a_dead_destination_is_reported_even_with_nothing_to_post(monkeypatch):
    """A nightly run that finds no new videos must still fail loudly on a
    broken connection — otherwise it reports success until the day it has
    something to publish."""

    class _Refusing:
        name = "composio"

        def check_ready(self, platforms):
            raise publish_mod.PublishError("no connection")

        def publish(self, clip, platform):  # pragma: no cover
            raise AssertionError("must not be reached")

    monkeypatch.setattr(runner, "unseen", lambda items: [])
    with pytest.raises(publish_mod.PublishError):
        runner.run([_item()], publisher=_Refusing())


def test_selected_clips_are_published_to_every_platform(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "unseen", lambda items: items)
    monkeypatch.setattr(runner, "_process", lambda *a, **k: "j1")
    monkeypatch.setattr(runner.queue, "get_job", lambda job_id: type("J", (), {"dir": tmp_path})())
    monkeypatch.setattr(runner, "select", lambda *a, **k: [_clip(clip=0), _clip(clip=1)])

    publisher = publish_mod.DryRunPublisher()
    report = runner.run([_item()], publisher=publisher, platforms=["instagram", "tiktok"])

    assert len(publisher.calls) == 4  # 2 clips x 2 platforms
    assert report.clips_selected == 2
    assert report.posts_ok == 4


def test_an_already_posted_clip_is_skipped(monkeypatch, tmp_path):
    clip = _clip(clip=0)
    publish_mod.record(
        publish_mod.PublishResult(clip=clip, platform="instagram", ok=True, post_id="1")
    )
    monkeypatch.setattr(runner, "unseen", lambda items: items)
    monkeypatch.setattr(runner, "_process", lambda *a, **k: "j1")
    monkeypatch.setattr(runner.queue, "get_job", lambda job_id: type("J", (), {"dir": tmp_path})())
    monkeypatch.setattr(runner, "select", lambda *a, **k: [clip])

    publisher = publish_mod.DryRunPublisher()
    runner.run([_item()], publisher=publisher, platforms=["instagram"])
    assert publisher.calls == []


# --- TikTok and YouTube ----------------------------------------------------


class _FakeSession:
    """Records tool calls and replays canned responses."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def execute(self, tool_slug, arguments):
        self.calls.append((tool_slug, arguments))
        return self.responses.get(tool_slug, {"data": {}})


def _wired(monkeypatch, responses, visibility="private"):
    publisher = publish_mod.ComposioPublisher(visibility=visibility)
    session = _FakeSession(responses)
    monkeypatch.setattr(publisher, "_connect", lambda: session)
    return publisher, session


def test_tiktok_uploads_and_publishes_in_one_call(monkeypatch):
    publisher, session = _wired(
        monkeypatch, {"TIKTOK_UPLOAD_VIDEO": {"data": {"publish_id": "pub_1"}}}
    )
    result = publisher.publish(_clip(path="/tmp/a.mp4"), "tiktok")
    assert result.ok and result.post_id == "pub_1"
    assert [slug for slug, _ in session.calls] == ["TIKTOK_UPLOAD_VIDEO"]
    args = session.calls[0][1]
    assert args["publish"] is True
    assert args["privacy_level"] == "SELF_ONLY"


def test_public_visibility_maps_to_tiktoks_public_level(monkeypatch):
    publisher, session = _wired(
        monkeypatch, {"TIKTOK_UPLOAD_VIDEO": {"data": {"publish_id": "p"}}}, visibility="public"
    )
    publisher.publish(_clip(path="/tmp/a.mp4"), "tiktok")
    assert session.calls[0][1]["privacy_level"] == "PUBLIC_TO_EVERYONE"


def test_tiktok_has_no_unlisted_so_it_falls_back_to_self_only(monkeypatch):
    publisher, session = _wired(
        monkeypatch, {"TIKTOK_UPLOAD_VIDEO": {"data": {"publish_id": "p"}}}, visibility="unlisted"
    )
    publisher.publish(_clip(path="/tmp/a.mp4"), "tiktok")
    assert session.calls[0][1]["privacy_level"] == "SELF_ONLY"


def test_an_unaudited_tiktok_app_gets_an_actionable_error(monkeypatch):
    """TikTok reports this inside a successful envelope; trusting the
    envelope would record a post that never happened."""
    publisher, _ = _wired(
        monkeypatch,
        {"TIKTOK_UPLOAD_VIDEO": {"data": {"error": {"code": "unaudited_client_can_only_post_to_private_accounts"}}}},
    )
    result = publisher.publish(_clip(path="/tmp/a.mp4"), "tiktok")
    assert not result.ok
    assert "audited" in result.error


def test_a_nested_error_is_never_reported_as_success(monkeypatch):
    publisher, _ = _wired(
        monkeypatch,
        {"TIKTOK_UPLOAD_VIDEO": {"data": {"publish_id": "p", "error": {"code": "url_ownership_unverified"}}}},
    )
    result = publisher.publish(_clip(path="/tmp/a.mp4"), "tiktok")
    assert not result.ok and "url_ownership_unverified" in result.error


def test_youtube_sends_every_required_field(monkeypatch):
    publisher, session = _wired(
        monkeypatch, {"YOUTUBE_MULTIPART_UPLOAD_VIDEO": {"data": {"id": "vid123"}}}
    )
    result = publisher.publish(_clip(path="/tmp/a.mp4", summary="a long summary"), "youtube")
    assert result.ok and result.post_id == "vid123"
    assert result.url == "https://www.youtube.com/watch?v=vid123"
    args = session.calls[0][1]
    for required in ("title", "description", "categoryId", "privacyStatus", "videoFile"):
        assert args.get(required), f"{required} missing"
    assert args["privacyStatus"] == "private"


def test_youtube_title_is_short_and_description_is_full(monkeypatch):
    publisher, session = _wired(
        monkeypatch, {"YOUTUBE_MULTIPART_UPLOAD_VIDEO": {"data": {"id": "v"}}}
    )
    long_summary = "word " * 100
    publisher.publish(_clip(path="/tmp/a.mp4", summary=long_summary), "youtube")
    args = session.calls[0][1]
    assert len(args["title"]) <= 90
    assert len(args["description"]) > len(args["title"])


def test_a_platform_returning_no_id_is_a_failure_not_a_success(monkeypatch):
    publisher, _ = _wired(monkeypatch, {"YOUTUBE_MULTIPART_UPLOAD_VIDEO": {"data": {}}})
    result = publisher.publish(_clip(path="/tmp/a.mp4"), "youtube")
    assert not result.ok and "no video id" in result.error


# --- visibility safety -----------------------------------------------------


def test_instagram_is_refused_when_the_run_asked_for_private(monkeypatch):
    """A Reel is public the moment it exists; posting one under a private
    request would be exactly the surprise the default guards against."""
    publisher = publish_mod.ComposioPublisher(visibility="private")
    monkeypatch.setattr(publisher, "_connect", lambda: _FakeSession({}))
    with pytest.raises(publish_mod.PublishError) as err:
        publisher.check_ready(["instagram"])
    assert "no private" in str(err.value)


def test_instagram_is_allowed_once_public_is_explicit(monkeypatch):
    publisher = publish_mod.ComposioPublisher(visibility="public")
    monkeypatch.setattr(publisher, "_connect", lambda: _FakeSession({}))
    publisher.check_ready(["instagram"])  # must not raise


def test_tiktok_and_youtube_are_fine_while_private(monkeypatch):
    publisher = publish_mod.ComposioPublisher(visibility="private")
    monkeypatch.setattr(publisher, "_connect", lambda: _FakeSession({}))
    publisher.check_ready(["tiktok", "youtube"])  # must not raise


def test_an_invalid_visibility_is_refused_at_construction():
    with pytest.raises(publish_mod.PublishError):
        publish_mod.ComposioPublisher(visibility="semi-public")


def test_the_default_visibility_is_private():
    assert publish_mod.make_publisher("dry-run").visibility == "private"


# --- Postiz backend --------------------------------------------------------


def _postiz(monkeypatch, tmp_path, visibility="private", **kw):
    from publikclip_pipeline.autopilot import postiz

    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    monkeypatch.setenv(postiz.API_KEY_ENV, "pk_test")
    monkeypatch.setenv(postiz.URL_ENV, "http://localhost:4007")
    return postiz.PostizPublisher(visibility=visibility, **kw), postiz


def test_postiz_is_reachable_through_make_publisher(monkeypatch, tmp_path):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    assert publish_mod.make_publisher("postiz").name == "postiz"


def test_postiz_maps_providers_to_platforms(monkeypatch, tmp_path):
    pub, postiz = _postiz(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pub, "_get",
        lambda path: [
            {"id": "i1", "providerIdentifier": "instagram-standalone"},
            {"id": "t1", "providerIdentifier": "tiktok"},
            {"id": "x1", "providerIdentifier": "x"},
        ],
    )
    assert pub.integrations() == {"instagram": "i1", "tiktok": "t1"}


def test_postiz_refuses_a_platform_with_no_connected_channel(monkeypatch, tmp_path):
    """Better than discovering it after the pipeline has run."""
    pub, _ = _postiz(monkeypatch, tmp_path, visibility="public")
    monkeypatch.setattr(pub, "_get", lambda path: [{"id": "t1", "providerIdentifier": "tiktok"}])
    with pytest.raises(publish_mod.PublishError) as err:
        pub.check_ready(["tiktok", "youtube"])
    assert "youtube" in str(err.value)
    assert "Add Channel" in str(err.value)


def test_postiz_keeps_the_instagram_visibility_guard(monkeypatch, tmp_path):
    pub, _ = _postiz(monkeypatch, tmp_path, visibility="private")
    monkeypatch.setattr(pub, "_get", lambda path: [{"id": "i1", "providerIdentifier": "instagram"}])
    with pytest.raises(publish_mod.PublishError) as err:
        pub.check_ready(["instagram"])
    assert "no private" in str(err.value)


def test_postiz_schedules_by_default_rather_than_posting_now(monkeypatch, tmp_path):
    """A queued post can be looked at and cancelled; an immediate one cannot."""
    pub, _ = _postiz(monkeypatch, tmp_path)
    kind, when = pub._when()
    assert kind == "schedule"
    assert when > datetime.now(timezone.utc).isoformat()


def test_postiz_can_be_told_to_post_now(monkeypatch, tmp_path):
    pub, _ = _postiz(monkeypatch, tmp_path, post_now=True)
    assert pub._when()[0] == "now"


def test_postiz_uploads_then_creates_the_post(monkeypatch, tmp_path):
    pub, postiz = _postiz(monkeypatch, tmp_path)
    clip_file = tmp_path / "clip.mp4"
    clip_file.write_bytes(b"video")
    monkeypatch.setattr(pub, "_get", lambda path: [{"id": "t1", "providerIdentifier": "tiktok"}])

    calls = []

    class _Res:
        status_code = 200

        def __init__(self, payload):
            self._p = payload

        def json(self):
            return self._p

    def fake_post(url, headers=None, files=None, json=None, timeout=None):
        calls.append((url.rsplit("/", 1)[-1], json))
        if url.endswith("/upload"):
            return _Res({"id": "media1", "path": "/uploads/media1.mp4"})
        return _Res({"id": "post1"})

    monkeypatch.setattr(postiz.httpx, "post", fake_post)
    result = pub.publish(_clip(path=clip_file), "tiktok")

    assert [c[0] for c in calls] == ["upload", "posts"], "the media must exist before the post"
    body = calls[1][1]
    assert body["type"] == "schedule"
    assert body["posts"][0]["integration"]["id"] == "t1"
    assert body["posts"][0]["value"][0]["image"] == [{"id": "media1", "path": "/uploads/media1.mp4"}]
    assert result.ok and result.post_id == "post1"


def test_postiz_rate_limit_is_actionable(monkeypatch, tmp_path):
    pub, postiz = _postiz(monkeypatch, tmp_path)
    clip_file = tmp_path / "clip.mp4"
    clip_file.write_bytes(b"video")
    monkeypatch.setattr(pub, "_get", lambda path: [{"id": "t1", "providerIdentifier": "tiktok"}])

    class _Res:
        def __init__(self, code, payload=None):
            self.status_code = code
            self.text = "rate limited"
            self._p = payload or {}

        def json(self):
            return self._p

    def fake_post(url, **kw):
        return _Res(200, {"id": "m"}) if url.endswith("/upload") else _Res(429)

    monkeypatch.setattr(postiz.httpx, "post", fake_post)
    result = pub.publish(_clip(path=clip_file), "tiktok")
    assert not result.ok and "--clips" in result.error


def test_postiz_says_when_the_stack_is_down(monkeypatch, tmp_path):
    pub, postiz = _postiz(monkeypatch, tmp_path)

    def boom(*a, **k):
        raise postiz.httpx.ConnectError("connection refused")

    monkeypatch.setattr(postiz.httpx, "get", boom)
    with pytest.raises(publish_mod.PublishError) as err:
        pub.integrations()
    assert "docker compose ps" in str(err.value)


# --- the selection floor's scale -------------------------------------------


def test_the_floor_sits_on_the_hundred_point_scale():
    """rubric.composite multiplies a 0-1 average by 100. The first version of
    this default read that as 0-10, so the floor never rejected anything and
    an unattended run would have published every clip it rendered."""
    assert select.DEFAULT_MIN_SCORE == 50.0


def test_a_zero_to_ten_floor_is_refused_as_a_units_mistake(tmp_path):
    """Nobody asks for a third-percentile floor on purpose."""
    with pytest.raises(ValueError) as err:
        select.select("j1", tmp_path, min_score=5.5)
    assert "0-100" in str(err.value)
    assert "55" in str(err.value), "the message should say what was meant"


def test_zero_still_disables_the_floor(tmp_path):
    job_dir = _job_dir(tmp_path, [_out(0, 3.0)])
    assert len(select.select("j1", job_dir, min_score=0)) == 1


def test_the_default_floor_keeps_the_good_clips_and_drops_the_bad(tmp_path):
    """The scores a real job produced, and the judgement a human gave them."""
    job_dir = _job_dir(
        tmp_path,
        [_out(0, 52.9), _out(1, 50.9), _out(2, 49.7), _out(3, 26.5), _out(4, 22.9)],
    )
    kept = [c.clip for c in select.select("j1", job_dir, take=10)]
    assert kept == [0, 1], "the floor should keep what the operator called good"
