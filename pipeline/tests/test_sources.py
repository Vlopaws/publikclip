"""Discovery: what gets proposed for clipping, before any download.

Everything here is offline. The network paths are exercised by hand; what
these lock in is the logic that decides which candidates survive, because a
bad filter either wastes hours of compute or silently skips good material.
"""

import pytest

from publikclip_pipeline.sources import common, opportunity, twitch, youtube
from publikclip_pipeline.sources.common import SourceItem


def _item(**kw):
    base = dict(id="abc", url="https://example.invalid/abc", title="t", source="youtube")
    base.update(kw)
    return SourceItem(**base)


# --- channel reference normalisation --------------------------------------


@pytest.mark.parametrize(
    "ref, expected",
    [
        ("@Handle", "https://www.youtube.com/@Handle/videos"),
        ("Handle", "https://www.youtube.com/@Handle/videos"),
        ("  @Handle  ", "https://www.youtube.com/@Handle/videos"),
        ("https://www.youtube.com/@Handle", "https://www.youtube.com/@Handle/videos"),
        ("https://www.youtube.com/@Handle/", "https://www.youtube.com/@Handle/videos"),
        # Already on a tab: must not stack a second one.
        ("https://www.youtube.com/@Handle/videos", "https://www.youtube.com/@Handle/videos"),
        ("https://www.youtube.com/@Handle/streams", "https://www.youtube.com/@Handle/videos"),
        (
            "UCXuqSBlHAE6Xw-yeJA0Tunw",
            "https://www.youtube.com/channel/UCXuqSBlHAE6Xw-yeJA0Tunw/videos",
        ),
    ],
)
def test_channel_url_normalises_every_reference_form(ref, expected):
    assert youtube.channel_url(ref) == expected


def test_a_display_name_loses_its_spaces():
    """Handles cannot contain whitespace; "Sans Permission" has to become
    @SansPermission or the URL cannot resolve at all."""
    assert youtube.channel_url("Sans Permission") == "https://www.youtube.com/@SansPermission/videos"


def test_channel_url_can_target_another_tab():
    assert youtube.channel_url("@Handle", tab="streams").endswith("/streams")


def test_channel_url_rejects_an_empty_reference():
    with pytest.raises(ValueError):
        youtube.channel_url("   ")


# --- duration filtering ---------------------------------------------------


def test_within_duration_drops_both_ends():
    items = [_item(id="short", duration_sec=30), _item(id="ok", duration_sec=600),
             _item(id="long", duration_sec=20000)]
    kept = [i.id for i in common.within_duration(items, 120, 4 * 3600)]
    assert kept == ["ok"]


def test_unknown_duration_is_kept_for_ingest_to_judge():
    """Flat listings occasionally omit duration; skipping silently would
    lose good material for a missing field."""
    kept = common.within_duration([_item(duration_sec=None)], 120, 3600)
    assert len(kept) == 1


def test_open_ended_bounds_are_allowed():
    items = [_item(id="a", duration_sec=5), _item(id="b", duration_sec=99999)]
    assert len(common.within_duration(items, None, None)) == 2


# --- entry mapping --------------------------------------------------------


def test_youtube_entry_maps_to_an_item():
    item = youtube._entry_to_item(
        {"id": "vid1", "title": "Hello", "duration": 812, "view_count": 1234}, "Chan"
    )
    assert item.url == "https://www.youtube.com/watch?v=vid1"
    assert item.duration_sec == 812.0
    assert item.channel == "Chan"
    assert item.source == "youtube"


def test_entry_without_an_id_is_dropped():
    """A private or removed video comes back as an entry with no id."""
    assert youtube._entry_to_item({"title": "ghost"}, "Chan") is None


def test_twitch_entry_falls_back_to_the_clip_url():
    item = twitch._entry_to_item({"id": "9876", "title": "Nice", "duration": 41.0}, "streamer")
    assert item.url == "https://clips.twitch.tv/9876"
    assert item.source == "twitch"
    assert item.channel == "streamer"


# --- deduplication against the job queue ----------------------------------


def test_unseen_drops_what_the_queue_already_has(monkeypatch):
    class _Job:
        def __init__(self, source):
            self.source = source

    items = [_item(id="a", url="https://x.invalid/a"), _item(id="b", url="https://x.invalid/b")]
    monkeypatch.setattr(
        common.queue, "list_jobs", lambda limit=500: [_Job("https://x.invalid/a")]
    )
    assert [i.id for i in common.unseen(items)] == ["b"]


def test_unseen_keeps_everything_when_the_queue_is_empty(monkeypatch):
    monkeypatch.setattr(common.queue, "list_jobs", lambda limit=500: [])
    items = [_item(id="a"), _item(id="b", url="https://x.invalid/b")]
    assert len(common.unseen(items)) == 2


# --- Twitch credentials ---------------------------------------------------


def test_category_mode_explains_exactly_what_is_missing(monkeypatch, tmp_path):
    """There is no anonymous route for browse-by-category, so the error has
    to name the credentials and point at the alternative."""
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    monkeypatch.delenv(twitch.CLIENT_ID_ENV, raising=False)
    monkeypatch.delenv(twitch.CLIENT_SECRET_ENV, raising=False)
    with pytest.raises(twitch.TwitchError) as err:
        twitch.category_clips("Just Chatting")
    message = str(err.value)
    assert twitch.CLIENT_ID_ENV in message
    assert "dev.twitch.tv" in message
    assert "channel_clips" in message


def test_half_configured_credentials_count_as_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    monkeypatch.setenv(twitch.CLIENT_ID_ENV, "id-only")
    monkeypatch.delenv(twitch.CLIENT_SECRET_ENV, raising=False)
    assert twitch.credentials() is None


def test_credentials_are_read_when_both_are_present(monkeypatch, tmp_path):
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    monkeypatch.setenv(twitch.CLIENT_ID_ENV, "cid")
    monkeypatch.setenv(twitch.CLIENT_SECRET_ENV, "secret")
    assert twitch.credentials() == ("cid", "secret")


# --- presentation ---------------------------------------------------------


def test_summary_is_readable_without_optional_fields():
    assert _item(title="Bare").summary() == "Bare"


def test_summary_formats_duration_and_views():
    text = _item(title="X", duration_sec=125, view_count=1234).summary()
    assert "2:05" in text and "1 234" in text


# --- clip-scene saturation ------------------------------------------------
# The heuristic decides whether a creator is worth pursuing, so its edges
# matter more than its centre.


def _entries(*rows):
    return [{"channel": c, "title": t, "view_count": v} for c, t, v in rows]


def _scan(monkeypatch, rows, creator="Someone"):
    monkeypatch.setattr(opportunity.ytdlp, "ensure_ytdlp", lambda progress: "yt-dlp")
    calls = []

    def fake_search(binary, query, limit):
        calls.append(query)
        return rows if len(calls) == 1 else []

    monkeypatch.setattr(opportunity, "_search", fake_search)
    report = opportunity.clip_saturation(creator)
    return report, calls


def test_channels_that_do_not_announce_themselves_are_ignored(monkeypatch):
    report, _ = _scan(monkeypatch, _entries(("Some News Channel", "a", 10)))
    assert report.verdict == "open"
    assert report.results_scanned == 1


def test_clip_channels_are_recognised_across_languages(monkeypatch):
    rows = _entries(
        ("Zerator Clips", "a", 100),
        ("Best Of du Grenier", "b", 200),
        ("Extraits Podcast", "c", 300),
        ("Zapping du COVID", "d", 400),
    )
    report, _ = _scan(monkeypatch, rows)
    assert report.dedicated_channels == 4


def test_the_creators_own_channel_is_not_competition(monkeypatch):
    """A creator posting their own best-of does not make the lane crowded."""
    rows = _entries(("Best Of Someone", "a", 10), ("Someone", "own upload", 999))
    report, _ = _scan(monkeypatch, rows, creator="Someone")
    assert [c.name for c in report.clip_channels] == ["Best Of Someone"]


def test_top_views_tracks_the_best_clip_not_the_last(monkeypatch):
    rows = _entries(("X Clips", "small", 10), ("X Clips", "huge", 5000), ("X Clips", "mid", 100))
    report, _ = _scan(monkeypatch, rows)
    channel = report.clip_channels[0]
    assert channel.top_views == 5000
    assert channel.top_title == "huge"
    assert channel.total_views == 5110
    assert channel.videos_found == 3


def test_verdicts_span_the_three_cases(monkeypatch):
    open_report, _ = _scan(monkeypatch, [])
    assert open_report.verdict == "open"

    thin_report, _ = _scan(monkeypatch, _entries(("A Clips", "t", 5_000)))
    assert thin_report.verdict == "thin"

    # A six-figure clip means the audience is already being served.
    crowded_report, _ = _scan(monkeypatch, _entries(("A Clips", "t", 500_000)))
    assert crowded_report.verdict == "crowded"


def test_three_channels_is_crowded_even_when_small(monkeypatch):
    rows = _entries(("A Clips", "t", 10), ("B Clips", "t", 10), ("C Clips", "t", 10))
    report, _ = _scan(monkeypatch, rows)
    assert report.verdict == "crowded"


def test_every_query_template_is_used(monkeypatch):
    _, calls = _scan(monkeypatch, [], creator="Someone")
    assert len(calls) == len(opportunity._QUERIES)
    assert all("Someone" in q for q in calls)


def test_a_handle_is_stripped_before_searching(monkeypatch):
    _, calls = _scan(monkeypatch, [], creator="@Someone")
    assert all("@" not in q for q in calls)


# --- demand: reach, not just competition ----------------------------------


def test_median_ignores_a_single_viral_outlier():
    """Mean would let one hit make a quiet channel look big."""
    assert opportunity._median([1000, 1100, 1200, 900, 5_000_000]) == 1100


def test_median_of_nothing_is_unknown_not_zero():
    assert opportunity._median([]) is None


def test_median_averages_the_middle_pair_when_even():
    assert opportunity._median([10, 20, 30, 40]) == 25


def test_resolve_channel_takes_the_majority(monkeypatch):
    """Search returns the creator's videos plus stray re-uploads; the
    channel that appears most is the answer."""
    monkeypatch.setattr(opportunity.ytdlp, "ensure_ytdlp", lambda progress: "yt-dlp")
    monkeypatch.setattr(
        opportunity, "_search",
        lambda b, q, n: [
            {"channel_id": "UCreal"}, {"channel_id": "UCreal"},
            {"channel_id": "UCfan"}, {"channel_id": "UCreal"},
        ],
    )
    assert opportunity.resolve_channel("Someone") == "UCreal"


def test_resolve_channel_admits_when_it_cannot(monkeypatch):
    monkeypatch.setattr(opportunity.ytdlp, "ensure_ytdlp", lambda progress: "yt-dlp")
    monkeypatch.setattr(opportunity, "_search", lambda b, q, n: [{"title": "no channel"}])
    assert opportunity.resolve_channel("Someone") is None


def test_an_explicit_handle_skips_resolution(monkeypatch):
    """A caller who already knows the handle must not have their answer
    overridden by a search guess."""
    called = []
    monkeypatch.setattr(
        opportunity, "resolve_channel", lambda *a, **k: called.append(1) or "UCwrong"
    )
    monkeypatch.setattr(
        opportunity.youtube_source, "recent_uploads",
        lambda channel, **kw: [_item(view_count=100, id="a")],
    )
    median, seen, _ = opportunity.audience("@Known")
    assert called == [], "an explicit handle was re-resolved"
    assert (median, seen) == (100, 1)


def test_audience_reports_the_channel_it_measured(monkeypatch):
    monkeypatch.setattr(
        opportunity.youtube_source, "recent_uploads",
        lambda channel, **kw: [_item(view_count=100, channel="Real Channel")],
    )
    _, _, measured = opportunity.audience("@Known")
    assert measured == "Real Channel"


def test_an_unreachable_channel_is_unknown_not_zero(monkeypatch):
    def boom(channel, **kw):
        raise RuntimeError("404")

    monkeypatch.setattr(opportunity.youtube_source, "recent_uploads", boom)
    assert opportunity.audience("@Gone") == (None, 0, None)


@pytest.mark.parametrize(
    "median, saturation_verdict, expected",
    [
        (None, "open", "unknown"),
        (500, "open", "too small"),          # nobody clips them, nobody watches them
        (50_000, "open", "worth a look"),
        (500_000, "thin", "sweet spot"),     # real reach, nobody serving it
        (500_000, "crowded", "taken"),       # reach is real but the lane is full
    ],
)
def test_the_two_axes_combine_into_one_verdict(median, saturation_verdict, expected):
    class _Sat:
        verdict = saturation_verdict

    assessment = opportunity.Opportunity(
        creator="X", saturation=_Sat(), median_views=median, uploads_seen=10
    )
    assert assessment.verdict == expected


# --- CLI argument handling -------------------------------------------------


def test_duration_bounds_are_optional_on_subcommands_that_lack_them():
    """`scan` measures a creator rather than filtering a list, so it defines
    no --min-duration. Reading it unconditionally made the command fail
    before it started."""
    import argparse

    from publikclip_pipeline import cli

    args = argparse.Namespace(
        source_cmd="scan", creator="X", per_query=1, channel=None,
        no_audience=True, json=True, new_only=False,
    )
    calls = {}

    class _Report:
        verdict = "open"

        @staticmethod
        def to_json():
            return {"verdict": "open"}

    def fake_saturation(creator, per_query=12, progress=None):
        calls["creator"] = creator
        return _Report()

    import publikclip_pipeline.sources.opportunity as opp

    original = opp.clip_saturation
    opp.clip_saturation = fake_saturation
    try:
        assert cli.cmd_sources(args) == 0
    finally:
        opp.clip_saturation = original
    assert calls["creator"] == "X"
