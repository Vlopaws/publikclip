"""Pooling many channels into one ranked list.

Built for an event: fifty streamers live for three days, and the question is
never "the best clips from Zerator" but "the best clips from any of them,
right now". Two things must hold for that to be usable — the pool is ranked
as one, and one unreachable channel out of fifty does not take the batch
with it.
"""

from __future__ import annotations

import pytest

from publikclip_pipeline.sources import roster
from publikclip_pipeline.sources.common import SourceItem


def clip(channel, views, clip_id=None):
    return SourceItem(
        id=clip_id or f"{channel}-{views}",
        url=f"https://www.twitch.tv/{channel}/clip/{views}",
        title=f"{channel} {views}",
        source="twitch",
        duration_sec=30.0,
        view_count=views,
        channel=channel,
    )


# --- reading the file ----------------------------------------------------

def test_a_plain_list_is_read_in_order():
    assert roster.parse("zerator\nmistermv\ndomingo") == [
        "zerator", "mistermv", "domingo"
    ]


def test_comments_and_blank_lines_are_ignored():
    text = "# the roster\n\nzerator  # on site\n\n  domingo\n#mistermv\n"
    assert roster.parse(text) == ["zerator", "domingo"]


def test_urls_and_at_handles_reduce_to_the_channel_name():
    text = "https://www.twitch.tv/domingo/\n@zerator\ntwitch.tv/mistermv"
    assert roster.parse(text) == ["domingo", "zerator", "mistermv"]


def test_the_same_channel_written_two_ways_appears_once():
    # Pasting from two places is how a roster actually gets built.
    assert roster.parse("zerator\n@zerator\nhttps://twitch.tv/ZeratoR") == ["zerator"]


def test_order_is_preserved_because_it_often_means_priority():
    assert roster.parse("ccc\naaa\nbbb") == ["ccc", "aaa", "bbb"]


def test_something_that_is_not_a_channel_name_is_dropped():
    # A Twitch login is 3-25 characters of letters, digits and underscore.
    text = "zerator\nnot a channel\nx\n" + "z" * 40 + "\nok_name"
    assert roster.parse(text) == ["zerator", "ok_name"]


def test_a_runaway_file_is_capped():
    names = roster.parse("\n".join(f"chan{i:04d}" for i in range(500)))
    assert len(names) == roster.MAX_CHANNELS


def test_an_empty_roster_is_empty_not_an_error():
    assert roster.parse("") == []
    assert roster.parse("# nothing but comments\n\n") == []


# --- sweeping ------------------------------------------------------------

def fake_channels(monkeypatch, table, failing=()):
    def fake(name, limit=5, **kwargs):
        if name in failing:
            raise RuntimeError(f"{name} is unreachable")
        return table.get(name, [])[:limit]

    monkeypatch.setattr(roster, "channel_clips", fake)


def test_the_pool_is_ranked_across_channels_not_within_them(monkeypatch):
    """The whole point.

    Ranking per channel and interleaving would give a quiet streamer's best
    clip the same standing as the event's biggest moment.
    """
    fake_channels(monkeypatch, {
        "quiet": [clip("quiet", 40), clip("quiet", 30)],
        "busy": [clip("busy", 9000), clip("busy", 8000)],
    })
    out = roster.sweep(["quiet", "busy"], workers=2)
    assert [i.view_count for i in out.items] == [9000, 8000, 40, 30]


def test_one_unreachable_channel_does_not_lose_the_others(monkeypatch):
    fake_channels(
        monkeypatch,
        {"good": [clip("good", 100)], "alsogood": [clip("alsogood", 50)]},
        failing=("broken",),
    )
    out = roster.sweep(["good", "broken", "alsogood"], workers=3)
    assert [i.view_count for i in out.items] == [100, 50]
    assert "broken" in out.failed
    assert sorted(out.reached) == ["alsogood", "good"]
    assert out.channels == 3


def test_every_failure_is_reported_not_just_the_first(monkeypatch):
    fake_channels(monkeypatch, {"ok": [clip("ok", 1)]}, failing=("a", "b", "c"))
    out = roster.sweep(["a", "ok", "b", "c"], workers=4)
    assert set(out.failed) == {"a", "b", "c"}
    assert out.items


def test_a_channel_that_exists_but_has_nothing_is_not_a_failure(monkeypatch):
    # yt-dlp answers an empty playlist for a channel with no recent clips,
    # and for one that does not exist. Neither is an error worth reporting.
    fake_channels(monkeypatch, {"live": [clip("live", 10)], "silent": []})
    out = roster.sweep(["live", "silent"], workers=2)
    assert out.failed == {}
    assert len(out.items) == 1


def test_per_channel_bounds_what_each_contributes(monkeypatch):
    fake_channels(monkeypatch, {
        "a": [clip("a", n) for n in (500, 400, 300, 200, 100)],
        "b": [clip("b", n) for n in (450, 350)],
    })
    out = roster.sweep(["a", "b"], per_channel=2, workers=2)
    assert len(out.items) == 4
    assert [i.view_count for i in out.items] == [500, 450, 400, 350]


def test_an_empty_roster_sweeps_nothing_without_touching_the_network(monkeypatch):
    monkeypatch.setattr(
        roster, "channel_clips",
        lambda *a, **k: pytest.fail("listed a channel that was not in the roster"),
    )
    out = roster.sweep([])
    assert out.items == [] and out.channels == 0


def test_a_clip_with_no_view_count_still_ranks(monkeypatch):
    """Ranking must not raise on a missing field; unknown sorts last.

    yt-dlp's flat listing does not always carry view_count, and a roster
    sweep that crashed on one such clip would lose every channel with it.
    """
    missing = SourceItem(
        id="x-none",
        url="https://www.twitch.tv/x/clip/none",
        title="no views recorded",
        source="twitch",
        duration_sec=30.0,
        view_count=None,
        channel="x",
    )
    fake_channels(monkeypatch, {"x": [missing], "y": [clip("y", 5)]})
    out = roster.sweep(["x", "y"], workers=2)
    assert [i.channel for i in out.items] == ["y", "x"]


def test_progress_reports_each_channel_as_it_lands(monkeypatch):
    fake_channels(monkeypatch, {"a": [clip("a", 1)], "b": [clip("b", 2)]})
    seen = []
    roster.sweep(["a", "b"], progress=lambda f, m: seen.append(m), workers=2)
    assert len(seen) == 2
    assert any("a:" in m for m in seen)
