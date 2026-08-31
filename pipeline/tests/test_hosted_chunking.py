"""Splitting long audio for a hosted transcriber with an upload limit.

The failure this prevents is not subtle — a two-hour interview simply could
not be transcribed — but the failures it introduces would be: a gap between
parts loses speech silently, and a bad overlap reconciliation duplicates it.
Both produce a transcript that looks fine until someone reads it.
"""

from __future__ import annotations

import pytest

from publikclip_pipeline.asr import hosted

LIMIT = hosted.UPLOAD_LIMIT_BYTES


def plan(duration, mb):
    return hosted.chunk_plan(duration, int(mb * 1_000_000))


# --- planning ------------------------------------------------------------

def test_audio_that_already_fits_is_sent_whole():
    assert plan(1200, 20) == [(0.0, 1200)]


def test_a_zero_length_file_does_not_loop_forever():
    assert hosted.chunk_plan(0, 999_000_000) == [(0.0, 0)]


@pytest.mark.parametrize(
    "duration,mb",
    [(3768, 59), (7200, 120), (10800, 180), (1800, 40), (600, 30)],
)
def test_every_second_of_audio_lands_in_some_part(duration, mb):
    spans = plan(duration, mb)
    assert spans[0][0] == 0.0
    assert spans[-1][0] + spans[-1][1] == pytest.approx(duration)
    for prev, nxt in zip(spans, spans[1:]):
        # The next part must begin before the previous one ends, or the
        # speech in between is never sent to anybody.
        assert nxt[0] < prev[0] + prev[1]


@pytest.mark.parametrize("duration,mb", [(3768, 59), (7200, 120), (10800, 180)])
def test_no_part_would_be_refused_by_the_upload_limit(duration, mb):
    bytes_per_sec = mb * 1_000_000 / duration
    for _, length in plan(duration, mb):
        assert length * bytes_per_sec <= LIMIT


def test_consecutive_parts_actually_overlap():
    spans = plan(7200, 120)
    for prev, nxt in zip(spans, spans[1:]):
        overlap = (prev[0] + prev[1]) - nxt[0]
        assert overlap == pytest.approx(hosted.CHUNK_OVERLAP_SEC)


def test_a_short_tail_is_folded_in_rather_than_sent_alone():
    # A remainder under MIN_CHUNK_SEC would otherwise be its own request,
    # billed at the 10-second minimum for a few seconds of speech.
    spans = plan(3600, 60)
    assert spans[-1][1] >= hosted.MIN_CHUNK_SEC


def test_a_pathological_byte_rate_still_terminates():
    # Absurdly dense audio would ask for sub-second parts; the floor keeps
    # the plan finite instead of producing tens of thousands of requests.
    spans = hosted.chunk_plan(600, 10_000_000_000)
    assert len(spans) < 100
    assert all(length >= hosted.MIN_CHUNK_SEC for _, length in spans[:-1])


# --- stitching -----------------------------------------------------------

def seg(start, end, *words):
    return {
        "start": start,
        "end": end,
        "text": " ".join(words),
        "words": [
            {"word": w, "start": start + i, "end": start + i + 1, "score": 0.0}
            for i, w in enumerate(words)
        ],
    }


def test_a_later_part_is_shifted_into_absolute_time():
    out = []
    hosted._merge(out, [seg(0.0, 3.0, "a", "b", "c")], offset=600.0)
    assert out[0]["start"] == 600.0
    assert out[0]["end"] == 603.0
    assert out[0]["words"][0]["start"] == 600.0


def test_speech_heard_twice_across_a_seam_is_kept_once():
    out = []
    hosted._merge(out, [seg(0.0, 10.0, "premiere", "partie")], offset=0.0)
    # The next part starts 2 s before the previous ended, so its opening
    # segment repeats what is already recorded.
    hosted._merge(out, [seg(0.0, 4.0, "partie"), seg(4.0, 9.0, "suite")], offset=8.0)
    assert [s["text"] for s in out] == ["premiere partie", "suite"]


def test_a_seam_does_not_swallow_genuinely_new_speech():
    out = []
    hosted._merge(out, [seg(0.0, 10.0, "un")], offset=0.0)
    hosted._merge(out, [seg(2.1, 6.0, "deux")], offset=8.0)  # starts at 10.1
    assert [s["text"] for s in out] == ["un", "deux"]


def test_merging_into_an_empty_list_keeps_everything():
    out = []
    hosted._merge(out, [seg(0.0, 1.0, "x"), seg(1.0, 2.0, "y")], offset=0.0)
    assert len(out) == 2


def test_word_times_stay_inside_their_segment_after_shifting():
    out = []
    hosted._merge(out, [seg(0.0, 3.0, "a", "b", "c")], offset=42.0)
    s = out[0]
    assert all(s["start"] <= w["start"] <= s["end"] + 1 for w in s["words"])
