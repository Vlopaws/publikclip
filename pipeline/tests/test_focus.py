"""Cutting a region the interest curve does not rate.

Measured on a 72-minute video: a ten-minute rap battle scored 0.254-0.300
where the rest of the video sat at 0.32-0.43, so it produced ZERO candidate
windows — not low-ranked ones, none at all. The curve reads audio dynamics,
laughter and replay density, and sustained rapping over a continuous beat
has little of any. It is measuring the wrong thing for that content, and no
threshold fixes it. The operator has to be able to say "cut here anyway".
"""

from __future__ import annotations

import numpy as np
import pytest

from publikclip_pipeline.candidates import windows
from publikclip_pipeline.cli import parse_focus


def a_curve(length=4000, quiet=(1500, 2160)):
    """A video that is lively everywhere except one stretch.

    Numbers taken from the real one: 0.32-0.43 across a 72-minute video,
    0.254-0.300 through the rap battle in the middle of it.
    """
    rng = np.random.default_rng(0)
    curve = 0.35 + rng.random(length) * 0.10
    lo, hi = max(0, quiet[0]), min(length, quiet[1])
    if hi > lo:
        curve[lo:hi] = 0.26 + rng.random(hi - lo) * 0.04
    return curve


def segments_for(length):
    return [
        {"start": float(i), "end": float(i) + 3.0, "words": []}
        for i in range(0, int(length), 3)
    ]


# --- reading the flag ----------------------------------------------------

def test_a_range_in_minutes_and_seconds():
    assert parse_focus(["25:00-36:00"]) == [[1500.0, 2160.0]]


def test_plain_seconds_work_too():
    assert parse_focus(["90-150"]) == [[90.0, 150.0]]


def test_hours_work():
    assert parse_focus(["1:02:03-1:03:00"]) == [[3723.0, 3780.0]]


def test_several_ranges_comma_separated():
    assert parse_focus(["24:30-36:00,41:00-42:00"]) == [
        [1470.0, 2160.0], [2460.0, 2520.0]
    ]


def test_the_flag_is_repeatable():
    assert parse_focus(["1:00-2:00", "3:00-4:00"]) == [[60.0, 120.0], [180.0, 240.0]]


def test_a_backwards_range_is_refused():
    with pytest.raises(SystemExit):
        parse_focus(["36:00-25:00"])


def test_a_range_without_a_dash_is_refused():
    with pytest.raises(SystemExit):
        parse_focus(["25:00"])


def test_nothing_asked_is_nothing_forced():
    assert parse_focus(None) == []
    assert parse_focus([]) == []


# --- forcing the region --------------------------------------------------

def test_the_quiet_region_yields_nothing_on_its_own():
    """The bug, stated as a test: ranking the whole video drops it."""
    curve = a_curve()
    peaks = windows.local_maxima(curve)
    assert not [p for p in peaks if 1500 <= p < 2160]


def test_focusing_the_quiet_region_produces_peaks_there():
    curve = a_curve()
    picked = windows.focus_peaks(curve, [(1500, 2160)])
    assert picked
    assert all(1500 <= p < 2160 for p in picked)


def test_peaks_inside_a_focus_are_ranked_against_that_region():
    """The best moments OF the rap, not the best moments of the video that
    happen to fall inside it."""
    curve = a_curve()
    curve[1600] = 0.31  # the high point of the quiet stretch
    picked = windows.focus_peaks(curve, [(1500, 2160)])
    assert picked[0] <= 1620 or 1600 in picked


def test_focused_peaks_keep_their_distance_from_each_other():
    curve = a_curve()
    picked = windows.focus_peaks(curve, [(1500, 2160)], min_distance_sec=20)
    assert all(b - a >= 20 for a, b in zip(picked, picked[1:]))


def test_a_span_shorter_than_a_clip_is_ignored():
    curve = a_curve()
    assert windows.focus_peaks(curve, [(1500, 1505)]) == []


def test_a_span_off_the_end_does_not_raise():
    curve = a_curve(length=1000, quiet=(400, 600))
    assert windows.focus_peaks(curve, [(900, 99999)]) is not None


def test_a_span_entirely_past_the_end_yields_nothing():
    curve = a_curve(length=1000, quiet=(400, 600))
    assert windows.focus_peaks(curve, [(5000, 6000)]) == []


def test_several_spans_are_all_covered():
    curve = a_curve()
    picked = windows.focus_peaks(curve, [(1500, 2160), (3000, 3600)])
    assert any(1500 <= p < 2160 for p in picked)
    assert any(3000 <= p < 3600 for p in picked)


def test_extract_without_focus_is_unchanged():
    curve, segs = a_curve(), segments_for(4000)
    plain = windows.extract(curve, {}, segs, 4000.0)
    same = windows.extract(curve, {}, segs, 4000.0, focus=[])
    assert [(c.start, c.end) for c in plain] == [(c.start, c.end) for c in same]


def test_extract_with_focus_reaches_into_the_quiet_stretch():
    """End to end, at the length where the ranking cutoff actually bites.

    On a short curve every peak survives and the bug does not reproduce —
    which is why this uses a video-length one.
    """
    curve, segs = a_curve(), segments_for(4000)
    in_quiet = lambda cs: [c for c in cs if 1500 <= c.peak_time < 2160]

    assert not in_quiet(windows.extract(curve, {}, segs, 4000.0))
    assert in_quiet(windows.extract(curve, {}, segs, 4000.0, focus=[(1500, 2160)]))
