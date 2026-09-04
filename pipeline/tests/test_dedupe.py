"""Two clips that are mostly the same clip.

Overlap was measured as IoU, which divides by the union and so forgives a
window that repeats most of itself as long as it also reaches somewhere new.
A viewer does not compute a union: they recognise footage they have already
watched. These tests use the bounds from the run where that went wrong.
"""

from __future__ import annotations

from publikclip_pipeline.candidates import windows


def _c(start, end, score=0.5, **kw):
    return windows.Candidate(
        start=start, end=end, peak_time=(start + end) / 2,
        curve_score=score, **kw,
    )


def test_half_a_clip_already_seen_is_dropped():
    """Measured: 1847-1890 shared 25 of its 43 seconds with 1865-1910.
    IoU called that 0.40, under the gate, so both were rendered."""
    kept = windows.dedupe([_c(1865.0, 1910.0, 0.9), _c(1847.2, 1890.1, 0.8)])
    assert [c.start for c in kept] == [1865.0]


def test_the_higher_scored_of_the_pair_is_the_one_kept():
    kept = windows.dedupe([_c(1865.0, 1910.0, 0.3), _c(1847.2, 1890.1, 0.9)])
    assert [c.start for c in kept] == [1847.2]


def test_a_clip_that_merely_touches_survives():
    """The dictated pair: back to back, sharing nothing."""
    kept = windows.dedupe([_c(1820.0, 1865.0, 0.9), _c(1865.0, 1910.0, 0.8)])
    assert len(kept) == 2


def test_a_third_of_a_clip_is_still_its_own_clip():
    """13.9s of a 42s window is a shared run-up, not the same moment."""
    kept = windows.dedupe([_c(1820.0, 1865.0, 0.9), _c(1791.7, 1833.9, 0.8)])
    assert len(kept) == 2


def test_a_short_clip_inside_a_long_one_is_a_duplicate():
    """IoU is at its most forgiving when the lengths differ, which is exactly
    when containment matters most."""
    kept = windows.dedupe([_c(100.0, 200.0, 0.9), _c(150.0, 175.0, 0.8)])
    assert [c.start for c in kept] == [100.0]


def test_two_dictated_cuts_are_both_kept_however_they_overlap():
    """Somebody named both sets of bounds; repetition between them is their
    call, and dropping one would be the pipeline overruling an instruction."""
    kept = windows.dedupe([
        _c(1820.0, 1900.0, 0.2, forced=True, dictated=True),
        _c(1830.0, 1890.0, 0.1, forced=True, dictated=True),
    ])
    assert len(kept) == 2


def test_a_dictated_cut_still_evicts_a_curve_pick_that_repeats_it():
    kept = windows.dedupe([
        _c(1820.0, 1865.0, 0.1, forced=True, dictated=True),
        _c(1825.0, 1870.0, 0.9),
    ])
    assert [c.start for c in kept] == [1820.0]
