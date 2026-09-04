"""Not opening a clip on the source's own transition.

Reported on a real clip: "le début commence sur une transition de la
vidéo". A clip that starts a second before a cut shows a moment of the
outgoing shot, then a wipe, then the content — which reads as a broken
upload. Scene changes were already detected and fed to the interest curve
as a channel; nothing used them to place an edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from publikclip_pipeline.candidates import windows


def cuts(*times):
    return np.asarray(sorted(times), dtype=float)


# --- the rule ------------------------------------------------------------

def test_a_start_just_before_a_cut_moves_onto_the_new_shot():
    start, end = windows._clear_of_cuts(100.0, 140.0, cuts(100.8))
    assert start == 100.8, "the clip still opens on the outgoing shot"
    assert end == 140.0


def test_an_end_just_after_a_cut_pulls_back_to_it():
    start, end = windows._clear_of_cuts(100.0, 140.0, cuts(139.5))
    assert end == 139.5
    assert start == 100.0


def test_a_cut_comfortably_inside_is_left_alone():
    """Cuts mid-clip are the video's own editing and are fine."""
    assert windows._clear_of_cuts(100.0, 140.0, cuts(120.0)) == (100.0, 140.0)


def test_a_cut_before_the_start_is_not_our_problem():
    assert windows._clear_of_cuts(100.0, 140.0, cuts(99.0)) == (100.0, 140.0)


def test_a_cut_exactly_at_the_start_needs_no_move():
    # Already opening on the new shot.
    assert windows._clear_of_cuts(100.0, 140.0, cuts(100.0)) == (100.0, 140.0)


def test_no_cuts_changes_nothing():
    assert windows._clear_of_cuts(10.0, 50.0, cuts()) == (10.0, 50.0)


def test_the_latest_opening_cut_wins():
    """Two transitions in the pad: land after both, not between them."""
    start, _ = windows._clear_of_cuts(100.0, 140.0, cuts(100.3, 100.9))
    assert start == 100.9


def test_the_earliest_closing_cut_wins():
    _, end = windows._clear_of_cuts(100.0, 140.0, cuts(139.1, 139.8))
    assert end == 139.1


# --- through the window placer -------------------------------------------

def a_clip_setup(length=600):
    curve = np.full(length, 0.4)
    curve[300] = 0.9
    segs = np.arange(0.0, length, 3.0)
    return curve, segs, segs + 3.0


def test_window_around_without_cuts_is_unchanged():
    curve, starts, ends = a_clip_setup()
    plain = windows.window_around(300, curve, starts, ends, 600.0)
    same = windows.window_around(300, curve, starts, ends, 600.0, cuts())
    assert plain == same


def test_window_around_moves_off_a_transition():
    curve, starts, ends = a_clip_setup()
    plain = windows.window_around(300, curve, starts, ends, 600.0)
    assert plain is not None
    # Put a cut just inside the start it chose.
    moved = windows.window_around(
        300, curve, starts, ends, 600.0, cuts(plain[0] + 0.5)
    )
    assert moved is not None
    assert moved[0] > plain[0]


def test_a_window_squeezed_below_the_minimum_is_dropped():
    """Better no clip than a three-second one."""
    curve = np.full(60, 0.4)
    curve[30] = 0.9
    segs = np.arange(0.0, 60.0, 3.0)
    tight = cuts(*[t for t in np.arange(10.0, 50.0, 0.4)])
    out = windows.window_around(30, curve, segs, segs + 3.0, 60.0, tight)
    assert out is None or out[1] - out[0] >= windows.MIN_LEN


def test_extract_passes_cuts_through():
    curve = np.full(4000, 0.35)
    curve[1000] = 0.95
    segs = [
        {"start": float(i), "end": float(i) + 3.0, "words": []}
        for i in range(0, 4000, 3)
    ]
    without = windows.extract(curve, {}, segs, 4000.0)
    peak = next(c for c in without if abs(c.peak_time - 1000) < 30)
    withc = windows.extract(
        curve, {}, segs, 4000.0, scene_times=[peak.start + 0.6]
    )
    moved = next(c for c in withc if abs(c.peak_time - 1000) < 30)
    assert moved.start > peak.start
