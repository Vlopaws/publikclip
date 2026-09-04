"""Clips that continue each other, and clips that merely follow.

Two clips cut back to back are one moment in two pieces; a viewer meeting
the second one first has no way to know. Saying "1/2" costs six characters.
Saying it about two unrelated clips an hour apart is worse than saying
nothing, so the only signal used is that the second starts where the first
ends.
"""

from __future__ import annotations

import pytest

from publikclip_pipeline.autopilot import parts
from publikclip_pipeline.candidates import windows
from publikclip_pipeline.cli import parse_focus

import numpy as np


# --- grouping ------------------------------------------------------------

def test_two_clips_that_touch_are_a_pair():
    """The case asked for: 30:20-31:05 and 31:05-31:50."""
    got = parts.group([(1820.0, 1865.0), (1865.0, 1910.0)])
    assert got == {0: (1, 2), 1: (2, 2)}


def test_a_lone_clip_is_not_labelled():
    """1/1 is a label that says nothing about anything."""
    assert parts.group([(100.0, 140.0)]) == {}


def test_clips_far_apart_are_not_parts():
    assert parts.group([(100.0, 140.0), (3000.0, 3040.0)]) == {}


def test_a_small_gap_from_snapping_still_counts():
    # Sentence snapping and transition nudging move an edge a second or two.
    assert parts.group([(100.0, 140.0), (141.5, 180.0)]) == {0: (1, 2), 1: (2, 2)}


def test_a_full_gap_does_not():
    assert parts.group([(100.0, 140.0), (145.0, 180.0)]) == {}


def test_three_in_a_row_are_numbered_out_of_three():
    got = parts.group([(0.0, 40.0), (40.0, 80.0), (80.0, 120.0)])
    assert got == {0: (1, 3), 1: (2, 3), 2: (3, 3)}


def test_order_of_the_input_does_not_matter():
    forwards = parts.group([(0.0, 40.0), (40.0, 80.0)])
    backwards = parts.group([(40.0, 80.0), (0.0, 40.0)])
    assert forwards == {0: (1, 2), 1: (2, 2)}
    assert backwards == {1: (1, 2), 0: (2, 2)}


def test_a_long_run_is_not_a_serial():
    """Eight consecutive clips is the whole video, not a series."""
    bounds = [(i * 40.0, i * 40.0 + 40.0) for i in range(8)]
    assert parts.group(bounds) == {}


def test_two_runs_are_numbered_separately():
    got = parts.group([
        (0.0, 40.0), (40.0, 80.0),        # one pair
        (900.0, 940.0), (940.0, 980.0),   # another
    ])
    assert got[0] == (1, 2) and got[1] == (2, 2)
    assert got[2] == (1, 2) and got[3] == (2, 2)


def test_overlapping_clips_are_not_parts():
    # A negative gap means they cover the same seconds, not consecutive ones.
    assert parts.group([(100.0, 150.0), (140.0, 190.0)]) == {}


def test_an_overlapping_neighbour_does_not_sever_a_pair():
    """The real failure. 1820-1865 and 1865-1910 touch to the second, but a
    third window at 1847-1890 overlaps both and sorts between them, so a
    file-order walk broke the chain and the pair came out unlabelled."""
    got = parts.group([(1820.0, 1865.0), (1847.2, 1890.1), (1865.0, 1910.0)])
    assert got == {0: (1, 2), 2: (2, 2)}, "the intruder is not part of anything"


def test_the_nearest_successor_wins():
    """Two clips both starting near the end of a third: only one continues it,
    and it is the one that actually touches."""
    got = parts.group([(0.0, 40.0), (40.0, 80.0), (42.0, 82.0)])
    assert got[0] == (1, 2) and got[1] == (2, 2)
    assert 2 not in got


def test_a_clip_continues_at_most_one_other():
    got = parts.group([(0.0, 40.0), (0.5, 40.5), (40.5, 80.0)])
    runs = [i for i in got]
    assert len(runs) == 2, "one pair, not two claims on the same successor"


# --- labelling -----------------------------------------------------------

def test_the_marker_is_appended():
    assert parts.label("Maxime se fait clash 😱", (1, 2)) == "Maxime se fait clash 😱 (1/2)"


def test_a_clip_with_no_part_is_untouched():
    assert parts.label("Un titre", None) == "Un titre"


def test_an_empty_title_stays_empty():
    assert parts.label("", (1, 2)) == ""
    assert parts.label(None, (1, 2)) is None


# --- exact cuts ----------------------------------------------------------

def test_a_dictated_cut_is_used_as_given():
    """No sentence snapping, no transition nudge, no length clamp: somebody
    watched the video and chose the bounds."""
    curve = np.full(4000, 0.4)
    segs = [{"start": float(i), "end": float(i) + 3.0, "words": []}
            for i in range(0, 4000, 3)]
    out = windows.extract(curve, {}, segs, 4000.0, exact=[(1820.0, 1865.0)])
    mine = [c for c in out if abs(c.start - 1820.0) < 0.01]
    assert mine, "the dictated cut did not survive"
    assert mine[0].end == pytest.approx(1865.0)
    assert mine[0].forced


def test_two_dictated_cuts_come_out_as_a_pair():
    curve = np.full(4000, 0.4)
    segs = [{"start": float(i), "end": float(i) + 3.0, "words": []}
            for i in range(0, 4000, 3)]
    out = windows.extract(
        curve, {}, segs, 4000.0, exact=[(1820.0, 1865.0), (1865.0, 1910.0)]
    )
    mine = sorted(
        [c for c in out if c.forced and 1800 <= c.start <= 1910],
        key=lambda c: c.start,
    )
    assert len(mine) == 2
    assert parts.group([(c.start, c.end) for c in mine]) == {0: (1, 2), 1: (2, 2)}


def test_a_dictated_cut_outside_the_video_is_dropped():
    curve = np.full(100, 0.4)
    segs = [{"start": 0.0, "end": 3.0, "words": []}]
    out = windows.extract(curve, {}, segs, 100.0, exact=[(500.0, 540.0)])
    assert not [c for c in out if c.start >= 100.0]


def test_the_cut_flag_reads_the_same_timecodes_as_focus():
    assert parse_focus(["30:20-31:05"]) == [[1820.0, 1865.0]]
    assert parse_focus(["31:05-31:50"]) == [[1865.0, 1910.0]]


# --- dictated is not the same as forced ---------------------------------

def test_a_dictated_cut_is_marked_as_such():
    """`--focus` says where to look and still competes on merit.
    `--cut` says what to produce and does not."""
    curve = np.full(4000, 0.4)
    segs = [{"start": float(i), "end": float(i) + 3.0, "words": []}
            for i in range(0, 4000, 3)]
    out = windows.extract(
        curve, {}, segs, 4000.0,
        focus=[(500, 700)], exact=[(1820.0, 1865.0)],
    )
    mine = next(c for c in out if abs(c.start - 1820.0) < 0.01)
    assert mine.dictated and mine.forced

    from_focus = [c for c in out if c.forced and not c.dictated]
    assert from_focus, "a focus should still produce forced-but-not-dictated windows"


def test_a_dictated_cut_outranks_everything_in_dedupe():
    """It was computed, ranked on merit and evicted at the cut — the worst
    of every option."""
    curve = np.full(4000, 0.4)
    curve[1830] = 0.99  # a curve peak overlapping the dictated bounds
    segs = [{"start": float(i), "end": float(i) + 3.0, "words": []}
            for i in range(0, 4000, 3)]
    out = windows.extract(curve, {}, segs, 4000.0, exact=[(1820.0, 1865.0)])
    survivors = [c for c in out if 1810 <= c.start <= 1830]
    assert any(c.dictated for c in survivors), "the dictated cut lost to a peak"


def test_the_flag_survives_serialisation():
    curve = np.full(200, 0.4)
    segs = [{"start": 0.0, "end": 3.0, "words": []}]
    out = windows.extract(curve, {}, segs, 200.0, exact=[(10.0, 60.0)])
    row = next(c for c in out if c.dictated).to_json()
    assert row["dictated"] is True
    assert row["forced"] is True
