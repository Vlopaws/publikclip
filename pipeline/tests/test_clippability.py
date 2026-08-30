"""The creator probe's verdict logic.

Sampling and detection need network and a model, so they are exercised by
running the command. What is worth pinning down here is the arithmetic that
turns samples into a recommendation — including the stride bug that would
have halved every measurement.
"""

from __future__ import annotations

import pytest

from publikclip_pipeline.sources import clippability as cp


def sample(coverage, mode="vertical", frames=1500, error=None):
    s = cp.VideoSample(title="t", url="u")
    s.frames = frames
    s.face_coverage = coverage
    s.face_height = 0.25
    s.mode = mode
    s.error = error
    return s


def report(*samples):
    return cp.Clippability(creator="@x", samples=list(samples))


def test_faces_everywhere_reads_as_face_driven():
    r = report(sample(0.95), sample(0.9))
    assert r.verdict == "face-driven"
    assert "vertical clips will carry" in r.advice


def test_a_screen_show_reads_as_screen_driven():
    r = report(sample(0.1, "wide"), sample(0.2, "wide"))
    assert r.verdict == "screen-driven"
    assert "letterboxed" in r.advice


def test_half_and_half_reads_as_mixed():
    r = report(sample(0.5, "wide"), sample(0.5, "vertical"))
    assert r.verdict == "mixed"


def test_high_coverage_that_still_cuts_wide_is_not_face_driven():
    # Faces present but small: exactly the case where presence alone would
    # give a falsely reassuring answer.
    r = report(sample(0.9, "wide"), sample(0.95, "wide"))
    assert r.verdict == "mixed"
    assert r.vertical_share == 0.0


def test_a_failed_sample_is_excluded_not_counted_as_zero():
    r = report(sample(0.9), sample(0.0, error="HTTP 403"))
    assert len(r.measured) == 1
    assert r.face_coverage == pytest.approx(0.9)


def test_nothing_measurable_says_so_rather_than_guessing():
    r = report(sample(0.0, error="boom"))
    assert r.verdict == "unknown"
    assert r.face_coverage == 0.0
    assert "nothing could be sampled" in r.advice


def test_no_samples_at_all_is_unknown():
    assert report().verdict == "unknown"


def test_strided_frames_are_not_counted_as_faceless():
    # detection_pass records None for frames it never looked at. Feeding
    # those to the analysis would report half the real coverage.
    looked_at = [[_box(0.3, 0.6)] for _ in range(10)]
    analysis = cp._SampledAnalysis(looked_at)
    coverage, height = _measure(analysis)
    assert coverage == 1.0
    assert height == pytest.approx(0.3)


def test_the_tallest_face_is_the_one_measured():
    frame = [_box(0.1, 0.2), _box(0.4, 0.8)]
    analysis = cp._SampledAnalysis([frame])
    _, height = _measure(analysis)
    assert height == pytest.approx(0.4)


def test_frames_without_a_face_lower_the_coverage():
    analysis = cp._SampledAnalysis([[_box(0.3, 0.6)], [], [_box(0.3, 0.6)], []])
    coverage, _ = _measure(analysis)
    assert coverage == pytest.approx(0.5)


# --- helpers -------------------------------------------------------------

def _box(height, top):
    from publikclip_pipeline.camera.detect import FaceBox

    return FaceBox(x1=0.4, y1=top, x2=0.6, y2=top + height, score=0.9)


def _measure(analysis):
    from publikclip_pipeline.camera import framing

    return framing.measure(analysis)
