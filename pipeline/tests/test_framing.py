"""Framing decision, title placement, and the filtergraph that realises them.

The bug these guard against is not a crash — every path here produces a
valid video. It is a video that shows the wrong thing: a 9:16 crop of a
screen share, or a headline printed across somebody's face.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from publikclip_pipeline.camera import framing
from publikclip_pipeline.captions import ass as ass_mod
from publikclip_pipeline.render import renderer


@dataclass
class FakeTrack:
    start: int
    heights: list
    tops: list


@dataclass
class FakeAnalysis:
    frame_count: int
    tracks: list


def analysis(n=100, height=0.2, top=0.3, covered=None, start=0):
    covered = n if covered is None else covered
    return FakeAnalysis(n, [FakeTrack(start, [height] * covered, [top] * covered)])


# --- the decision --------------------------------------------------------

def test_a_framed_talking_head_stays_vertical():
    shape = framing.decide(analysis(height=0.25))
    assert shape.mode == "vertical"
    assert "25%" in shape.reason


def test_faces_too_small_to_carry_a_crop_go_wide():
    # A three-shot around a table: faces are present the whole time but each
    # head is a twentieth of the frame. This is the Underscore_ case.
    shape = framing.decide(analysis(height=0.05))
    assert shape.mode == "wide"
    assert "too small" in shape.reason


def test_a_clip_with_no_faces_goes_wide():
    shape = framing.decide(FakeAnalysis(100, []))
    assert shape.mode == "wide"
    assert shape.face_coverage == 0.0


def test_faces_in_a_minority_of_frames_go_wide():
    shape = framing.decide(analysis(n=100, covered=20, height=0.3))
    assert shape.mode == "wide"
    assert shape.face_coverage == pytest.approx(0.2)


def test_the_operator_can_override_the_measurement():
    forced = framing.decide(analysis(height=0.25), forced="wide")
    assert forced.mode == "wide"
    assert "forced" in forced.reason
    # ...and the measurement is still reported, so the override is auditable.
    assert forced.face_height == pytest.approx(0.2, abs=0.06)


def test_the_threshold_is_a_boundary_not_a_cliff():
    assert framing.decide(analysis(height=framing.MIN_FACE_HEIGHT)).mode == "vertical"
    assert framing.decide(analysis(height=framing.MIN_FACE_HEIGHT - 0.001)).mode == "wide"


# --- geometry ------------------------------------------------------------

def test_wide_keeps_three_quarters_of_a_16_9_width():
    src_w, src_h = 1920, 1080
    crop_w = src_h * framing.crop_aspect("wide")
    assert crop_w == pytest.approx(1440)
    assert crop_w / src_w == pytest.approx(0.75)
    # ...where vertical keeps barely a third, which is the whole problem.
    assert (src_h * framing.crop_aspect("vertical")) / src_w < 0.35


def test_the_wide_picture_is_centred_with_equal_bars():
    x, y, w, h = framing.picture_box("wide")
    assert (x, w) == (0, framing.OUT_W)
    assert y == (framing.OUT_H - h) // 2
    assert h == 810  # 1080 wide at 4:3


def test_vertical_fills_the_canvas():
    assert framing.picture_box("vertical") == (0, 0, framing.OUT_W, framing.OUT_H)


# --- where a title may go ------------------------------------------------

def test_wide_puts_the_title_in_the_empty_bar():
    band = framing.title_band("wide", face_top=None)
    _, picture_y, _, _ = framing.picture_box("wide")
    assert band is not None
    assert band[1] <= picture_y  # never overlaps the picture


def test_vertical_keeps_the_title_above_the_highest_face():
    band = framing.title_band("vertical", face_top=0.4)
    assert band is not None
    assert band[1] < 0.4 * framing.OUT_H


def test_no_room_above_the_face_means_no_title():
    # A face framed near the top of the canvas leaves nothing to write in,
    # and a headline there would land on it.
    assert framing.title_band("vertical", face_top=0.05) is None


def test_a_face_outside_the_crop_is_not_avoided():
    # Faces the camera never frames must not push the title around.
    frames = [[0.0, 0.0, 608.0, 1080.0]] * 10
    off = FakeAnalysis(10, [FakeTrack(0, [0.2] * 10, [2.0] * 10)])
    assert framing.highest_face_in_output(off, frames, 1080) is None


def test_the_face_top_is_measured_through_the_crop():
    # Crop starting a quarter down the source: a face at source y=0.5 sits
    # halfway down a half-height crop, i.e. at output 0.5.
    frames = [[0.0, 270.0, 608.0, 540.0]] * 10
    a = FakeAnalysis(10, [FakeTrack(0, [0.1] * 10, [0.5] * 10)])
    assert framing.highest_face_in_output(a, frames, 1080) == pytest.approx(0.5)


# --- the title itself ----------------------------------------------------

def test_a_title_is_dropped_rather_than_drawn_over_a_face():
    doc = ass_mod.build_ass([], [], title="Un titre", title_band=None)
    assert "Dialogue: 2" not in doc


def test_a_title_that_cannot_shrink_enough_is_dropped():
    assert ass_mod.fit_title("mot " * 60, band_px=160) is None


def test_the_title_style_lands_in_the_styles_block():
    doc = ass_mod.build_ass(
        [], [], title="Microsoft equipe nos ecoles", title_band=(40, 515)
    )
    assert doc.index("Style: Title") < doc.index("[Events]")
    assert "Dialogue: 2" in doc


def test_the_title_hangs_from_the_top_of_the_band():
    """Anchored to the band's top edge, not centred inside it.

    Centring reads fine when the band is a letterbox bar and badly when it
    is not: a streamer whose webcam sits low in frame leaves a band most of
    the picture tall, and the title lands in the middle of the shot. Seen on
    the first real VM render. The band says how far down the title may
    start; it does not say where to put it.
    """
    doc = ass_mod.build_ass([], [], title="Court", title_band=(100, 500))
    assert "\\an8" in doc, "the block must grow downward from its anchor"
    assert "\\pos(540,100)" in doc
    assert "\\pos(540,300)" not in doc, "still centring in the band"


def test_a_tall_band_does_not_push_the_title_into_the_picture():
    """The regression itself: a face low in frame leaves a huge band."""
    doc = ass_mod.build_ass([], [], title="Court", title_band=(40, 1016))
    assert "\\pos(540,40)" in doc


def test_a_wide_title_holds_for_the_whole_clip():
    held = ass_mod.build_ass(
        [], [], title="Court", title_band=(40, 515),
        clip_duration=30.0, hold_whole_clip=True,
    )
    assert "0:00:30.00" in held
    hook = ass_mod.build_ass(
        [], [], title="Court", title_band=(40, 515),
        clip_duration=30.0, hold_whole_clip=False,
    )
    assert f"0:00:0{int(ass_mod.TITLE_HOLD_SEC)}.00" in hook


def test_a_title_never_outlives_a_short_clip():
    doc = ass_mod.build_ass(
        [], [], title="Court", title_band=(40, 515),
        clip_duration=2.0, hold_whole_clip=True,
    )
    assert "0:00:02.00" in doc


# --- the filtergraph -----------------------------------------------------

def test_vertical_is_a_single_chain():
    graph = renderer._join(["sendcmd=x"] + renderer._compose("vertical", (608, 1080, 0, 0)))
    assert ";" not in graph
    assert f"scale={renderer.OUT_W}:{renderer.OUT_H}" in graph


def test_wide_splits_into_picture_and_blurred_fill():
    graph = renderer._join(["sendcmd=x"] + renderer._compose("wide", (1440, 1080, 240, 0)))
    assert graph.count(";") == 3
    assert "split=2[wide_fg][wide_bg]" in graph
    assert "gblur=" in graph
    # The picture is overlaid at the top of the centred box, not at 0.
    _, y, _, _ = framing.picture_box("wide")
    assert f"overlay=0:{y}" in graph


def test_both_branches_read_the_same_crop():
    # One crop@c, so sendcmd drives picture and fill together; a second
    # would silently desynchronise them.
    graph = renderer._join(renderer._compose("wide", (1440, 1080, 240, 0)))
    assert graph.count("crop@c") == 1


def test_subtitles_chain_onto_the_composed_frame():
    parts = ["sendcmd=x"] + renderer._compose("wide", (1440, 1080, 240, 0)) + ["subtitles=f"]
    assert renderer._join(parts).endswith("setsar=1,subtitles=f")
