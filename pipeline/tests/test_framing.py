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


def test_a_webcam_over_gameplay_goes_wide():
    """The case that shipped a bad clip.

    A streamer's webcam is a small fixed box in a corner. Measured on a real
    Twitch clip it is 12% of frame height — the old threshold was 10%, so
    the camera framed the box and the game, which is the entire content,
    was cropped away. The face was there; it was not the subject.
    """
    shape = framing.decide(analysis(height=0.117))
    assert shape.mode == "wide"


def test_the_threshold_sits_in_the_gap_between_the_two_populations():
    """Set from measurement, so state the measurement.

    webcam inset over gameplay   0.12
    people talking to a camera   0.20 - 0.44  (17 clips, two creators)
    """
    webcam, smallest_talking_head = 0.12, 0.20
    assert webcam < framing.MIN_FACE_HEIGHT < smallest_talking_head


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


def test_the_title_does_not_inherit_the_caption_font():
    """Captions and titles do different jobs.

    A caption is read while the clip plays; a title has about a second to
    stop a thumb. Tying them together gave a "classic" run a headline in
    Inter, which reads as a caption that wandered upward.
    """
    doc = ass_mod.build_ass(
        [], [], preset_name="classic", title="Un titre", title_band=(40, 515)
    )
    style = next(l for l in doc.splitlines() if l.startswith("Style: Title"))
    assert ass_mod.TITLE_FONT in style
    assert "Inter" not in style


def test_the_title_is_upper_case_and_heavily_outlined():
    doc = ass_mod.build_ass([], [], title="un titre court", title_band=(40, 515))
    line = next(l for l in doc.splitlines() if l.startswith("Dialogue: 2"))
    assert "UN TITRE COURT" in line
    style = next(l for l in doc.splitlines() if l.startswith("Style: Title"))
    assert f",{ass_mod.TITLE_OUTLINE},{ass_mod.TITLE_SHADOW}," in style


def test_the_title_font_is_one_that_ships_with_the_project():
    """libass resolves by family name against fontsdir. A face that is not
    in there renders as a substitute, silently."""
    assert (ass_mod.FONTS_DIR / ass_mod.TITLE_FONT_FILE).exists()


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


# --- a crowded shot ------------------------------------------------------

def crowd_analysis(n_people, frames=100, height=0.25):
    """n people on screen at the same time, for the whole clip."""
    return FakeAnalysis(
        frames,
        [FakeTrack(0, [height] * frames, [0.3] * frames) for _ in range(n_people)],
    )


def test_three_people_at_once_go_wide():
    """Reported from a real clip: face tracking swings when several people
    are in shot, and speaker detection is least reliable exactly then."""
    shape = framing.decide(crowd_analysis(3))
    assert shape.mode == "wide"
    assert "at once" in shape.reason


def test_two_people_still_get_a_vertical_cut():
    """Two is a conversation: the camera cuts between them and it reads."""
    assert framing.decide(crowd_analysis(2)).mode == "vertical"


def test_one_person_is_unaffected():
    assert framing.decide(crowd_analysis(1)).mode == "vertical"


def test_people_taking_turns_alone_are_not_a_crowd():
    """Three sequential close-ups crop beautifully; three people in one shot
    do not. Counting who appears at some point would confuse them."""
    frames = 90
    tracks = [
        FakeTrack(0, [0.25] * 30, [0.3] * 30),
        FakeTrack(30, [0.25] * 30, [0.3] * 30),
        FakeTrack(60, [0.25] * 30, [0.3] * 30),
    ]
    analysis = FakeAnalysis(frames, tracks)
    assert framing.crowding(analysis) == 1
    assert framing.decide(analysis).mode == "vertical"


def test_crowding_is_measured_and_recorded():
    shape = framing.decide(crowd_analysis(4))
    assert shape.crowd == 4
    assert shape.mode == "wide"


def test_an_empty_analysis_has_no_crowd():
    assert framing.crowding(FakeAnalysis(50, [])) == 0.0


def test_a_crowded_wide_cut_is_still_a_tripod():
    """The whole point: wide already means centred, unzoomed and without
    punch-ins, so a crowded clip inherits a still camera."""
    from publikclip_pipeline.camera import director

    assert framing.decide(crowd_analysis(5)).mode == "wide"
    # director.build_trajectory parks the camera in wide mode; guard that
    # the contract it relies on has not moved.
    import inspect

    source = inspect.getsource(director.build_trajectory)
    assert "wide = mode == \"wide\"" in source
    assert "and not wide" in source


# --- a vertical clip that had nowhere to put its title -------------------
#
# The band was only ever looked for above the highest face. A 9:16 crop that
# follows a face puts it high in frame by construction, so the rule rejected
# five vertical clips out of seven and they shipped with no headline at all:
# a rule meant to keep a title off a face was deciding there would be no
# title. The numbers below are the ones those clips measured.

def test_no_headroom_falls_back_under_the_chin():
    """Clip 8: face from 7.7% to 50.7% of the frame."""
    band = framing.title_band("vertical", 0.077, 0.507)
    assert band is not None
    assert band[0] > 0.507 * framing.OUT_H, "the title would sit on the face"
    assert band[1] <= framing.OUT_H


def test_headroom_is_still_preferred_when_it_exists():
    """Clip 1: 15.7% down leaves room above, and a title belongs at the top."""
    assert framing.title_band("vertical", 0.157, 0.666) == (40, 261)


def test_a_face_filling_the_frame_still_gets_no_title():
    """Clip 7: 2.5% to 99.3%. There is no free band, and inventing one puts
    a headline across someone's face."""
    assert framing.title_band("vertical", 0.025, 0.993) is None


def test_a_chin_low_in_frame_leaves_too_little():
    """Clip 16: 92.4% down — 106 pixels, under the minimum."""
    assert framing.title_band("vertical", 0.064, 0.924) is None


def test_the_band_under_the_chin_clears_the_minimum():
    """Clip 2: 87.0% down leaves 170 pixels, which is a band."""
    band = framing.title_band("vertical", 0.101, 0.870)
    assert band is not None
    assert band[1] - band[0] >= framing.MIN_TITLE_BAND


def test_without_a_chin_measurement_nothing_changes():
    """Older trajectories carry no face_bottom; they must not crash or
    suddenly grow a band."""
    assert framing.title_band("vertical", 0.077) is None


def test_wide_mode_ignores_the_chin_entirely():
    """A wide clip's band is the letterbox bar; faces are in the picture."""
    assert framing.title_band("wide", 0.05, 0.99) == framing.title_band("wide", None)
