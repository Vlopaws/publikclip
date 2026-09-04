"""Which shape to cut a clip into, and where a title can sit without
landing on somebody's face.

A 9:16 crop of a 16:9 source keeps 34% of the width. That is the right
trade when the subject IS a person: their face carries the frame and the
discarded thirds are background. It is the wrong trade the moment the
subject is something on screen — a demo, a slide, a shared window. Then the
crop throws away the very thing being discussed, and the viewer watches a
presenter react to something they cannot see.

Underscore_ is the case that forced this: three people around a table with
a screen between them, most of the conversation pointing at it. Every clip
came out technically correct and substantively empty.

So the decision is made per clip from what the ASD pass already measured —
no new model, no second decode:

  vertical  faces are tracked through the clip AND big enough to carry a
            frame on their own  ->  9:16 crop, as before
  wide      anything else  ->  4:3 crop, letterboxed into the 9:16 canvas

4:3 rather than the full 16:9 because a full-width letterbox leaves a
picture 608 px tall in a 1920 canvas — a stripe. 4:3 keeps three quarters
of the width (enough that a centred screen survives) at 810 px tall, and
the bars it leaves are where the title goes.
"""

from __future__ import annotations

from dataclasses import dataclass

# Fraction of analysis frames that must carry a tracked face before the
# clip is treated as being about a person.
MIN_FACE_COVERAGE = 0.5

# ...and how tall that face must be, as a fraction of frame height.
#
# Set from the two populations this has actually seen, which separate
# cleanly:
#
#   webcam inset over gameplay     0.12
#   people talking to a camera     0.20 - 0.44   (17 clips, two creators)
#
# The first value is the one that matters. A streamer's webcam is a small
# fixed box in a corner, and treating it as the subject crops to the box
# and throws away the game — which is the entire content. The first
# threshold here was 0.10, just under that box, so a Twitch clip came out
# framed on a face while the action happened off-screen.
#
# 0.16 sits in the gap, nearer the gaming end because that is the error
# worth avoiding: a wide cut of a talking head is merely less tight, while
# a tight cut of a gameplay clip shows none of what happened.
MIN_FACE_HEIGHT = 0.16

# ...and how many people can be on screen before following any one of them
# stops making sense.
#
# A 9:16 crop shows one person. With two, the camera picks whoever is
# speaking and cuts between them, which reads fine. With three or more it
# swings — and active-speaker detection is least reliable exactly then,
# because several mouths are moving and the crop lands on whoever the model
# guessed. Reported from a real clip: "la reconnaissance de visage déconne de
# fou quand y a beaucoup de personnes à l'écran".
#
# A wide cut of a crowded shot loses nothing: everyone is in it, nothing
# moves, and nobody is framed on the wrong face. It is the one case where
# doing less is strictly better than doing the clever thing badly.
MAX_FACES_FOR_VERTICAL = 3

# Output canvas. Matches renderer.OUT_W/OUT_H and the ASS PlayRes; kept
# here so the band arithmetic is readable in one place.
OUT_W = 1080
OUT_H = 1920

WIDE_ASPECT = 4 / 3

# A title must not touch the picture in wide mode, nor a face in vertical
# mode. Keep it off the very edge either way.
TITLE_EDGE_MARGIN = 40
# Below this a band is too thin to hold two lines of a headline, and the
# title is better dropped than crushed.
MIN_TITLE_BAND = 150


@dataclass(frozen=True)
class Framing:
    """What shape a clip is cut into, and why."""

    mode: str                 # "vertical" | "wide"
    face_coverage: float      # fraction of frames with a tracked face
    face_height: float        # median height of the largest face, 0..1
    crowd: float              # median number of faces on screen at once
    reason: str               # human-readable, recorded in provenance

    @property
    def is_wide(self) -> bool:
        return self.mode == "wide"


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def crowding(analysis) -> float:
    """Median number of faces on screen at the same time.

    Counted per frame across every track, not as "how many people appear at
    some point": three people who take turns alone are three sequential
    close-ups and crop beautifully, while three people in shot together
    cannot be followed by one 9:16 window at all.
    """
    frames = getattr(analysis, "frame_count", 0) or 0
    tracks = getattr(analysis, "tracks", None) or []
    if not frames or not tracks:
        return 0.0

    per_frame = [0] * frames
    for track in tracks:
        heights = getattr(track, "heights", None) or []
        for i in range(len(heights)):
            f = track.start + i
            if 0 <= f < frames:
                per_frame[f] += 1

    present = [n for n in per_frame if n]
    return _median(present) if present else 0.0


def measure(analysis) -> tuple[float, float]:
    """(face coverage, median height of the largest face) from an AsdAnalysis.

    Height rather than area because area confounds a wide face with a tall
    one, and it is height that decides whether a head survives a crop.
    """
    frames = getattr(analysis, "frame_count", 0) or 0
    tracks = getattr(analysis, "tracks", None) or []
    if not frames or not tracks:
        return 0.0, 0.0

    # Largest face present in each frame, or nothing if the frame has none.
    per_frame: dict[int, float] = {}
    for track in tracks:
        heights = getattr(track, "heights", None)
        if not heights:
            continue
        for i, h in enumerate(heights):
            f = track.start + i
            if 0 <= f < frames and h > per_frame.get(f, 0.0):
                per_frame[f] = h

    coverage = len(per_frame) / frames
    return coverage, _median(list(per_frame.values()))


def decide(analysis, forced: str | None = None) -> Framing:
    """Pick the shape for one clip. `forced` overrides the measurement."""
    coverage, height = measure(analysis)
    crowd = crowding(analysis)
    if forced in ("vertical", "wide"):
        return Framing(forced, coverage, height, crowd, f"forced by settings ({forced})")

    if coverage < MIN_FACE_COVERAGE:
        return Framing(
            "wide", coverage, height, crowd,
            f"faces tracked in only {coverage:.0%} of frames — the subject is "
            "not a person",
        )
    if crowd >= MAX_FACES_FOR_VERTICAL:
        return Framing(
            "wide", coverage, height, crowd,
            f"{crowd:.0f} faces on screen at once — no single crop can follow "
            "them, and speaker detection is least reliable here",
        )
    if height < MIN_FACE_HEIGHT:
        return Framing(
            "wide", coverage, height, crowd,
            f"largest face is {height:.0%} of frame height — too small to "
            "carry a vertical crop",
        )
    return Framing(
        "vertical", coverage, height, crowd,
        f"faces in {coverage:.0%} of frames at {height:.0%} of frame height",
    )


def crop_aspect(mode: str) -> float:
    """Width / height of the crop window this mode takes from the source."""
    return WIDE_ASPECT if mode == "wide" else 9 / 16


def picture_box(mode: str) -> tuple[int, int, int, int]:
    """Where the cropped picture lands in the output canvas: (x, y, w, h).

    Vertical fills the canvas. Wide is scaled to full width and centred,
    leaving equal bars above and below.
    """
    if mode != "wide":
        return 0, 0, OUT_W, OUT_H
    h = int(round(OUT_W / WIDE_ASPECT))
    h -= h % 2
    return 0, (OUT_H - h) // 2, OUT_W, h


def highest_face_in_output(
    analysis, frames: list, src_h: int, until_frame: int | None = None
) -> float | None:
    """Top edge of the highest face ever framed, as a fraction of output
    height — or None if no face is framed at all.

    A face's position on screen is not its position in the source: the crop
    moves and zooms under it. So each face top is mapped through the crop
    rect of its own frame. The minimum is taken rather than a percentile
    because the number exists to keep a title off a face, and a title drawn
    over someone's eyes for half a second is still a title drawn over
    someone's eyes.

    `until_frame` bounds it to the frames the title is actually visible
    for. Measured over a whole clip the rule rejected 7 vertical clips out
    of 7 — one moment anywhere in forty seconds where somebody leans up
    killed the headline for the entire clip, including the thirty-six
    seconds after it had already gone. A minimum over frames nobody is
    looking at is not caution, it is a wrong question.
    """
    tracks = getattr(analysis, "tracks", None) or []
    if not tracks or not frames:
        return None

    limit = len(frames) if until_frame is None else min(until_frame, len(frames))
    lowest: float | None = None
    for track in tracks:
        tops = getattr(track, "tops", None) or []
        for i, top in enumerate(tops):
            f = track.start + i
            if not (0 <= f < limit):
                continue
            _, crop_y, _, crop_h = frames[f]
            if crop_h <= 0:
                continue
            out = (top * src_h - crop_y) / crop_h
            if out < 0.0 or out > 1.0:
                continue  # this face is outside the crop; it is not on screen
            if lowest is None or out < lowest:
                lowest = out
    return lowest


def title_band(mode: str, face_top: float | None) -> tuple[int, int] | None:
    """The vertical span, in output pixels, where a title may be drawn.

    Wide mode has a real empty bar, so the title goes there and touches
    nothing. Vertical mode has no empty region at all — the title has to
    share the frame, so it takes the space above the highest face the
    camera ever frames. When that space is too thin, this returns None and
    the caller draws no title rather than a title across someone's eyes.
    """
    if mode == "wide":
        _, y, _, _ = picture_box(mode)
        top, bottom = TITLE_EDGE_MARGIN, y - TITLE_EDGE_MARGIN
        return (top, bottom) if bottom - top >= MIN_TITLE_BAND else None

    if face_top is None:
        # No face was ever framed; nothing to avoid, so use the top eighth.
        return TITLE_EDGE_MARGIN, OUT_H // 8
    ceiling = int(face_top * OUT_H) - TITLE_EDGE_MARGIN
    top = TITLE_EDGE_MARGIN
    return (top, ceiling) if ceiling - top >= MIN_TITLE_BAND else None
