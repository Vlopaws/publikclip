"""Does this creator point a camera at people, or at a screen?

Two creators can look identical to the saturation scan — same audience, same
niche, nobody clipping them — and produce completely different clips. What
separates them is not reach, it is what fills the frame. A conversation
between faces cuts to 9:16 beautifully. A screen-driven show does not: the
crop throws away the subject, and even framed on a speaker, the clip is
about something the viewer cannot see.

That was learned the expensive way, an hour of pipeline at a time. So this
measures it up front, using the detector the camera stage already carries:
pull a minute from the middle of a couple of recent uploads, count faces,
and report what the framing decision would be.

It is a sample, not a census — a creator who does interviews and one
screen-share episode will read as whatever the sampled minute contained,
which is why more than one video is sampled and each is reported
separately. Needs no API key. Downloads a few MB per video, at 480p, and
deletes it afterwards.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..camera import framing as framing_mod
from ..ingest import ytdlp
from . import youtube as youtube_source

# Where in a video to sample from, as a fraction of its runtime. The opening
# is titles and sponsor reads; the end is outros. The middle is the show.
SAMPLE_AT = 0.5
SAMPLE_SECONDS = 60.0

# Above this share of sampled frames carrying a usable face, the material is
# face-driven enough that vertical clips will mostly work.
FACE_DRIVEN = 0.7


@dataclass
class VideoSample:
    title: str
    url: str
    frames: int = 0
    face_coverage: float = 0.0
    face_height: float = 0.0
    mode: str = "wide"
    error: str | None = None

    def to_json(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "frames": self.frames,
            "face_coverage": round(self.face_coverage, 3),
            "face_height": round(self.face_height, 3),
            "would_frame": self.mode,
            "error": self.error,
        }


@dataclass
class Clippability:
    creator: str
    samples: list[VideoSample] = field(default_factory=list)

    @property
    def measured(self) -> list[VideoSample]:
        return [s for s in self.samples if s.error is None and s.frames]

    @property
    def face_coverage(self) -> float:
        ok = self.measured
        return sum(s.face_coverage for s in ok) / len(ok) if ok else 0.0

    @property
    def face_height(self) -> float:
        ok = self.measured
        return sum(s.face_height for s in ok) / len(ok) if ok else 0.0

    @property
    def vertical_share(self) -> float:
        ok = self.measured
        return sum(s.mode == "vertical" for s in ok) / len(ok) if ok else 0.0

    @property
    def verdict(self) -> str:
        if not self.measured:
            return "unknown"
        if self.face_coverage >= FACE_DRIVEN and self.vertical_share >= 0.5:
            return "face-driven"
        if self.face_coverage < 0.3:
            return "screen-driven"
        return "mixed"

    @property
    def advice(self) -> str:
        return {
            "face-driven": (
                "faces fill the frame — vertical clips will carry on their own"
            ),
            "mixed": (
                "faces some of the time — expect a mix of vertical and "
                "letterboxed clips, and check that the talk stands without "
                "the screen"
            ),
            "screen-driven": (
                "the camera is mostly on a screen — clips will be letterboxed "
                "and the talk may not stand without what is being shown"
            ),
            "unknown": "nothing could be sampled",
        }[self.verdict]

    def to_json(self) -> dict:
        return {
            "creator": self.creator,
            "verdict": self.verdict,
            "advice": self.advice,
            "face_coverage": round(self.face_coverage, 3),
            "face_height": round(self.face_height, 3),
            "vertical_share": round(self.vertical_share, 3),
            "samples": [s.to_json() for s in self.samples],
        }


class _SampledAnalysis:
    """The shape framing.measure expects, filled from a raw detection pass.

    The real AsdAnalysis carries per-track scores from the active-speaker
    model. None of that is needed to answer "is there a face here", and
    running it would multiply the cost of a probe by an order of magnitude,
    so this presents one synthetic track per frame instead.
    """

    def __init__(self, per_frame_boxes: list) -> None:
        self.frame_count = len(per_frame_boxes)
        self.tracks = []
        for i, boxes in enumerate(per_frame_boxes):
            if not boxes:
                continue
            tallest = max(boxes, key=lambda b: b.y2 - b.y1)
            self.tracks.append(
                _Track(start=i, heights=[tallest.y2 - tallest.y1], tops=[tallest.y1])
            )


@dataclass
class _Track:
    start: int
    heights: list
    tops: list


def _measure_file(path: Path) -> tuple[int, float, float, str]:
    from ..camera.asd import detection_pass
    from ..camera.detect import FaceDetector
    from ..models import registry, specs

    detector = FaceDetector(str(registry.ensure(specs.ULTRAFACE, lambda f, m: None)))
    faces, _cuts, frames = detection_pass(str(path), 0.0, SAMPLE_SECONDS, detector)
    # detection_pass strides: frames it skipped hold None, which is "not
    # looked at", not "no face". Counting those as faceless would halve
    # every measurement.
    looked_at = [f for f in faces if f is not None]
    analysis = _SampledAnalysis(looked_at)
    coverage, height = framing_mod.measure(analysis)
    return frames, coverage, height, framing_mod.decide(analysis).mode


def assess(
    channel: str,
    videos: int = 2,
    progress=None,
    binary: Path | None = None,
) -> Clippability:
    """Sample a channel's recent uploads and report how clippable they look."""
    emit = progress or (lambda fraction, message: None)
    uploads = youtube_source.recent_uploads(channel, limit=videos, progress=emit)
    out = Clippability(creator=channel)

    with tempfile.TemporaryDirectory(prefix="publikclip-probe-") as tmp:
        for i, item in enumerate(uploads[:videos]):
            sample = VideoSample(title=item.title, url=item.url)
            emit(i / max(1, videos), f"Sampling {item.title[:50]}…")
            dest = Path(tmp) / f"sample_{i:02d}.mp4"
            try:
                start = max(0.0, (item.duration_sec or 0.0) * SAMPLE_AT)
                ytdlp.sample_section(
                    item.url, dest, start, SAMPLE_SECONDS,
                    lambda f, m: emit(-1, m),
                )
                frames, coverage, height, mode = _measure_file(dest)
                sample.frames = frames
                sample.face_coverage = coverage
                sample.face_height = height
                sample.mode = mode
            except Exception as err:  # noqa: BLE001 — one bad video is not fatal
                sample.error = str(err)[:200]
            out.samples.append(sample)
    return out
