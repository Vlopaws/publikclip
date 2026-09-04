"""Candidate window extraction: interest-curve maxima → 15–75 s windows →
sentence-boundary snapping → IOU dedupe → ~35 candidates.

Boundary snapping follows clip-forge's sentences.ts snap() arithmetic (MIT):
expand/contract each edge to the nearest sentence boundary within a snap
radius, preferring boundaries that follow a pause. The IOU span dedupe is
autoclip's highlights.py pattern (MIT).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_LEN = 15.0
MAX_LEN = 75.0
TARGET_LEN = 42.0
SNAP_RADIUS = 6.0
DEDUPE_IOU = 0.55

# A clip must not open on the source's own editing. Starting a second
# before a cut means the viewer sees a moment of the outgoing shot, then a
# wipe, then the real content — which reads as a broken upload rather than
# as a clip. Scene changes were already detected and fed to the interest
# curve as a channel; they were never used to place an edge.
#
# 1.2 s: long enough to cover a wipe or a short fade, short enough that a
# genuine hard cut a second before the action does not drag the start.
SCENE_PAD = 1.2
MAX_CANDIDATES = 35


@dataclass
class Candidate:
    start: float
    end: float
    peak_time: float
    curve_score: float
    channel_scores: dict[str, float] = field(default_factory=dict)
    # True when this window exists because the operator named its region,
    # not because the curve found a peak there.
    forced: bool = False

    def to_json(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "forced": self.forced,
            "peak_time": round(self.peak_time, 3),
            "curve_score": round(self.curve_score, 4),
            "channel_scores": self.channel_scores,
        }


def sentence_boundaries(segments: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(starts, ends) of ASR sentences — the only legal cut points."""
    starts = np.array([float(s["start"]) for s in segments])
    ends = np.array([float(s["end"]) for s in segments])
    return starts, ends


def _snap(t: float, boundaries: np.ndarray, radius: float = SNAP_RADIUS) -> float | None:
    if len(boundaries) == 0:
        return None
    idx = int(np.argmin(np.abs(boundaries - t)))
    if abs(boundaries[idx] - t) <= radius:
        return float(boundaries[idx])
    return None


def local_maxima(curve: np.ndarray, min_distance_sec: int = 20) -> list[int]:
    """Peak indices, greedily suppressing neighbors within min_distance."""
    order = np.argsort(curve)[::-1]
    picked: list[int] = []
    for idx in order:
        if curve[idx] <= 0:
            break
        if all(abs(idx - p) >= min_distance_sec for p in picked):
            picked.append(int(idx))
        if len(picked) >= MAX_CANDIDATES * 3:  # generous pool pre-dedupe
            break
    return sorted(picked)


def _clear_of_cuts(
    start: float, end: float, cuts: np.ndarray
) -> tuple[float, float]:
    """Move the edges off the source's own transitions.

    A cut just after the start means the clip opens on the tail of the
    previous shot; the start moves to the cut, so it opens on the new one
    instead. A cut just before the end means it closes into a transition;
    the end pulls back to the cut.
    """
    if len(cuts) == 0:
        return start, end

    opening = cuts[(cuts > start) & (cuts < start + SCENE_PAD)]
    if len(opening):
        start = float(opening[-1])

    closing = cuts[(cuts > end - SCENE_PAD) & (cuts < end)]
    if len(closing):
        end = float(closing[0])

    return start, end


def window_around(
    peak: int,
    curve: np.ndarray,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    duration: float,
    cuts: np.ndarray | None = None,
) -> tuple[float, float] | None:
    """Grow a window around the peak until curve mass drops off, then snap
    both edges to sentence boundaries and off any scene transition."""
    half = TARGET_LEN / 2
    raw_start = max(0.0, peak - half)
    raw_end = min(duration, peak + half)

    start = _snap(raw_start, seg_starts)
    end = _snap(raw_end, seg_ends)
    if start is None:
        start = raw_start
    if end is None:
        end = raw_end
    # A clip must start where a sentence starts; drifting an end is tolerable,
    # a mid-word opening is not.
    if start >= end:
        return None
    length = end - start
    if length < MIN_LEN:
        # try extending the end to the next sentence end
        later = seg_ends[seg_ends > start + MIN_LEN]
        if len(later) == 0:
            return None
        end = float(later[0])
        length = end - start
    if length > MAX_LEN:
        earlier = seg_ends[(seg_ends > start + MIN_LEN) & (seg_ends <= start + MAX_LEN)]
        if len(earlier) == 0:
            return None
        end = float(earlier[-1])

    # Last, so it cannot be undone by the length adjustments above.
    if cuts is not None:
        start, end = _clear_of_cuts(start, end, cuts)
        if end - start < MIN_LEN:
            return None
    return (round(start, 3), round(end, 3))


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union


def dedupe(candidates: list[Candidate], iou: float = DEDUPE_IOU) -> list[Candidate]:
    """Keep the higher-scored of any overlapping pair (autoclip pattern).

    Forced candidates rank ahead of everything else, whatever the curve
    thinks of them. Sorting on curve_score alone put them last by
    construction — a region is forced precisely because the curve rates it
    low — so they were computed, then evicted at the MAX_CANDIDATES cut and
    the flag did nothing at all.
    """
    kept: list[Candidate] = []
    for cand in sorted(
        candidates, key=lambda c: (c.forced, c.curve_score), reverse=True
    ):
        if all(_iou((cand.start, cand.end), (k.start, k.end)) < iou for k in kept):
            kept.append(cand)
        if len(kept) >= MAX_CANDIDATES:
            break
    return sorted(kept, key=lambda c: c.start)


def focus_peaks(
    curve: np.ndarray, spans: list[tuple[float, float]], min_distance_sec: int = 20
) -> list[int]:
    """Peaks inside operator-chosen regions, ranked against those regions.

    The curve ranks the whole video, and `local_maxima` takes the top of
    that ranking. A stretch that is genuinely good but quiet — measured on
    a 72-minute video, a ten-minute rap battle scored 0.254-0.300 where the
    rest of the video sat at 0.32-0.43 — never reaches the cut, so it
    yields nothing at all rather than something mediocre.

    The curve is not wrong about what it measures. It reads audio dynamics,
    laughter and replay density, and sustained rapping over a continuous
    beat has little of any: few silences, few bursts, music masking the
    peaks it looks for. It is measuring the wrong thing for that content,
    and no threshold fixes that.

    So the operator can say "cut here anyway", and inside those spans the
    ranking is local: the best moments OF THE RAP, not the best moments of
    the video that happen to be in the rap.
    """
    picked: list[int] = []
    for start, end in spans:
        lo, hi = max(0, int(start)), min(len(curve), int(end))
        if hi - lo < MIN_LEN:
            continue
        region = curve[lo:hi]
        for offset in np.argsort(region)[::-1]:
            idx = lo + int(offset)
            if all(abs(idx - p) >= min_distance_sec for p in picked):
                picked.append(idx)
            if len(picked) >= MAX_CANDIDATES:
                break
    return sorted(picked)


def extract(
    curve: np.ndarray,
    channels: dict[str, np.ndarray],
    segments: list[dict],
    duration: float,
    focus: list[tuple[float, float]] | None = None,
    scene_times: list[float] | None = None,
    exact: list[tuple[float, float]] | None = None,
) -> list[Candidate]:
    """`focus` biases where windows are looked for; `exact` states where they
    are. A focus range still lets the curve choose within it — an exact cut
    does not, because somebody watched the video and picked the bounds, and
    second-guessing that is the one thing they did not ask for.
    """
    seg_starts, seg_ends = sentence_boundaries(segments)
    cuts = np.asarray(sorted(scene_times or []), dtype=float)
    peaks = local_maxima(curve)
    forced_peaks: set[int] = set()
    if focus:
        forced_peaks = set(focus_peaks(curve, focus))
        peaks = sorted(forced_peaks | set(peaks))
    out: list[Candidate] = []

    # Stated bounds go in first and unmodified: no sentence snapping, no
    # transition nudging, no length clamp. They came from a person watching.
    for start, end in exact or []:
        start, end = max(0.0, float(start)), min(duration, float(end))
        if end <= start:
            continue
        a, b = int(start), max(int(start) + 1, int(np.ceil(end)))
        out.append(
            Candidate(
                start=round(start, 3),
                end=round(end, 3),
                peak_time=float((start + end) / 2),
                curve_score=float(np.mean(curve[a : min(b, len(curve))])),
                channel_scores={
                    name: round(float(np.mean(ch[a : min(b, len(ch))])), 4)
                    for name, ch in channels.items()
                    if len(ch) > a
                },
                forced=True,
            )
        )

    for peak in peaks:
        window = window_around(peak, curve, seg_starts, seg_ends, duration, cuts)
        if window is None:
            continue
        start, end = window
        a, b = int(start), max(int(start) + 1, int(np.ceil(end)))
        per_channel = {
            name: round(float(np.mean(ch[a : min(b, len(ch))])), 4)
            for name, ch in channels.items()
            if len(ch) > a
        }
        out.append(
            Candidate(
                start=start,
                end=end,
                peak_time=float(peak),
                curve_score=float(np.mean(curve[a : min(b, len(curve))])),
                channel_scores=per_channel,
                forced=peak in forced_peaks,
            )
        )
    return dedupe(out)
