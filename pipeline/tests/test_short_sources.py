"""Sources shorter than a target clip.

Twitch clips run 20 to 60 seconds, and they are the whole of this
operation's live input. A window is grown around a peak and its edges are
snapped to sentence boundaries — and a sentence end can sit past the end of
the audio, because Whisper pads its last segment. raw_end was clamped to
the media; nothing re-clamped after the snap.

Measured on a real 27.084-second clip: the window came out 0.0–30.0, ffmpeg
produced the 27.1 seconds that exist, and the render's duration check threw
the clip away for being 2.9 seconds short of a length no cut could ever
have reached.
"""

from __future__ import annotations

import numpy as np
import pytest

from publikclip_pipeline.candidates import windows


def _segments(duration, step=3.0, pad_last=0.0):
    """ASR segments across a clip, with the last one optionally running past
    the end of the audio the way a real transcriber's does."""
    segs = []
    t = 0.0
    while t < duration:
        end = min(t + step, duration)
        segs.append({"start": t, "end": end, "words": []})
        t += step
    if pad_last and segs:
        segs[-1]["end"] = duration + pad_last
    return segs


def _extract(duration, pad_last=0.0, peak_at=None):
    curve = np.full(max(1, int(np.ceil(duration))), 0.5)
    if peak_at is not None:
        curve[int(peak_at)] = 0.99
    return windows.extract(curve, {}, _segments(duration, pad_last=pad_last), duration)


def test_a_window_never_runs_past_the_source():
    """The real case: 27.084 seconds of media, a segment ending at 30."""
    out = _extract(27.084, pad_last=2.916)
    assert out, "the clip was dropped entirely"
    for c in out:
        assert c.end <= 27.084 + 0.01, f"window ends at {c.end}, past the media"


def test_the_short_clip_is_still_produced():
    """Refusing it leaves nothing at all, and a 27-second clip is postable."""
    out = _extract(27.084, pad_last=2.916)
    assert len(out) >= 1


def test_a_source_shorter_than_the_minimum_is_taken_whole():
    out = _extract(12.0)
    assert out, "a 12-second source produced no window"
    c = out[0]
    assert c.start == pytest.approx(0.0, abs=0.01)
    assert c.end <= 12.01


def test_a_long_source_is_unaffected():
    out = _extract(600.0, peak_at=300)
    assert out
    for c in out:
        assert c.end <= 600.01
        assert c.end - c.start >= windows.MIN_LEN - 0.01


def test_the_clamp_survives_the_scene_cut_step():
    """_clear_of_cuts runs last and pads edges away from transitions; it must
    not push an edge back past the end of the media."""
    curve = np.full(28, 0.5)
    segs = _segments(27.084, pad_last=2.916)
    cuts = np.array([26.5])
    out = windows.extract(curve, {}, segs, 27.084, scene_times=list(cuts))
    for c in out:
        assert c.end <= 27.084 + 0.01


# --- and the report that sent the search the wrong way -------------------

def test_verification_names_what_actually_failed(monkeypatch, tmp_path):
    """"failed verification (duration 27.1s, 1080x1920)" printed two numbers
    that were both fine and named neither the expected duration nor a
    missing stream."""
    from publikclip_pipeline.render import renderer

    class Proc:
        stdout = (
            '{"streams":[{"codec_type":"video","width":1080,"height":1920},'
            '{"codec_type":"audio"}],"format":{"duration":"27.1"}}'
        )

    monkeypatch.setattr(renderer.subprocess, "run", lambda *a, **k: Proc())
    monkeypatch.setattr(renderer.ffmpeg_bin, "ffprobe", lambda: "ffprobe")
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00")

    ok = renderer.verify_output(f, 27.0)
    assert ok["ok"] and ok["problems"] == []

    bad = renderer.verify_output(f, 30.0)
    assert not bad["ok"]
    assert "expected 30.0s" in bad["problems"][0]
    assert "27.1" in bad["problems"][0]


def test_a_missing_stream_is_named_as_such(monkeypatch, tmp_path):
    from publikclip_pipeline.render import renderer

    class Proc:
        stdout = ('{"streams":[{"codec_type":"video","width":1080,'
                  '"height":1920}],"format":{"duration":"27.0"}}')

    monkeypatch.setattr(renderer.subprocess, "run", lambda *a, **k: Proc())
    monkeypatch.setattr(renderer.ffmpeg_bin, "ffprobe", lambda: "ffprobe")
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00")
    out = renderer.verify_output(f, 27.0)
    assert out["problems"] == ["no audio stream"]
