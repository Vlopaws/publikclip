"""Ingest plumbing.

The ffmpeg lookup gets the attention here because getting it wrong is
invisible: yt-dlp fetches video and audio, cannot merge them, exits 0, and
the only symptom is a missing file.
"""

import os

import pytest

from publikclip_pipeline.ingest import ytdlp


def _silent(fraction, message):
    return None


def test_the_managed_ffmpeg_is_preferred(monkeypatch, tmp_path):
    managed = tmp_path / "ffmpeg.exe"
    managed.write_bytes(b"binary")
    monkeypatch.setattr(
        "publikclip_pipeline.render.ffmpeg_bin.ffmpeg", lambda: str(managed)
    )
    assert ytdlp._ffmpeg_location(_silent) == str(managed)


def test_a_bare_name_is_resolved_through_path(monkeypatch):
    """resolve() returns the bare string 'ffmpeg' when nothing is managed
    yet; that has to go through PATH rather than be handed to yt-dlp raw."""
    monkeypatch.setattr("publikclip_pipeline.render.ffmpeg_bin.ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(ytdlp.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert ytdlp._ffmpeg_location(_silent) == "/usr/bin/ffmpeg"


def test_a_missing_ffmpeg_triggers_the_managed_fetch(monkeypatch, tmp_path):
    """The bug this replaces: no system ffmpeg meant no merge, and the
    download silently produced nothing."""
    fetched = tmp_path / "ffmpeg.exe"
    calls = []

    def fake_ensure(progress=None):
        calls.append(1)
        fetched.write_bytes(b"binary")
        return True

    state = {"path": "ffmpeg"}
    monkeypatch.setattr(
        "publikclip_pipeline.render.ffmpeg_bin.ffmpeg", lambda: state["path"]
    )
    monkeypatch.setattr(ytdlp.shutil, "which", lambda name: None)

    def ensure_and_point(progress=None):
        fake_ensure(progress)
        state["path"] = str(fetched)
        return True

    monkeypatch.setattr(
        "publikclip_pipeline.render.ffmpeg_bin.ensure_capable", ensure_and_point
    )
    assert ytdlp._ffmpeg_location(_silent) == str(fetched)
    assert calls == [1]


def test_no_ffmpeg_anywhere_is_none_not_a_crash(monkeypatch):
    """A progressive-format download needs no merge, so None is a valid
    answer — it must not take the whole ingest down."""
    monkeypatch.setattr("publikclip_pipeline.render.ffmpeg_bin.ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(ytdlp.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        "publikclip_pipeline.render.ffmpeg_bin.ensure_capable", lambda progress=None: False
    )
    assert ytdlp._ffmpeg_location(_silent) is None


def test_the_fetch_is_announced(monkeypatch):
    """Downloading ~80 MB of ffmpeg mid-ingest should not look like a hang."""
    messages = []
    monkeypatch.setattr("publikclip_pipeline.render.ffmpeg_bin.ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(ytdlp.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        "publikclip_pipeline.render.ffmpeg_bin.ensure_capable", lambda progress=None: False
    )
    ytdlp._ffmpeg_location(lambda fraction, message: messages.append(message))
    assert any("ffmpeg" in m for m in messages)


@pytest.mark.parametrize(
    "stderr, expected",
    [
        ("ERROR: [youtube] abc: Private video", True),
        ("ERROR: Sign in to confirm your age", True),
        ("ERROR: Unable to download webpage: timed out", False),
    ],
)
def test_auth_errors_are_told_apart_from_transport_ones(stderr, expected):
    """They need opposite next steps: give cookies, versus just retry."""
    assert ytdlp.is_auth_error(stderr) is expected


# --- making ffmpeg findable by third-party code ---------------------------


def test_ensure_on_path_makes_a_bare_ffmpeg_resolvable(monkeypatch, tmp_path):
    """whisperx shells out to a bare `ffmpeg`; resolving it only for our own
    calls left transcription dying on WinError 2 with nothing to go on."""
    import shutil as real_shutil

    from publikclip_pipeline.render import ffmpeg_bin

    managed = tmp_path / "bin" / "ffmpeg.exe"
    managed.parent.mkdir(parents=True)
    managed.write_bytes(b"binary")
    monkeypatch.setattr(ffmpeg_bin, "ffmpeg", lambda: str(managed))
    monkeypatch.setenv("PATH", "")

    added = ffmpeg_bin.ensure_on_path()
    assert added == str(managed.parent)
    assert real_shutil.which("ffmpeg") is not None


def test_ensure_on_path_does_not_duplicate_entries(monkeypatch, tmp_path):
    from publikclip_pipeline.render import ffmpeg_bin

    managed = tmp_path / "bin" / "ffmpeg.exe"
    managed.parent.mkdir(parents=True)
    managed.write_bytes(b"binary")
    monkeypatch.setattr(ffmpeg_bin, "ffmpeg", lambda: str(managed))
    monkeypatch.setenv("PATH", "")

    ffmpeg_bin.ensure_on_path()
    first = os.environ["PATH"]
    ffmpeg_bin.ensure_on_path()
    assert os.environ["PATH"] == first


def test_ensure_on_path_reports_failure_rather_than_lying(monkeypatch):
    from publikclip_pipeline.render import ffmpeg_bin

    monkeypatch.setattr(ffmpeg_bin, "ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(ffmpeg_bin.shutil, "which", lambda name: None)
    monkeypatch.setattr(ffmpeg_bin, "ensure_capable", lambda progress=None: False)
    assert ffmpeg_bin.ensure_on_path() is None
