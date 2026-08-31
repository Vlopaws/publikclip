"""Rejecting an ffmpeg that reports the filters and then cannot run them.

Ubuntu 24.04 ships ffmpeg 6.1.1. It lists `subtitles` and `sendcmd` in
`-filters`, passes every capability check this project had, and then takes
over thirty minutes on a twenty-six second clip that a current build
finishes in seventeen seconds. The autopilot on the deployment VM died on a
1800-second render timeout with nothing in the logs to say why.

Capability and usability are different questions, and only the second one
needs a clock.
"""

from __future__ import annotations

import platform
import subprocess

import pytest

from publikclip_pipeline.render import ffmpeg_bin


def _record(monkeypatch, rc=0):
    """Capture both ffmpeg invocations the probe makes.

    The probe encodes a sample and then reads it back, so a fake that only
    understands one call fails on the other. The sample must actually appear
    on disk or the probe gives up before the interesting part.
    """
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        for i, a in enumerate(args):
            if a.endswith("sample.mp4") and args[i - 1] != "-i":
                open(a, "wb").write(b"fake")
        # The probe works in a temporary directory that is gone by the time
        # the test reads it, so the command file is captured here, while it
        # still exists.
        if "-vf" in args:
            graph = args[args.index("-vf") + 1]
            path = graph.split("sendcmd=f='", 1)[1].split("'", 1)[0]
            calls.append(("commands", open(path, encoding="utf-8").read()))
        return subprocess.CompletedProcess(args, rc, b"", b"")

    monkeypatch.setattr(ffmpeg_bin.subprocess, "run", fake_run)
    ffmpeg_bin._sendcmd_is_sane.cache_clear()
    return calls


def _graph(calls):
    for args in calls:
        if isinstance(args, list) and "-vf" in args:
            return args[args.index("-vf") + 1]
    raise AssertionError("the probe never ran a filtergraph")


def _commands(calls):
    for entry in calls:
        if isinstance(entry, tuple) and entry[0] == "commands":
            return entry[1]
    raise AssertionError("the probe wrote no command file")


def _invocations(calls):
    return [c for c in calls if isinstance(c, list)]


def test_the_probe_runs_the_shape_we_actually_render(monkeypatch):
    """Not a synthetic smoke test: sendcmd driving a named crop, which is
    the only filtergraph the renderer ever builds."""
    calls = _record(monkeypatch)
    assert ffmpeg_bin._sendcmd_is_sane("/fake/ffmpeg-a") is True
    graph = _graph(calls)
    assert "sendcmd=" in graph
    assert "crop@c" in graph


def test_the_probe_decodes_a_real_stream(monkeypatch):
    """Second thing an earlier version got wrong.

    The identical graph over `lavfi` frames runs in 0.2 s on the build that
    hangs on an encoded file. A probe that pipes generated frames straight
    into the graph therefore reports the broken ffmpeg as healthy — measured,
    not assumed. So the probe must encode a sample and read it back.
    """
    calls = _record(monkeypatch)
    ffmpeg_bin._sendcmd_is_sane("/fake/ffmpeg-decode-check")
    runs = _invocations(calls)
    assert len(runs) == 2, "expected an encode followed by a decode"

    encode, probe = runs
    assert "lavfi" in encode, "the sample should be generated, not shipped"
    assert "-vf" not in encode

    src_index = probe.index("-i") + 1
    assert probe[src_index].endswith(".mp4"), "the graph must read a file, not lavfi"
    assert "lavfi" not in probe


def test_the_probe_resizes_the_crop_not_just_moves_it(monkeypatch):
    """The first thing an earlier version got wrong.

    That probe only moved the window. Measured against the build it was
    written to reject: position-only commands finish in 2.3 s, resize
    commands never finish. Changing the crop's dimensions is what forces the
    chain to reconfigure, and it is what the punch-in zoom does on nearly
    every real clip.
    """
    calls = _record(monkeypatch)
    ffmpeg_bin._sendcmd_is_sane("/fake/ffmpeg-resize-check")

    commands = _commands(calls)

    assert " crop@c w " in commands, "the probe never resizes, so it cannot detect the fault"
    assert " crop@c h " in commands
    widths = {
        line.split(" w ")[1].rstrip(";")
        for line in commands.splitlines()
        if " crop@c w " in line
    }
    assert len(widths) > 1, "every command asks for the same width; nothing reconfigures"


def test_a_build_that_cannot_even_make_the_sample_is_rejected(monkeypatch):
    monkeypatch.setattr(
        ffmpeg_bin.subprocess, "run",
        lambda args, **kw: subprocess.CompletedProcess(args, 1, b"", b"nope"),
    )
    ffmpeg_bin._sendcmd_is_sane.cache_clear()
    assert ffmpeg_bin._sendcmd_is_sane("/fake/ffmpeg-cannot-encode") is False


def test_a_build_that_hangs_is_rejected(monkeypatch):
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 1))

    monkeypatch.setattr(ffmpeg_bin.subprocess, "run", fake_run)
    ffmpeg_bin._sendcmd_is_sane.cache_clear()
    assert ffmpeg_bin._sendcmd_is_sane("/fake/ffmpeg-slow") is False


def test_a_build_that_errors_is_rejected(monkeypatch):
    monkeypatch.setattr(
        ffmpeg_bin.subprocess, "run",
        lambda args, **kw: subprocess.CompletedProcess(args, 1, b"", b"boom"),
    )
    ffmpeg_bin._sendcmd_is_sane.cache_clear()
    assert ffmpeg_bin._sendcmd_is_sane("/fake/ffmpeg-broken") is False


def test_a_missing_binary_is_rejected_not_raised(monkeypatch):
    def fake_run(args, **kwargs):
        raise OSError("no such file")

    monkeypatch.setattr(ffmpeg_bin.subprocess, "run", fake_run)
    ffmpeg_bin._sendcmd_is_sane.cache_clear()
    assert ffmpeg_bin._sendcmd_is_sane("/fake/absent") is False


def test_the_probe_is_cached_per_binary(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args[0])
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(ffmpeg_bin.subprocess, "run", fake_run)
    ffmpeg_bin._sendcmd_is_sane.cache_clear()
    for _ in range(3):
        ffmpeg_bin._sendcmd_is_sane("/fake/one")
    assert len(calls) == 1, "the probe re-ran for a binary it had already judged"


def test_linux_downloads_when_the_system_build_is_unusable(monkeypatch):
    """The bug: supports_captions() alone short-circuited the download, so a
    capable-but-hanging distro build was accepted and never replaced."""
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(ffmpeg_bin, "supports_captions", lambda: True)
    monkeypatch.setattr(ffmpeg_bin, "ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(ffmpeg_bin, "_sendcmd_is_sane", lambda b: False)
    monkeypatch.setattr(ffmpeg_bin.config, "ensure_home", lambda: None)

    called = []
    monkeypatch.setattr(
        ffmpeg_bin, "_ensure_capable_linux",
        lambda progress: called.append(True) or False,
    )
    ffmpeg_bin.ensure_capable()
    assert called, "a hanging system ffmpeg was accepted instead of replaced"


def test_a_healthy_linux_build_is_left_alone(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(ffmpeg_bin, "supports_captions", lambda: True)
    monkeypatch.setattr(ffmpeg_bin, "ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(ffmpeg_bin, "_sendcmd_is_sane", lambda b: True)
    monkeypatch.setattr(
        ffmpeg_bin, "_ensure_capable_linux",
        lambda progress: pytest.fail("downloaded over a working ffmpeg"),
    )
    assert ffmpeg_bin.ensure_capable() is True


def test_other_platforms_do_not_pay_for_the_probe(monkeypatch):
    """Windows and macOS get their build from us already; the extra second
    would buy nothing there."""
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setattr(ffmpeg_bin, "supports_captions", lambda: True)
    monkeypatch.setattr(
        ffmpeg_bin, "_sendcmd_is_sane",
        lambda b: pytest.fail("probed on a platform that does not need it"),
    )
    assert ffmpeg_bin.ensure_capable() is True


def test_the_linux_build_is_verified_against_the_publishers_manifest(monkeypatch):
    """Same supply-chain rule as the Windows path: no digest, no download."""
    monkeypatch.setattr(ffmpeg_bin, "_published_digest", lambda sums, name: None)
    monkeypatch.setattr(
        ffmpeg_bin, "_download",
        lambda *a, **k: pytest.fail("downloaded without a published digest"),
    )
    monkeypatch.setattr(
        ffmpeg_bin.config, "bin_dir",
        lambda: __import__("pathlib").Path("/nonexistent-bin-dir"),
    )
    assert ffmpeg_bin._ensure_capable_linux(None) is False


def test_the_linux_url_and_manifest_come_from_one_release():
    assert ffmpeg_bin._LINUX_TARBALL in ffmpeg_bin._STATIC_LINUX
    assert ffmpeg_bin._STATIC_LINUX.startswith(ffmpeg_bin._WINDOWS_RELEASE_BASE)
    assert ffmpeg_bin._WINDOWS_SUMS.startswith(ffmpeg_bin._WINDOWS_RELEASE_BASE)
