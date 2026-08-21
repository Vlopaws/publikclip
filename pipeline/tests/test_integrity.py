"""Integrity tests — a weight file is a pickle, so "we verified it" has to
be true by construction, not by convention."""

import dataclasses
import hashlib

import pytest

from publikclip_pipeline import integrity
from publikclip_pipeline.models import registry, specs


def _write(tmp_path, name, payload: bytes):
    path = tmp_path / name
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def test_verify_accepts_matching_digest(tmp_path):
    path, digest = _write(tmp_path, "weights.bin", b"good bytes")
    integrity.verify(path, digest, "weights")
    assert path.exists()


def test_verify_is_case_insensitive(tmp_path):
    path, digest = _write(tmp_path, "weights.bin", b"good bytes")
    integrity.verify(path, digest.upper(), "weights")


def test_verify_rejects_and_discards_on_mismatch(tmp_path):
    path, _ = _write(tmp_path, "weights.bin", b"tampered bytes")
    with pytest.raises(integrity.IntegrityError):
        integrity.verify(path, "0" * 64, "weights")
    # Left on disk it would be picked up unchecked by the next run's
    # `dest.exists()` fast path — deletion is the point, not tidiness.
    assert not path.exists()


def test_verify_can_keep_the_file_when_asked(tmp_path):
    path, _ = _write(tmp_path, "weights.bin", b"tampered bytes")
    with pytest.raises(integrity.IntegrityError):
        integrity.verify(path, "0" * 64, "weights", discard=False)
    assert path.exists()


@pytest.mark.parametrize(
    "manifest, wanted, expected",
    [
        ("abc  yt-dlp.exe", "yt-dlp.exe", "abc"),
        ("ABC  yt-dlp.exe", "yt-dlp.exe", "abc"),          # normalised
        ("abc *yt-dlp.exe", "yt-dlp.exe", "abc"),          # binary-mode marker
        ("abc  dist/yt-dlp.exe", "yt-dlp.exe", "abc"),     # directory prefix
        ("abc  other.exe", "yt-dlp.exe", None),            # not listed
        ("garbage line", "yt-dlp.exe", None),
        ("", "yt-dlp.exe", None),
    ],
)
def test_digest_from_manifest(manifest, wanted, expected):
    assert integrity.digest_from_manifest(manifest, wanted) == expected


def test_manifest_does_not_match_on_suffix():
    """`ffmpeg.exe` must not satisfy a request for `ffprobe.exe` or vice
    versa — matching is on the whole basename."""
    assert integrity.digest_from_manifest("abc  not-ffmpeg.exe", "ffmpeg.exe") is None


def test_model_spec_cannot_be_registered_without_a_pin():
    with pytest.raises(TypeError):
        registry.ModelSpec(name="x", filename="y.onnx", url="https://example.invalid/y")


def test_every_registered_model_is_pinned_to_an_immutable_revision():
    assert registry.REGISTRY, "registry is empty — specs did not import"
    for key, spec in registry.REGISTRY.items():
        assert len(spec.sha256) == 64, f"{key}: sha256 is not a full digest"
        int(spec.sha256, 16)  # hex only
        for mutable in ("/raw/main/", "/raw/master/", "/resolve/main/", "/resolve/master/"):
            assert mutable not in spec.url, f"{key}: {mutable} is a moving target"


def test_registry_reverifies_a_file_swapped_after_install(tmp_path, monkeypatch):
    """The `dest.exists()` fast path used to trust anything already on disk,
    including weights fetched before pinning existed."""
    monkeypatch.setenv("PUBLIKCLIP_HOME", str(tmp_path))
    payload = b"legitimate weights"
    spec = dataclasses.replace(
        specs.LR_ASD_BACKEND, sha256=hashlib.sha256(payload).hexdigest()
    )
    dest = registry.model_path(spec)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)

    # First use verifies in place; no network involved.
    assert registry.ensure(spec, lambda f, m: None) == dest

    # A later process sees the swap. (Within one process the result is
    # memoised, so clear that cache to model the next run.)
    dest.write_bytes(b"swapped by an attacker")
    registry._verified.clear()
    with pytest.raises(integrity.IntegrityError):
        registry.ensure(spec, lambda f, m: None)
    assert not dest.exists(), "a failed verification must not leave the file behind"
