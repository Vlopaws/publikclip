"""Checksum verification for everything we download and then trust.

Three classes of artifact land in PUBLIKCLIP_HOME and are later executed or
deserialized: model weights (fed straight into torch/onnxruntime), the
yt-dlp binary, and a static ffmpeg. TLS only proves we reached the right
host — it says nothing about the bytes a compromised release asset, a
rewritten branch, or a poisoned mirror would serve. Anything with a
publisher-provided digest is checked against it *before* it is moved into
place; anything without one says so out loud at the call site rather than
skipping the check silently.

Model weights are the sharpest edge: a .pth is a pickle, so loading an
attacker-controlled one is arbitrary code execution, not just a bad result.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


class IntegrityError(Exception):
    """A download did not match its expected digest — never install it."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str, label: str, *, discard: bool = True) -> None:
    """Raise unless `path` hashes to `expected`, discarding the file first.

    Deleting on mismatch is deliberate: a half-trusted binary left on disk
    is exactly what the next run's `if dest.exists()` fast path would pick
    up without re-checking.
    """
    actual = sha256_file(path)
    if actual.lower() != expected.strip().lower():
        if discard:
            path.unlink(missing_ok=True)
        raise IntegrityError(
            f"{label}: checksum mismatch (expected {expected}, got {actual}). "
            "The download was corrupted or tampered with and has been discarded."
        )


def digest_from_manifest(text: str, filename: str) -> str | None:
    """Pull one digest out of a `sha256sum`-style manifest.

    Both manifests we consume — yt-dlp's SHA2-256SUMS and BtbN's
    checksums.sha256 — use the coreutils format `<hex>  <name>`, where the
    name may carry a `*` binary-mode marker or a directory prefix.
    """
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        digest, name = parts
        if name.lstrip("*").rsplit("/", 1)[-1] == filename:
            return digest.lower()
    return None
