"""Model weight registry + downloader.

All weights land in PUBLIKCLIP_HOME/models/<name>. Downloads resume (Range)
and every entry carries a pinned sha256 — `ModelSpec.sha256` has no default,
so a model physically cannot be registered without one. This matters more
than a corruption check: a .pth is a pickle and onnxruntime loads whatever
graph it is handed, so an unverified weight file is arbitrary code
execution, not a bad score.

Verification runs on every use, not just on download. A digest recorded
beside the file would prove nothing (whoever can rewrite the weights can
rewrite the record), and the `dest.exists()` fast path would otherwise trust
forever anything already on disk — including weights fetched by an older
build that verified nothing at all. Re-hashing the full model set costs
~0.3 s against a job measured in minutes; the result is memoised per process
so a stage that asks twice pays once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from .. import config
from ..integrity import verify

ProgressFn = Callable[[float, str], None]


@dataclass(frozen=True)
class ModelSpec:
    name: str            # registry key + subdir name
    filename: str
    url: str
    sha256: str          # required: see module docstring
    approx_mb: int = 0


REGISTRY: dict[str, ModelSpec] = {}

# (path, sha256) pairs already verified in this process.
_verified: set[tuple[str, str]] = set()


def register(spec: ModelSpec) -> ModelSpec:
    REGISTRY[f"{spec.name}/{spec.filename}"] = spec
    return spec


def model_path(spec: ModelSpec) -> Path:
    return config.models_dir() / spec.name / spec.filename


def is_present(spec: ModelSpec) -> bool:
    return model_path(spec).exists()


def _check(path: Path, spec: ModelSpec, progress: ProgressFn | None = None) -> None:
    """Verify unless this exact (path, digest) already passed this process.

    On mismatch `verify` deletes the file, so the next call re-downloads
    rather than failing forever on a bad copy.
    """
    key = (str(path), spec.sha256)
    if key in _verified:
        return
    if progress and spec.approx_mb >= 100:
        progress(-1, f"Verifying {spec.name}…")
    verify(path, spec.sha256, spec.name)
    _verified.add(key)


def ensure(spec: ModelSpec, progress: ProgressFn) -> Path:
    """Download with resume if needed; always verify before returning."""
    dest = model_path(spec)
    if dest.exists():
        _check(dest, spec, progress)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    offset = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    label = f"Downloading {spec.name}" + (f" (~{spec.approx_mb} MB)" if spec.approx_mb else "")
    with httpx.stream(
        "GET", spec.url, headers=headers, follow_redirects=True, timeout=config.HTTP_TIMEOUT
    ) as res:
        if res.status_code == 200 and offset:
            offset = 0  # server ignored Range; start over
            tmp.unlink(missing_ok=True)
        elif res.status_code not in (200, 206):
            raise RuntimeError(f"Model download failed for {spec.name}: HTTP {res.status_code}")
        total = int(res.headers.get("content-length", 0)) + offset
        seen = offset
        with open(tmp, "ab") as fh:
            for chunk in res.iter_bytes():
                fh.write(chunk)
                seen += len(chunk)
                if total:
                    progress(seen / total, label)
    verify(tmp, spec.sha256, spec.name)   # nothing reaches dest unverified
    tmp.replace(dest)
    _verified.add((str(dest), spec.sha256))
    return dest
