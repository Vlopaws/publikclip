"""ffmpeg binary resolution.

Caption burning needs an ffmpeg built with libass, and (as this very
machine demonstrates) Homebrew's slimmed `ffmpeg` formula ships without it.
Resolution order: PUBLIKCLIP_FFMPEG env → bundled sidecar binary (packaged
app) → Homebrew ffmpeg-full keg → PATH. The first candidate that actually
has the `subtitles` filter wins; if none do, the plain PATH binary is
returned with `has_subtitles=False` so the caller can degrade (render
without burned captions) with an honest message instead of a crash.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import zipfile
from functools import lru_cache
from pathlib import Path

from .. import config
from ..integrity import IntegrityError, digest_from_manifest, verify

_KEG_CANDIDATES = [
    "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",
    "/usr/local/opt/ffmpeg-full/bin/ffmpeg",
]

# Static macOS builds with libass (ffmpeg.martin-riedl.de). Used only when
# no capable ffmpeg exists on the machine — downloaded once into
# PUBLIKCLIP_HOME/bin so end users never touch Homebrew.
_STATIC_BASE = "https://ffmpeg.martin-riedl.de/redirect/latest/macos/{arch}/release/{tool}.zip"

# Static Windows build with libass: BtbN's GPL build ships ffmpeg.exe and
# ffprobe.exe (with the subtitles filter) in one zip under a stable
# latest-release URL. ~80 MB once, into PUBLIKCLIP_HOME/bin.
_WINDOWS_RELEASE_BASE = "https://github.com/BtbN/FFmpeg-Builds/releases/latest/download"
_WINDOWS_ZIP = "ffmpeg-master-latest-win64-gpl.zip"
_STATIC_WINDOWS = f"{_WINDOWS_RELEASE_BASE}/{_WINDOWS_ZIP}"

# BtbN ships a coreutils-style checksums.sha256 with every release; the zip
# is verified against it before anything is unpacked and executed. The macOS
# source publishes no digest at all (checked: no sidecar, no API) — see
# _ensure_capable_macos for what is done instead and what stays uncovered.
_WINDOWS_SUMS = f"{_WINDOWS_RELEASE_BASE}/checksums.sha256"

# The same publisher's Linux build, verified against the same manifest.
# Linux was the one platform with no managed option, which meant the
# deployment target — a cloud VM — ran whatever its distro shipped. Ubuntu
# 24.04 ships 6.1.1, and that build takes over 30 minutes on a filtergraph
# the current build finishes in 17 seconds. See _sendcmd_is_sane.
_LINUX_TARBALL = "ffmpeg-master-latest-linux64-gpl.tar.xz"
_STATIC_LINUX = f"{_WINDOWS_RELEASE_BASE}/{_LINUX_TARBALL}"

# How long a one-second sendcmd encode may take before the build is
# considered unusable. A healthy ffmpeg does it in well under a second; the
# pathological one does not finish at all. The gap is large enough that no
# reasonable machine speed sits between the two.
_SENDCMD_PROBE_TIMEOUT = 20.0

# Mach-O magics: 64-bit little-endian and the universal/fat wrappers.
_MACHO_MAGICS = (bytes.fromhex("cffaedfe"), bytes.fromhex("cafebabe"), bytes.fromhex("bebafeca"))

_EXE = ".exe" if platform.system() == "Windows" else ""


def _has_subtitles_filter(binary: str) -> bool:
    try:
        proc = subprocess.run(
            [binary, "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return " subtitles " in proc.stdout


@lru_cache(maxsize=1)
def resolve() -> tuple[str, bool]:
    """(ffmpeg_path, has_subtitles)."""
    candidates: list[str] = []
    env = os.environ.get("PUBLIKCLIP_FFMPEG")
    if env:
        candidates.append(env)
    candidates.append(str(config.bin_dir() / f"ffmpeg{_EXE}"))  # our downloaded static
    bundled = os.environ.get("PUBLIKCLIP_BUNDLED_FFMPEG")  # set by the app shell
    if bundled:
        candidates.append(bundled)
    if platform.system() == "Darwin":
        candidates.extend(_KEG_CANDIDATES)
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        candidates.append(path_ffmpeg)

    fallback: str | None = None
    for cand in candidates:
        if not os.path.exists(cand):
            continue
        fallback = fallback or cand
        if _has_subtitles_filter(cand):
            return cand, True
    return (fallback or "ffmpeg"), False


def ffmpeg() -> str:
    return resolve()[0]


def ffprobe() -> str:
    """ffprobe next to the resolved ffmpeg when present, else PATH."""
    sibling = Path(ffmpeg()).parent / f"ffprobe{_EXE}"
    if sibling.exists():
        return str(sibling)
    return shutil.which("ffprobe") or "ffprobe"


def supports_captions() -> bool:
    return resolve()[1]


def _download(url: str, dest: Path, sha256: str | None = None) -> bool:
    """Fetch `url` to `dest`. With `sha256`, the file is discarded and False
    returned unless it matches — callers treat False as "no capable ffmpeg",
    which degrades to rendering without burned captions rather than running
    an unverified binary."""
    import httpx

    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as res:
            if res.status_code != 200:
                return False
            with open(dest, "wb") as fh:
                for chunk in res.iter_bytes():
                    fh.write(chunk)
    except (httpx.HTTPError, OSError):
        return False
    if sha256:
        try:
            verify(dest, sha256, dest.name)
        except (IntegrityError, OSError):
            return False
    return True


def _published_digest(sums_url: str, filename: str) -> str | None:
    """The digest the publisher lists for `filename`, or None if the
    manifest is unreachable or silent about it."""
    import httpx

    try:
        res = httpx.get(sums_url, follow_redirects=True, timeout=60.0)
        res.raise_for_status()
    except httpx.HTTPError:
        return None
    return digest_from_manifest(res.text, filename)


def _looks_like_macho(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(4) in _MACHO_MAGICS
    except OSError:
        return False


@lru_cache(maxsize=8)
def _sendcmd_is_sane(binary: str) -> bool:
    """Can this build run a sendcmd-driven crop at a sane speed?

    Every render is one `sendcmd` file driving one `crop@c`, so this is not
    an exotic corner — it is the only filtergraph this project produces.
    Ubuntu 24.04's ffmpeg 6.1.1 supports the filters, reports them in
    `-filters`, and then takes more than thirty minutes on a twenty-six
    second clip that a current build renders in seventeen seconds. A
    capability probe cannot see that; only a clock can.

    It matters WHICH parameter the commands change, and the first version of
    this probe got it wrong — it moved the window and passed on the very
    build it was written to reject. Measured against that build:

        position only (x)     2.3 s      resize (w/h)   never finishes

    Changing the crop's dimensions is what forces everything downstream to
    reconfigure, and that is the path 6.1.1 falls off. It is also not an
    exotic case here: the punch-in zoom resizes the window, so most clips
    carry these commands.

    Resolution turns out not to matter — the slow build hangs at 240x426
    too — so the probe stays small and a healthy build answers in about a
    second.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="publikclip-probe-") as tmp:
        cmd_file = Path(tmp) / "probe.cmd"
        lines = []
        for i in range(25):
            width = 320 - (i % 10) * 2
            lines.append(f"{i / 25:.4f} crop@c w {width};")
            lines.append(f"{i / 25:.4f} crop@c h {width * 2};")
        cmd_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        graph = (
            f"sendcmd=f='{cmd_file.as_posix()}',"
            "crop@c=w=320:h=240:x=100:y=0,"
            "scale=240:426:flags=lanczos,setsar=1"
        )
        try:
            proc = subprocess.run(
                [
                    binary, "-v", "error", "-f", "lavfi",
                    "-i", "testsrc=size=640x480:rate=25:duration=1",
                    "-vf", graph, "-c:v", "libx264", "-preset", "ultrafast",
                    "-f", "null", "-",
                ],
                capture_output=True, timeout=_SENDCMD_PROBE_TIMEOUT,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
    return proc.returncode == 0


def _ensure_capable_linux(progress) -> bool:
    """BtbN's Linux build, through the same verified path as Windows."""
    import tarfile

    wanted = {"ffmpeg": config.bin_dir() / "ffmpeg", "ffprobe": config.bin_dir() / "ffprobe"}
    if all(d.exists() for d in wanted.values()) and _has_subtitles_filter(str(wanted["ffmpeg"])):
        return True
    if progress:
        progress(-1, "Downloading ffmpeg (one-time, caption support)…")
    archive = config.bin_dir() / _LINUX_TARBALL
    expected = _published_digest(_WINDOWS_SUMS, _LINUX_TARBALL)
    if not expected:
        return False  # no manifest, no trust — degrade instead of guessing
    try:
        if not _download(_STATIC_LINUX, archive, sha256=expected):
            return False
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                base = member.name.rsplit("/", 1)[-1]
                if base in wanted and "/bin/" in member.name and member.isfile():
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        continue
                    wanted[base].write_bytes(extracted.read())
                    wanted[base].chmod(0o755)
    except (OSError, tarfile.TarError):
        return False
    finally:
        archive.unlink(missing_ok=True)
    return all(d.exists() for d in wanted.values())


def _ensure_capable_macos(progress) -> bool:
    """ffmpeg.martin-riedl.de publishes no checksum manifest, so there is no
    publisher digest to pin against — the honest state of this path is:
    HTTPS (host + transport authenticated), a well-formed zip, a real Mach-O
    payload, and the `-filters` capability probe before it is used. That
    rules out corruption and a wrong-file swap, but NOT a compromise of the
    build host itself. Windows takes the verified path below; if this
    upstream ever ships digests, route it through `_published_digest` too.
    """
    arch = "arm64" if platform.machine() == "arm64" else "amd64"
    for tool in ("ffmpeg", "ffprobe"):
        dest = config.bin_dir() / tool
        if dest.exists() and (_has_subtitles_filter(str(dest)) if tool == "ffmpeg" else True):
            continue
        if progress:
            progress(-1, f"Downloading {tool} (one-time, caption support)…")
        zpath = dest.with_suffix(".zip")
        try:
            if not _download(_STATIC_BASE.format(arch=arch, tool=tool), zpath):
                return False
            with zipfile.ZipFile(zpath) as zf:
                for name in zf.namelist():
                    if name.rstrip("/").endswith(tool):
                        dest.write_bytes(zf.read(name))
                        break
            if not _looks_like_macho(dest):
                dest.unlink(missing_ok=True)
                return False
            dest.chmod(0o755)
        except (OSError, zipfile.BadZipFile):
            return False
        finally:
            zpath.unlink(missing_ok=True)
    return True


def _ensure_capable_windows(progress) -> bool:
    """One zip carries both tools (BtbN GPL build, libass included)."""
    wanted = {
        "ffmpeg.exe": config.bin_dir() / "ffmpeg.exe",
        "ffprobe.exe": config.bin_dir() / "ffprobe.exe",
    }
    if all(dest.exists() for dest in wanted.values()) and _has_subtitles_filter(
        str(wanted["ffmpeg.exe"])
    ):
        return True
    if progress:
        progress(-1, "Downloading ffmpeg (one-time, caption support)…")
    zpath = config.bin_dir() / "ffmpeg-static.zip"
    expected = _published_digest(_WINDOWS_SUMS, _WINDOWS_ZIP)
    if not expected:
        return False  # no manifest, no trust — degrade instead of guessing
    try:
        if not _download(_STATIC_WINDOWS, zpath, sha256=expected):
            return False
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                base = name.rsplit("/", 1)[-1]
                if base in wanted and "/bin/" in name:
                    wanted[base].write_bytes(zf.read(name))
    except (OSError, zipfile.BadZipFile):
        return False
    finally:
        zpath.unlink(missing_ok=True)
    return all(dest.exists() for dest in wanted.values())


def ensure_on_path(progress=None) -> str | None:
    """Put the managed ffmpeg on PATH for this process.

    Resolving ffmpeg ourselves only helps the code that asks us. Third-party
    libraries shell out to a bare `ffmpeg` and fail with a naked
    "FileNotFoundError: [WinError 2]" that names nothing — whisperx does
    exactly this to decode audio. Rather than patch each library, make the
    binary findable the way they all expect.

    Returns the directory added, or None if no ffmpeg could be produced.
    """
    path = ffmpeg()
    if not os.path.exists(path):
        found = shutil.which(path)
        if found:
            path = found
        else:
            ensure_capable(progress)
            resolve.cache_clear()
            path = ffmpeg()
    if not os.path.exists(path):
        return None
    directory = str(Path(path).parent)
    current = os.environ.get("PATH", "")
    if directory not in current.split(os.pathsep):
        os.environ["PATH"] = directory + os.pathsep + current
    return directory


def ensure_capable(progress=None) -> bool:
    """If no libass ffmpeg exists anywhere, download a static build once
    into PUBLIKCLIP_HOME/bin (macOS arm64/x86_64, Windows x64), then
    re-resolve. Returns whether caption burning is available afterwards."""
    system = platform.system()
    # Supporting the filters is necessary and, on Linux, not sufficient: a
    # distro build can pass every capability probe and still be unusable on
    # the one filtergraph this project renders. Only there is the extra
    # timing probe worth its second.
    if supports_captions() and (system != "Linux" or _sendcmd_is_sane(ffmpeg())):
        return True
    config.ensure_home()
    if system == "Darwin":
        ok = _ensure_capable_macos(progress)
    elif system == "Windows":
        ok = _ensure_capable_windows(progress)
    elif system == "Linux":
        ok = _ensure_capable_linux(progress)
    else:
        return False
    if not ok:
        return False
    resolve.cache_clear()
    return supports_captions()
