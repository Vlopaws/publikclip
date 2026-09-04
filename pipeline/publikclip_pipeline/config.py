"""Paths and settings.

Everything lives under PUBLIKCLIP_HOME (default ~/.publikclip):

    ~/.publikclip/
      db.sqlite3          job + stage bookkeeping
      bin/                managed binaries (yt-dlp)
      models/             downloaded model weights
      jobs/<job_id>/      per-job artifacts (media, audio, stage checkpoints)

The desktop app points PUBLIKCLIP_HOME at its own app-data dir; the CLI uses
the default. Artifacts on disk are the source of truth — the DB only records
what should exist so a stage can decide whether to skip itself on resume.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path


def home_dir() -> Path:
    return Path(os.environ.get("PUBLIKCLIP_HOME", str(Path.home() / ".publikclip")))


def jobs_dir() -> Path:
    return home_dir() / "jobs"


def bin_dir() -> Path:
    return home_dir() / "bin"


def models_dir() -> Path:
    return home_dir() / "models"


def db_path() -> Path:
    return home_dir() / "db.sqlite3"


def read_text(path: Path) -> str:
    """Read a text artifact, tolerating files written before UTF-8 was
    explicit, and repairing them in place.

    Everything the pipeline writes is UTF-8 now, but jobs created by an
    earlier build carry checkpoints in the Windows locale encoding — reading
    those strictly turns an upgrade into a crash on somebody\'s half-finished
    work. Falling back and rewriting means the wrong encoding is fixed the
    first time the file is touched, rather than forever tolerated.
    """
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    text = raw.decode("cp1252", errors="replace")
    try:
        path.write_text(text, encoding="utf-8")
    except OSError:
        pass  # read-only artifact: still return what we decoded
    return text


def secrets() -> dict:
    """Everything in PUBLIKCLIP_HOME/secrets.json, or {} if unreadable."""
    path = home_dir() / "secrets.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def secret(field: str, env: str) -> str | None:
    """Env var first, then secrets.json.

    One reader for every credential the pipeline takes. Env wins so a single
    run can be pointed elsewhere without touching the file the app owns, and
    a lone implementation means a new secret cannot quietly miss the
    handling the others get.
    """
    value = os.environ.get(env)
    if not value:
        value = secrets().get(field)
    return value.strip() if isinstance(value, str) and value.strip() else None


def ensure_home() -> Path:
    root = home_dir()
    for d in (root, jobs_dir(), bin_dir(), models_dir()):
        d.mkdir(parents=True, exist_ok=True)
    return root


# Hard per-attempt network timeouts (seconds). A blackholed connection must
# never freeze the pipeline — every subprocess/network call takes one of these.
HTTP_TIMEOUT = 60.0
SUBPROCESS_INACTIVITY_TIMEOUT = 120.0  # kill if no output for this long
PROBE_TIMEOUT = 60.0

# Ingest
MAX_HEIGHT = 1080
AUDIO_SR = 16_000  # analysis sample rate; every M1 model consumes this wav


@dataclass
class CameraSettings:
    """User-facing camera preset knobs (locked decision #7: exposed options)."""

    # 'cut' = hard cut on speaker change (default), 'pan' = eased pan between
    # speakers, 'locked' = static crop on the dominant face.
    speaker_change: str = "cut"
    pan_duration_s: float = 0.6
    deadzone_frac: float = 0.05  # ignore drift below this fraction of width
    punch_in: bool = True
    punch_in_sensitivity: float = 1.0  # scales event/energy trigger thresholds
    zoom_lock_per_scene: bool = True
    # None = decide per clip from what the ASD pass measured (see
    # camera.framing). "vertical" or "wide" pins every clip to one shape,
    # for material the operator already knows the answer for.
    framing: str | None = None


@dataclass
class Settings:
    """Per-job settings snapshot. Serialized into the job dir at creation so a
    resumed job never silently picks up changed defaults."""

    camera: CameraSettings = field(default_factory=CameraSettings)
    # How many scored moments reach the camera and the renderer. Those two
    # stages cost more than everything before them combined — 25 minutes
    # each for twelve clips — and the autopilot then publishes three. Left
    # at None the historical 12 applies, which is right for an operator who
    # will look through the lot; the autopilot lowers it to what it can
    # actually use.
    max_finalists: int | None = None
    # Regions the operator wants cut whatever the interest curve thinks,
    # as [[start_sec, end_sec], ...]. The curve ranks the whole video, so a
    # stretch that is quiet but good never reaches the cut — see
    # candidates.windows.focus_peaks for the measurement that motivated it.
    focus: list = field(default_factory=list)
    lufs_target: float = -14.0  # decision #8: configurable per destination
    true_peak_db: float = -1.0
    # Defaults to the backend that cannot generate a bill. Every hosted
    # option here is metered, and an unattended run that picks one by
    # default turns a forgotten cron job into an invoice. Choosing a paid
    # backend stays an explicit act: --llm, or the picker in the app.
    llm_mode: str = "ollama"
    caption_preset: str = "classic"
    # jrgillick laughter specialist: 10 ms precision but ~300k CPU forward
    # passes on an hour-plus source. OFF by default — PANNs' AudioSet
    # laughter classes cover the bus at 320 ms resolution for a fraction of
    # the compute; flip on for the two-detector agreement boost.
    laughter_specialist: bool = False

    # Serialization walks the dataclass rather than listing fields by hand.
    #
    # The hand-written version needed a new setting added in three places,
    # and silently ignored it if you missed one. That is not hypothetical:
    # max_finalists was added to the dataclass and set by the autopilot, and
    # never reached the job, because to_json did not know about it. The
    # symptom was a run rendering twelve clips while the code that asked for
    # six looked correct at every point you would think to check.
    def to_json(self) -> dict:
        out: dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = value.__dict__.copy() if is_dataclass(value) else value
        return out

    @classmethod
    def from_json(cls, data: dict) -> "Settings":
        """Rebuild from a snapshot, tolerating both directions of drift.

        A job dir outlives the code that wrote it: settings.json may predate
        a field (use the default) or postdate one this build knows nothing
        about (ignore it). Neither is an error worth refusing to resume a
        job over.
        """
        known = {f.name: f for f in fields(cls)}
        kwargs = {}
        for name, field_def in known.items():
            if name not in data:
                continue
            value = data[name]
            nested = field_def.type
            if name == "camera" and isinstance(value, dict):
                allowed = {f.name for f in fields(CameraSettings)}
                kwargs[name] = CameraSettings(
                    **{k: v for k, v in value.items() if k in allowed}
                )
            else:
                kwargs[name] = value
        return cls(**kwargs)
