"""Hosted transcription, for machines without a usable GPU.

Whisper large-v3-turbo on eight laptop cores runs at roughly 3x realtime:
measured here, 63 minutes to transcribe 20 minutes of audio. That single
stage dominates the pipeline, and no amount of paying for a smarter scoring
model touches it. Groq runs the same model on its own hardware and returns
in seconds.

Two details make this a drop-in rather than a rewrite:

- The pipeline needs word-level timestamps (karaoke captions, prosodic
  emphasis, clip boundaries snapped to words). `timestamp_granularities`
  gives them, so the local forced-alignment pass is not needed.
- Groq returns words in one flat list, not nested per segment — the OpenAI
  shape. They are re-attached here by time containment so the rest of the
  pipeline sees exactly what whisperx produced.

Per-word confidence has no equivalent in the hosted response, so `score` is
reported as 0.0. Callers already guard on `score > 0` before using it.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import httpx

from .. import config

API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_MODEL = "whisper-large-v3-turbo"
API_KEY_ENV = "PUBLIKCLIP_GROQ_API_KEY"

# Groq rejects oversized uploads, and a 16 kHz mono WAV of a long episode
# clears 25 MB easily (36.5 MB for 20 minutes here). FLAC is lossless, so
# compressing costs nothing in accuracy and roughly halves the payload.
UPLOAD_LIMIT_BYTES = 24 * 1024 * 1024
UPLOAD_TIMEOUT = 600.0


class HostedAsrError(Exception):
    """User-actionable hosted-transcription failure."""


def api_key() -> str | None:
    return config.secret("groq_api_key", API_KEY_ENV)


def available() -> bool:
    return bool(api_key())


def _to_flac(audio_path: Path, progress) -> Path:
    """Lossless re-encode, so a long episode fits the upload limit."""
    from ..render import ffmpeg_bin

    dest = audio_path.with_suffix(".flac")
    if dest.exists() and dest.stat().st_size <= UPLOAD_LIMIT_BYTES:
        return dest
    progress(-1, "Compressing audio for upload…")
    proc = subprocess.run(
        [
            ffmpeg_bin.ffmpeg(), "-y", "-i", str(audio_path),
            "-ac", "1", "-ar", "16000", "-c:a", "flac", "-compression_level", "8",
            str(dest),
        ],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0 or not dest.exists():
        raise HostedAsrError(f"Could not compress the audio for upload: {proc.stderr[-300:]}")
    return dest


# Segment and word timestamps come from the same model but are rounded
# independently, so a word can start a few milliseconds before the segment
# that owns it. Without slack those words vanish from the captions; with too
# much, a word from a neighbouring segment gets stolen.
_BOUNDARY_SLACK = 0.05


def _attach_words(segments: list[dict], words: list[dict]) -> list[dict]:
    """Put each word back inside the segment that contains it.

    Walks both lists once in time order rather than searching per word — a
    long episode carries tens of thousands of words, and the quadratic
    version is slower than the network call it follows.
    """
    out = []
    index = 0
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        collected = []
        while index < len(words):
            word = words[index]
            w_start = float(word.get("start", 0.0))
            if w_start < start - _BOUNDARY_SLACK:
                index += 1  # genuinely elsewhere, not boundary jitter
                continue
            if w_start >= end:
                break
            collected.append(
                {
                    "word": (word.get("word") or "").strip(),
                    "start": round(w_start, 3),
                    "end": round(float(word.get("end", w_start)), 3),
                    # No per-word confidence in the hosted response.
                    "score": 0.0,
                }
            )
            index += 1
        out.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": (seg.get("text") or "").strip(),
                "words": collected,
            }
        )
    return out


def transcribe(
    audio_path: Path,
    language: str | None = None,
    model: str = DEFAULT_MODEL,
    progress=None,
) -> dict:
    """Transcribe with word timestamps. Returns the ASR stage's payload."""
    emit = progress or (lambda fraction, message: None)
    key = api_key()
    if not key:
        raise HostedAsrError(
            "Hosted transcription needs a Groq API key. Get one at "
            f"console.groq.com/keys, then set {API_KEY_ENV} (or groq_api_key "
            "in ~/.publikclip/secrets.json)."
        )

    upload = audio_path
    if audio_path.stat().st_size > UPLOAD_LIMIT_BYTES:
        upload = _to_flac(audio_path, emit)
        if upload.stat().st_size > UPLOAD_LIMIT_BYTES:
            raise HostedAsrError(
                f"Audio is still {upload.stat().st_size / 1e6:.0f} MB after "
                "compression, above the upload limit. Split the source, or "
                "transcribe locally with PUBLIKCLIP_ASR_BACKEND=local."
            )

    emit(-1, f"Transcribing via Groq ({model})…")
    data = {
        "model": model,
        "response_format": "verbose_json",
        "timestamp_granularities[]": ["word", "segment"],
    }
    if language:
        data["language"] = language

    started = time.monotonic()
    try:
        with open(upload, "rb") as fh:
            res = httpx.post(
                API_URL,
                headers={"Authorization": f"Bearer {key}"},
                data=data,
                files={"file": (upload.name, fh, "audio/flac")},
                timeout=UPLOAD_TIMEOUT,
            )
    except httpx.HTTPError as err:
        raise HostedAsrError(f"Groq transcription request failed: {err}") from err

    if res.status_code in (401, 403):
        raise HostedAsrError("Groq rejected the API key. Check it in Settings.")
    if res.status_code == 413:
        raise HostedAsrError(
            "Groq refused the upload as too large even after compression. "
            "Split the source, or use PUBLIKCLIP_ASR_BACKEND=local."
        )
    if res.status_code != 200:
        raise HostedAsrError(f"Groq transcription failed (HTTP {res.status_code}): {res.text[:300]}")

    payload = res.json()
    segments = payload.get("segments") or []
    words = payload.get("words") or []
    if not segments:
        raise HostedAsrError("Groq returned no segments — the audio may be silent or unreadable.")

    elapsed = time.monotonic() - started
    attached = _attach_words(segments, words)
    audio_sec = float(segments[-1].get("end", 0.0))
    emit(-1, f"Transcribed {audio_sec / 60:.0f} min in {elapsed:.0f}s")

    return {
        "language": payload.get("language") or language or "unknown",
        "model": model,
        "backend": "groq",
        "compute_type": "hosted",
        "word_count": sum(len(s["words"]) for s in attached),
        "benchmark": {
            "audio_sec": round(audio_sec, 1),
            "transcribe_sec": round(elapsed, 1),
            "align_sec": 0.0,
        },
        "segments": attached,
    }
