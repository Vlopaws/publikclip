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


# A long episode does not fit the upload limit at any compression, so it is
# sent in pieces. The chunk length is derived from the file's own byte rate
# rather than assumed from a bitrate, because the encoder's ratio depends on
# the material — speech over silence compresses far better than a live room.
CHUNK_SAFETY = 0.85

# Chunks overlap so a word straddling a boundary is heard whole by at least
# one of them. Two seconds is longer than any single word and short enough
# that the duplicate region stays trivial to reconcile.
CHUNK_OVERLAP_SEC = 2.0

# Below this a chunk is not worth a round trip; the tail is folded into its
# predecessor instead. Also keeps the 10-second minimum billing from being
# paid for a sliver.
MIN_CHUNK_SEC = 30.0


def chunk_plan(
    duration_sec: float, total_bytes: int, limit: int | None = None
) -> list[tuple[float, float]]:
    """(start, length) spans covering the audio, each expected to fit `limit`.

    Returns a single span when the whole file already fits — the common case,
    and the one that must not pay for any of this.
    """
    limit = limit or UPLOAD_LIMIT_BYTES
    if total_bytes <= limit or duration_sec <= 0:
        return [(0.0, duration_sec)]

    bytes_per_sec = total_bytes / duration_sec
    span = max(MIN_CHUNK_SEC, (limit * CHUNK_SAFETY) / bytes_per_sec)

    spans: list[tuple[float, float]] = []
    cursor = 0.0
    while cursor < duration_sec:
        remaining = duration_sec - cursor
        if remaining <= span + MIN_CHUNK_SEC:
            spans.append((cursor, remaining))  # fold the tail in
            break
        spans.append((cursor, span))
        cursor += span - CHUNK_OVERLAP_SEC
    return spans


def _audio_duration(path: Path) -> float:
    from ..render import ffmpeg_bin

    proc = subprocess.run(
        [
            ffmpeg_bin.ffprobe(), "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nokey=1:noprint_wrappers=1", str(path),
        ],
        capture_output=True, text=True, timeout=120,
    )
    try:
        return float((proc.stdout or "0").strip())
    except ValueError:
        return 0.0


def _extract_flac(audio_path: Path, dest: Path, start: float, length: float) -> Path:
    """One span, losslessly compressed, ready to upload."""
    from ..render import ffmpeg_bin

    proc = subprocess.run(
        [
            ffmpeg_bin.ffmpeg(), "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
            "-i", str(audio_path),
            "-ac", "1", "-ar", "16000", "-c:a", "flac", "-compression_level", "8",
            str(dest),
        ],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0 or not dest.exists():
        raise HostedAsrError(f"Could not cut the audio for upload: {proc.stderr[-300:]}")
    return dest


def _post(upload: Path, key: str, model: str, language: str | None) -> dict:
    """One transcription request. Raises HostedAsrError with what to do."""
    data = {
        "model": model,
        "response_format": "verbose_json",
        "timestamp_granularities[]": ["word", "segment"],
    }
    if language:
        data["language"] = language
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
            f"Groq refused a {upload.stat().st_size / 1e6:.0f} MB upload as too "
            "large. Lower UPLOAD_LIMIT_BYTES, or use PUBLIKCLIP_ASR_BACKEND=local."
        )
    if res.status_code != 200:
        raise HostedAsrError(
            f"Groq transcription failed (HTTP {res.status_code}): {res.text[:300]}"
        )
    return res.json()


def _merge(into: list[dict], new: list[dict], offset: float) -> None:
    """Append `new` to `into`, shifted by `offset`, dropping the overlap.

    Consecutive chunks share CHUNK_OVERLAP_SEC of audio, so the same speech
    is transcribed twice. The duplicate is resolved by time rather than by
    text: a segment starting before everything already accepted is the
    second telling of something already recorded.
    """
    last_end = into[-1]["end"] if into else float("-inf")
    for seg in new:
        start = seg["start"] + offset
        if start < last_end - 0.1:
            continue
        into.append(
            {
                "start": round(start, 3),
                "end": round(seg["end"] + offset, 3),
                "text": seg["text"],
                "words": [
                    {
                        "word": w["word"],
                        "start": round(w["start"] + offset, 3),
                        "end": round(w["end"] + offset, 3),
                        "score": w["score"],
                    }
                    for w in seg["words"]
                ],
            }
        )
        last_end = into[-1]["end"]


def transcribe(
    audio_path: Path,
    language: str | None = None,
    model: str = DEFAULT_MODEL,
    progress=None,
) -> dict:
    """Transcribe with word timestamps. Returns the ASR stage's payload.

    Splits the audio when it cannot be uploaded whole. That is not an edge
    case: the material this pipeline is pointed at — interviews, podcasts,
    streams — routinely runs past two hours, where even lossless
    compression leaves five times the limit.
    """
    emit = progress or (lambda fraction, message: None)
    key = api_key()
    if not key:
        raise HostedAsrError(
            "Hosted transcription needs a Groq API key. Get one at "
            "console.groq.com/keys, then set " + API_KEY_ENV + " (or "
            "groq_api_key in ~/.publikclip/secrets.json)."
        )

    # A file that already fits is sent exactly as it arrived: no ffprobe,
    # no re-encode, no plan. That is the ordinary case and it must stay
    # free of everything below.
    if audio_path.stat().st_size <= UPLOAD_LIMIT_BYTES:
        return _transcribe_spans(audio_path, [(0.0, 0.0)], key, model, language, emit)

    # Otherwise: what matters is the size of what gets uploaded, not of what
    # is on disk. Planning against the WAV split a twenty-minute file into
    # two parts that FLAC fitted into one — twice the requests, two seams,
    # for nothing. So compress once, measure that, and plan from it. The
    # ratio is a property of the material (a live room compresses far worse
    # than a quiet studio) and cannot be assumed.
    duration = _audio_duration(audio_path)
    emit(-1, "Compressing audio for upload…")
    source = _extract_flac(
        audio_path, audio_path.with_suffix(".upload.flac"), 0.0, duration
    )
    try:
        spans = chunk_plan(duration, source.stat().st_size)
        if len(spans) > 1:
            emit(-1, f"Audio is {duration / 60:.0f} min — sending in {len(spans)} parts")
        return _transcribe_spans(source, spans, key, model, language, emit)
    finally:
        source.unlink(missing_ok=True)


def _transcribe_spans(
    source: Path,
    spans: list[tuple[float, float]],
    key: str,
    model: str,
    language: str | None,
    emit,
) -> dict:

    segments: list[dict] = []
    detected: str | None = None
    started = time.monotonic()

    for i, (start, length) in enumerate(spans):
        label = f" (part {i + 1}/{len(spans)})" if len(spans) > 1 else ""
        emit(i / len(spans), f"Transcribing via Groq ({model}){label}…")
        single = len(spans) == 1
        chunk = source if single else source.with_name(
            f"{source.stem}.part{i:02d}.flac"
        )
        try:
            if not single:
                _extract_flac(source, chunk, start, length)
            size = chunk.stat().st_size
            if size > UPLOAD_LIMIT_BYTES:
                raise HostedAsrError(
                    f"A {length / 60:.0f}-minute part still compresses to "
                    f"{size / 1e6:.0f} MB, above the "
                    f"{UPLOAD_LIMIT_BYTES / 1e6:.0f} MB limit. Lower "
                    "CHUNK_SAFETY, or use PUBLIKCLIP_ASR_BACKEND=local."
                )
            payload = _post(chunk, key, model, language)
        finally:
            if chunk is not source:
                chunk.unlink(missing_ok=True)

        detected = detected or payload.get("language")
        chunk_segments = payload.get("segments") or []
        if not chunk_segments:
            continue
        _merge(segments, _attach_words(chunk_segments, payload.get("words") or []), start)

    if not segments:
        raise HostedAsrError(
            "Groq returned no segments — the audio may be silent or unreadable."
        )

    elapsed = time.monotonic() - started
    audio_sec = float(segments[-1]["end"])
    emit(1.0, f"Transcribed {audio_sec / 60:.0f} min in {elapsed:.0f}s")

    return {
        "language": detected or language or "unknown",
        "model": model,
        "backend": "groq",
        "compute_type": "hosted",
        "parts": len(spans),
        "word_count": sum(len(s["words"]) for s in segments),
        "benchmark": {
            "audio_sec": round(audio_sec, 1),
            "transcribe_sec": round(elapsed, 1),
            "align_sec": 0.0,
        },
        "segments": segments,
    }
