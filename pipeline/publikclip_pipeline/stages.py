"""The processing pipeline, in order.

Lives here rather than in the CLI because the CLI is no longer the only
caller — the autopilot runs the same stages unattended, and a second copy of
this list is a second thing to forget to update.

Stage imports stay deferred inside the function so `publikclip jobs` and
`publikclip sources` do not pay the torch import tax to list a directory.
"""

from __future__ import annotations

from .jobs import queue


def default_stages() -> list[queue.Stage]:
    """ingest → asr → diarize → events → candidates → score → camera → render."""
    from .asr.stage import AsrStage
    from .camera.stage import CameraStage
    from .candidates.stage import CandidatesStage
    from .diarize.stage import DiarizeStage
    from .events.stage import EventsStage
    from .ingest.stage import IngestStage
    from .render.stage import RenderStage
    from .scoring.stage import ScoreStage

    return [
        IngestStage(),
        AsrStage(),
        DiarizeStage(),
        EventsStage(),
        CandidatesStage(),
        ScoreStage(),
        CameraStage(),
        RenderStage(),
    ]
