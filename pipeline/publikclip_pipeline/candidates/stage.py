"""Candidates stage: build every free channel, weigh them into the interest
curve, extract ~35 sentence-snapped candidate windows. No LLM spend here —
this count is the cost gate for T1/T2."""

from __future__ import annotations

import json
from pathlib import Path

from ..jobs.queue import Stage, StageContext, StageError


def detect_scenes(media_path: str, progress=None) -> list[float]:
    """Scene-change timestamps via PySceneDetect ContentDetector (BSD-3) on
    a downscaled decode. On a static podcast this returns camera cuts; on
    gaming/vlog footage it captures visual pacing."""
    from scenedetect import ContentDetector, open_video
    from scenedetect.scene_manager import SceneManager

    video = open_video(media_path)
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=27.0))
    manager.auto_downscale = True
    manager.detect_scenes(video, show_progress=False)
    return [start.get_seconds() for start, _ in manager.get_scene_list()]


class CandidatesStage(Stage):
    name = "candidates"
    schema_version = 1

    def run(self, ctx: StageContext) -> dict:
        import numpy as np

        from . import curve as curve_mod
        from . import windows as windows_mod

        prior = ctx.prior or {}
        ingest = prior.get("ingest")
        diarize = prior.get("diarize")
        events = prior.get("events")
        if not (ingest and diarize and events):
            raise StageError("Candidates need ingest + diarize + events outputs.")

        segments = diarize["segments"]
        duration = float(ingest["probe"]["duration_sec"])
        n = int(np.ceil(duration))

        curves_path = Path(events["curves_path"])
        if not curves_path.exists():
            raise StageError("curves.json missing — re-run events.")
        curves = json.loads(curves_path.read_text(encoding="utf-8"))

        # Scene detection is deterministic and slow — about ten minutes on a
        # seventy-minute video — and its answer is already on disk from the
        # last run of this stage. The media in a job directory does not
        # change, so re-deriving it costs ten minutes to reach the same list.
        #
        # An empty list is NOT reused: that is what a failed detection also
        # writes, and caching a failure would disable the channel for the
        # life of the job with nothing to say why.
        scenes_path = ctx.job_dir / "scenes.json"
        scene_times: list[float] = []
        if scenes_path.exists():
            try:
                cached = json.loads(scenes_path.read_text(encoding="utf-8"))
                if isinstance(cached, list) and cached:
                    scene_times = [float(t) for t in cached]
                    ctx.emit(-1, f"Scene changes: {len(scene_times)} (cached)")
            except (json.JSONDecodeError, TypeError, ValueError):
                scene_times = []

        if not scene_times:
            ctx.emit(-1, "Detecting scene changes…")
            try:
                scene_times = detect_scenes(ingest["media_path"])
            except Exception:  # noqa: BLE001 — scenes are a minor channel; degrade
                scene_times = []
            scenes_path.write_text(json.dumps(scene_times), encoding="utf-8")

        ctx.emit(0.6, "Building interest curve…")
        channels = {
            "heatmap": curve_mod.heatmap_channel(ingest.get("heatmap"), n),
            "dynamics": curve_mod.dynamics_channel(curves["dynamics"], curves["grid_sec"], n),
            "events": curve_mod.events_channel(events["timeline"], n),
            "turns": curve_mod.turns_channel(diarize["turns"], n),
            "arousal": curve_mod.arousal_channel(
                curves.get("arousal", []), curves.get("arousal_grid_sec", 0.5), n
            ),
            "scenes": curve_mod.scenes_channel(scene_times, n),
            "lexical": curve_mod.lexical_channel(segments, n),
        }
        curve, effective_weights = curve_mod.interest_curve(channels)

        ctx.emit(0.8, "Extracting candidate windows…")
        candidates = windows_mod.extract(
            curve, channels, segments, duration,
            focus=[tuple(s) for s in (getattr(ctx.settings, "focus", None) or [])],
            scene_times=scene_times,
        )
        if not candidates:
            raise StageError(
                "No candidate moments found — the video may be too short or too quiet."
            )

        # Persist the curve for the review UI's timeline visualization.
        (ctx.job_dir / "interest_curve.json").write_text(
            json.dumps({"per_sec": np.round(curve, 4).tolist()})
        , encoding="utf-8")

        return {
            "candidates": [c.to_json() for c in candidates],
            "count": len(candidates),
            "effective_weights": effective_weights,
            "scene_count": len(scene_times),
            "heatmap_present": bool(ingest.get("heatmap")),
        }
