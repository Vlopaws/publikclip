"""Render stage: finalist clips + trajectories + captions → finished 9:16
MP4s, each verified (streams present, duration sane) before being reported."""

from __future__ import annotations

import json
from pathlib import Path

from ..jobs.queue import Stage, StageContext, StageError


class RenderStage(Stage):
    name = "render"
    # 2: framing mode (vertical / wide) and the title band it implies.
    schema_version = 2

    def artifacts_ok(self, ctx: StageContext, data: dict) -> bool:
        if data.get("caption_preset") != ctx.settings.caption_preset:
            return False  # restyle requested → re-render
        return all(Path(c["path"]).exists() for c in data.get("outputs", []))

    def run(self, ctx: StageContext) -> dict:
        import numpy as np

        from ..captions import ass as ass_mod
        from ..scoring import llm as llm_mod
        from ..scoring import rubric
        from ..autopilot import parts as parts_mod
        from ..sources import naming
        from . import ffmpeg_bin, renderer

        if not ffmpeg_bin.supports_captions():
            ctx.emit(-1, "No caption-capable ffmpeg found — fetching one…")
            if not ffmpeg_bin.ensure_capable(progress=lambda f, m: ctx.emit(f, m)):
                ctx.emit(-1, "Caption burning unavailable — rendering without captions.")

        prior = ctx.prior or {}
        ingest = prior.get("ingest")
        diarize = prior.get("diarize")
        events = prior.get("events")
        score = prior.get("score")
        camera = prior.get("camera")
        if not (ingest and diarize and events and score and camera):
            raise StageError("Render needs every prior stage output.")

        media = ingest["media_path"]
        probe = ingest["probe"]
        src_w, src_h = int(probe["width"]), int(probe["height"])
        segments = diarize["segments"]
        timeline = events["timeline"]
        curves = json.loads(Path(events["curves_path"]).read_text(encoding="utf-8"))
        rms = curves["rms"]
        grid = float(curves["grid_sec"])

        captions_ok = ffmpeg_bin.supports_captions()
        emoji_ok = ass_mod.emoji_probe() if captions_ok else False
        ctx.emit(-1, f"Emoji support: {'yes' if emoji_ok else 'no (dropping emoji)'}")

        # Copy is written per finalist, not per candidate — see
        # rubric.headline_prompt. A failure here must never lose a clip: a
        # video without a title is a video, a raised exception is nothing.
        headline_client = None
        try:
            headline_client = llm_mod.make_client(ctx.settings.llm_mode)
        except llm_mod.LlmError as err:
            ctx.emit(-1, f"No titles this run ({err})")

        # Clips cut back to back are one moment in two pieces, and a viewer
        # who meets the second one first cannot tell. Worked out here
        # because it is a fact about the cut, not something to ask a model.
        part_of = parts_mod.group([(c["start"], c["end"]) for c in score["clips"]])

        out_dir = ctx.job_dir / "clips"
        out_dir.mkdir(exist_ok=True)
        preset = ctx.settings.caption_preset
        outputs = []
        clips = score["clips"]
        for i, clip in enumerate(clips):
            traj_path = camera["trajectories"].get(str(i))
            if not traj_path or not Path(traj_path).exists():
                continue
            trajectory = json.loads(Path(traj_path).read_text(encoding="utf-8"))
            start, end = clip["start"], clip["end"]
            ctx.emit(i / max(1, len(clips)), f"Rendering clip {i + 1}/{len(clips)}…")

            # Words within the clip, clip-relative times.
            words = []
            for seg in segments:
                for w in seg.get("words", []):
                    if start <= w["start"] < end:
                        words.append(
                            ass_mod.Word(
                                text=w["word"],
                                start=round(w["start"] - start, 3),
                                end=round(min(w["end"], end) - start, 3),
                            )
                        )
            ass_mod.mark_emphasis(words, rms, grid, clip_start=start)
            clip_events = [
                {
                    "type": e["type"],
                    "start": round(max(0.0, e["start"] - start), 3),
                    "end": round(min(e["end"], end) - start, 3),
                }
                for e in timeline
                if e["end"] > start and e["start"] < end and e["type"] != "pause"
            ]
            framing = trajectory.get("framing") or {}
            mode = framing.get("mode", "vertical")
            band = framing.get("title_band")
            band = tuple(band) if band else None

            copy = {}
            spoken = " ".join(w.text for w in words)
            # Who the page says is in the video, narrowed to who this clip's
            # own words name. The second list is what a title may claim.
            cast = list(ingest.get("cast") or [])
            named = naming.mentioned_in(spoken, cast)
            if headline_client is not None and band:
                try:
                    copy = headline_client.generate_json(
                        rubric.headline_prompt(
                            spoken,
                            {
                                "duration": end - start,
                                "cast": cast,
                                "named": named,
                            },
                        ),
                        rubric.HEADLINE_SCHEMA,
                    )
                except Exception as err:  # noqa: BLE001 — copy is optional
                    ctx.emit(-1, f"clip {i + 1}: no title ({err})")

            title = (copy.get("title") or "").strip() or None
            if title and cast:
                # The prompt forbids guessing; this checks it was obeyed.
                # A title naming the wrong cast member is a false statement
                # about a real person, published — so it is dropped rather
                # than shipped, and the clip goes out without a headline.
                kept = naming.strip_unsupported(title, named, cast)
                if not kept:
                    ctx.emit(
                        -1,
                        f"clip {i + 1}: title dropped, it named someone this "
                        "clip does not",
                    )
                title = kept or None
            title = parts_mod.label(title, part_of.get(i)) if title else title
            ass_path = out_dir / f"clip_{i:02d}.ass"
            ass_path.write_text(
                ass_mod.build_ass(
                    words, clip_events, preset_name=preset, emoji_ok=emoji_ok,
                    title=title, title_band=band, clip_duration=end - start,
                    # A wide clip's band is an empty bar; a vertical clip's
                    # band is borrowed picture. Only the first can afford to
                    # hold the title for the whole runtime.
                    hold_whole_clip=(mode == "wide"),
                )
            , encoding="utf-8")

            out_path = out_dir / f"clip_{i:02d}.mp4"
            try:
                renderer.render_clip(
                    media, out_path, start, end, trajectory,
                    ass_path if captions_ok else None, ass_mod.FONTS_DIR,
                    lufs=ctx.settings.lufs_target,
                    true_peak=ctx.settings.true_peak_db,
                    src_w=src_w, src_h=src_h, mode=mode,
                )
            except RuntimeError as err:
                raise StageError(str(err)) from err
            check = renderer.verify_output(out_path, end - start)
            if not check["ok"]:
                raise StageError(
                    f"Clip {i} failed verification (duration {check['duration']:.1f}s, "
                    f"{check['width']}x{check['height']})."
                )
            outputs.append(
                {
                    "clip": i,
                    "path": str(out_path),
                    "ass": str(ass_path),
                    "score": clip["score"],
                    "best_platform": clip["best_platform"],
                    "duration": round(check["duration"], 2),
                    "words": len(words),
                    "event_tags": len(clip_events),
                    "mode": mode,
                    "title": title,
                    "part": list(part_of[i]) if i in part_of else None,
                    "cast": cast,
                    "named": named,
                    # Kept so publishing can judge the clip without
                    # re-deriving it from the whole job.
                    "transcript": spoken[:2000],
                    "description": (copy.get("description") or "").strip() or None,
                    "hashtags": copy.get("hashtags") or [],
                }
            )

        if not outputs:
            raise StageError("No clips were rendered.")
        return {
            "outputs": outputs,
            "emoji_ok": emoji_ok,
            "captions_burned": captions_ok,
            "caption_preset": preset,
        }
