"""publikclip CLI.

Doubles as the desktop app's sidecar: with --jsonl every progress event and
the final result are emitted as one JSON object per stdout line, so the
Tauri shell just spawns `publikclip --jsonl run <source>` and streams.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config
from .autopilot import select as select_mod
from .jobs import prune as prune_mod
from .jobs import queue
# Cheap import (httpx + config only) — safe at module level next to the
# deferred stage imports below, and argparse needs the mode list at
# parser-build time.
from .scoring import llm as llm_mod


def _stages() -> list[queue.Stage]:
    # The autopilot runs the same list; it lives in one place so a new stage
    # cannot be added to the CLI and forgotten for unattended runs.
    from .stages import default_stages

    return default_stages()


def _progress_printer(jsonl: bool):
    def emit(stage: str, fraction: float, message: str) -> None:
        if jsonl:
            print(
                json.dumps(
                    {"event": "progress", "stage": stage, "fraction": fraction, "message": message}
                ),
                flush=True,
            )
        else:
            pct = f"{fraction * 100:5.1f}%" if fraction >= 0 else "  ...."
            print(f"[{stage:<10}] {pct} {message}", file=sys.stderr, flush=True)

    return emit


def _emit_result(jsonl: bool, payload: dict) -> None:
    if jsonl:
        print(json.dumps({"event": "result", **payload}), flush=True)
    else:
        print(json.dumps(payload, indent=2))


def cmd_run(args: argparse.Namespace) -> int:
    source = args.source
    source_type = "url" if source.startswith(("http://", "https://")) else "file"
    settings = config.Settings()
    if args.llm:
        settings.llm_mode = args.llm
    if args.captions:
        settings.caption_preset = args.captions
    if args.camera:
        settings.camera.speaker_change = args.camera
    job = queue.create_job(source_type, source, json.dumps(settings.to_json()))
    return _execute(job, args.jsonl)


def cmd_resume(args: argparse.Namespace) -> int:
    job = queue.get_job(args.job_id)
    if job is None:
        print(f"No job {args.job_id}", file=sys.stderr)
        return 2
    if args.llm or args.captions or args.camera:
        settings = config.Settings.from_json(json.loads(job.settings_json))
        if args.llm:
            settings.llm_mode = args.llm
        if args.captions:
            settings.caption_preset = args.captions
        if args.camera:
            settings.camera.speaker_change = args.camera
        new_json = json.dumps(settings.to_json())
        with queue._connect() as conn:  # noqa: SLF001 — CLI is a queue friend
            conn.execute("UPDATE jobs SET settings_json = ? WHERE id = ?", (new_json, job.id))
        job = queue.get_job(args.job_id)
    return _execute(job, args.jsonl)


def _execute(job: queue.Job, jsonl: bool) -> int:
    emit = _progress_printer(jsonl)
    if jsonl:
        print(json.dumps({"event": "job", "job_id": job.id, "dir": str(job.dir)}), flush=True)
    else:
        print(f"job {job.id} → {job.dir}", file=sys.stderr)
    try:
        results = queue.run_stages(job, _stages(), emit)
    except queue.StageError as err:
        _emit_result(jsonl, {"ok": False, "job_id": job.id, "error": str(err)})
        return 1
    summary = {
        "ok": True,
        "job_id": job.id,
        "stages": list(results.keys()),
        "title": results.get("ingest", {}).get("title"),
        "heatmap_segments": len(results.get("ingest", {}).get("heatmap") or []),
    }
    _emit_result(jsonl, summary)
    return 0


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def cmd_jobs(args: argparse.Namespace) -> int:
    if getattr(args, "jobs_cmd", None) == "prune":
        report = prune_mod.plan(
            min_age_days=args.older_than, job_id=args.job_id
        )
        if args.apply and report.jobs:
            prune_mod.apply(report)
        if args.json:
            print(json.dumps(report.to_json(), indent=2, ensure_ascii=False))
            return 0
        for entry in report.jobs:
            print(
                f"{entry.job_id}  {_human(entry.bytes_freed):>9}  "
                f"{entry.status:<8} {entry.age_days:.0f}d  {(entry.title or '')[:44]}"
            )
        verb = "freed" if report.applied else "would free"
        print(f"\n{verb} {_human(report.bytes_freed)} from {len(report.jobs)} job(s)")
        if not report.applied and report.jobs:
            print("re-run with --apply to delete")
        for note in report.skipped[:5]:
            print(f"  skipped {note}")
        return 0

    for job in queue.list_jobs():
        stages = queue.stage_statuses(job.id)
        done = sum(1 for s in stages.values() if s == "done")
        print(f"{job.id}  {job.status:<8} {done} stage(s) done  {job.title or job.source}")
    return 0


def cmd_auto(args: argparse.Namespace) -> int:
    """Discover, clip and (optionally) publish, without a person in the loop."""
    from . import autopilot
    from .sources import twitch, youtube

    # Argument validation first. A mistyped platform name is free to catch
    # here and costs a channel listing to catch after — and the listing is
    # the cheap end of what follows it.
    platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]
    try:
        publisher = autopilot.make_publisher(args.publish, visibility=args.visibility)
        publisher.check_ready(platforms)
    except autopilot.PublishError as err:
        print(str(err), file=sys.stderr)
        return 1

    if args.youtube:
        candidates = youtube.recent_uploads(
            args.youtube, limit=args.limit, progress=_stderr_progress
        )
    elif args.twitch_roster:
        from .sources import roster as roster_mod

        names = roster_mod.load(args.twitch_roster)
        if not names:
            print(f"No channels in {args.twitch_roster}", file=sys.stderr)
            return 1
        note_ = _stderr_progress
        sweep = roster_mod.sweep(
            names, per_channel=args.per_channel, progress=note_
        )
        if sweep.failed:
            print(
                f"{len(sweep.failed)} channel(s) could not be listed: "
                f"{', '.join(sorted(sweep.failed)[:8])}",
                file=sys.stderr,
            )
        # The pool is already ranked by view count; --limit takes the head.
        candidates = sweep.items[: args.limit]
        print(
            f"{len(sweep.items)} clip(s) across {len(sweep.reached)} channel(s); "
            f"taking the top {len(candidates)}",
            file=sys.stderr,
        )
    else:
        candidates = twitch.channel_clips(
            args.twitch, limit=args.limit, progress=_stderr_progress
        )

    def note(kind: str, message: str) -> None:
        if args.jsonl:
            print(json.dumps({"event": "auto", "kind": kind, "message": message}), flush=True)
        else:
            print(message, file=sys.stderr, flush=True)

    try:
        report = autopilot.run(
            candidates,
            publisher=publisher,
            platforms=platforms,
            clips_per_video=args.clips,
            min_score=args.min_score,
            llm_mode=args.llm or "ollama",
            captions=args.captions,
            skip_seen=not args.include_seen,
            on_event=note,
        )
    except autopilot.PublishError as err:
        print(str(err), file=sys.stderr)
        return 1

    if args.jsonl:
        print(json.dumps({"event": "result", **report.to_json()}), flush=True)
    else:
        print(json.dumps(report.to_json(), indent=2))
    return 0 if report.failures == 0 else 1


def cmd_publish(args: argparse.Namespace) -> int:
    """Publish the clips of a job that has already been rendered.

    `run` produces clips and `auto` publishes them, and until now there was
    no way between the two: a job cut by hand could not be posted at all
    without re-running discovery over it. This is the same selection and the
    same ledger the autopilot uses, pointed at one job.
    """
    from . import autopilot
    from .autopilot import publish as publish_mod
    from .autopilot.select import select

    job = queue.get_job(args.job_id)
    job_dir = Path(args.dir) if args.dir else (job.dir if job else None)
    if job_dir is None or not job_dir.exists():
        print(
            f"No job {args.job_id} here. Pass --dir if its directory is "
            "somewhere else.",
            file=sys.stderr,
        )
        return 2

    platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]
    try:
        publisher = autopilot.make_publisher(args.publish, visibility=args.visibility)
        publisher.check_ready(platforms)
    except autopilot.PublishError as err:
        print(str(err), file=sys.stderr)
        return 1

    picked = select(
        args.job_id, job_dir, take=args.clips, min_score=args.min_score
    )
    if not picked:
        print(
            f"Nothing in {args.job_id} scored at or above {args.min_score}.",
            file=sys.stderr,
        )
        return 0

    results = []
    for clip in picked:
        for platform in platforms:
            if publish_mod.already_posted(clip, platform):
                print(f"  clip {clip.clip} already on {platform}", file=sys.stderr)
                continue
            result = publisher.publish(clip, platform)
            publish_mod.record(result)
            results.append(result)
            verb = (
                "would post" if result.dry_run
                else ("posted" if result.ok else "FAILED")
            )
            print(
                f"  {verb} clip {clip.clip} ({clip.score:.1f}) -> {platform}"
                f"{'' if result.ok else ': ' + str(result.error)[:160]}",
                file=sys.stderr,
            )

    payload = {
        "job_id": args.job_id,
        "selected": [c.to_json() for c in picked],
        "published": [r.to_json() for r in results],
        "failures": sum(1 for r in results if not r.ok),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 1 if payload["failures"] else 0


def cmd_sources(args: argparse.Namespace) -> int:
    """Discovery only — list what could be clipped, download nothing.

    Kept separate from `run` on purpose: deciding what is worth an hour of
    compute is a different question from doing the hour of compute, and
    seeing the list first is what makes automation reviewable.
    """
    from .sources import twitch, youtube
    from .sources import common as sources_common

    # The help promises "0 disables"; without this it silently means "reject
    # everything longer than zero seconds", which filters out every result.
    # Only the listing subcommands carry these — `scan` measures a creator
    # rather than filtering a list, and reading them unconditionally made it
    # fail before it started.
    def _bound(value):
        return None if not value else value

    for bound in ("min_duration", "max_duration"):
        if hasattr(args, bound):
            setattr(args, bound, _bound(getattr(args, bound)))

    if args.source_cmd == "roster":
        from .sources import roster as roster_mod

        names = roster_mod.load(args.file)
        emit = None if args.json else _stderr_progress
        sweep = roster_mod.sweep(names, per_channel=args.per_channel, progress=emit)
        if args.json:
            print(json.dumps(sweep.to_json(), indent=2, ensure_ascii=False))
            return 0
        print(
            f"{len(sweep.items)} clip(s) from {len(sweep.reached)}/{sweep.channels} "
            "channel(s), ranked by views"
        )
        for item in sweep.items[: args.top]:
            print(f"  {item.url}\t{(item.channel or '?')[:18]:<18} {item.summary()}")
        for name, why in sorted(sweep.failed.items())[:8]:
            print(f"  unreachable {name}: {why[:70]}")
        return 0

    if args.source_cmd == "probe":
        from .sources import clippability

        emit = None if args.json else _stderr_progress
        report = clippability.assess(
            args.channel, videos=args.videos, progress=emit
        )
        if args.json:
            print(json.dumps(report.to_json(), indent=2, ensure_ascii=False))
            return 0
        print(f"{report.creator}: {report.verdict} — {report.advice}")
        print(
            f"  faces in {report.face_coverage:.0%} of sampled frames, "
            f"{report.face_height:.0%} of frame height, "
            f"{report.vertical_share:.0%} would cut vertical"
        )
        for sample in report.samples:
            if sample.error:
                print(f"  - {sample.title[:56]}: {sample.error}")
            else:
                print(
                    f"  - {sample.title[:56]}: {sample.face_coverage:.0%} faces, "
                    f"{sample.mode}"
                )
        return 0

    if args.source_cmd == "scan":
        from .sources import opportunity

        emit = None if args.json else _stderr_progress
        if args.no_audience:
            report = opportunity.clip_saturation(
                args.creator, per_query=args.per_query, progress=emit
            )
            payload, headline = report.to_json(), report.verdict
            saturation = report
            demand = None
        else:
            assessment = opportunity.assess(
                args.creator, channel=args.channel, per_query=args.per_query, progress=emit
            )
            payload, headline = assessment.to_json(), assessment.verdict
            saturation = assessment.saturation
            demand = assessment

        if args.json:
            print(json.dumps(payload))
            return 0

        print(f"{args.creator}: {headline.upper()}")
        if demand is not None:
            views = f"{demand.median_views:,}".replace(",", " ") if demand.median_views else "?"
            print(f"  demand   median {views} views over {demand.uploads_seen} recent uploads"
                  f"  (measured on: {demand.measured_channel or 'unresolved'})")
        print(f"  supply   {saturation.dedicated_channels} dedicated clip channel(s), "
              f"best clip {saturation.top_clip_views:,} views".replace(",", " "))
        for channel in saturation.clip_channels:
            print(f"    {channel.name[:32]:32s} {channel.videos_found:>2} found  "
                  f"{channel.top_views:>9,} top".replace(",", " ") + f"  {channel.top_title[:36]}")
        if not saturation.clip_channels:
            print("    none surfaced — heuristic over search results, not a census")
        return 0

    if args.source_cmd == "youtube":
        items = youtube.recent_uploads(
            args.channel,
            limit=args.limit,
            min_duration_sec=args.min_duration,
            max_duration_sec=args.max_duration,
            progress=None if args.json else _stderr_progress,
        )
        window = "latest uploads"
    elif args.source_cmd == "twitch":
        if args.category:
            items = twitch.category_clips(
                args.category, limit=args.limit, days=args.days,
                min_duration_sec=args.min_duration, max_duration_sec=args.max_duration,
            )
            window = f"top clips, last {args.days} d"
        else:
            items = twitch.channel_clips(
                args.channel, limit=args.limit,
                min_duration_sec=args.min_duration, max_duration_sec=args.max_duration,
                progress=None if args.json else _stderr_progress,
            )
            # Stated, not guessed: yt-dlp ignores the period filter.
            window = twitch.CHANNEL_CLIP_WINDOW
    else:  # pragma: no cover - argparse enforces the choices
        raise SystemExit(f"unknown source {args.source_cmd}")

    if args.new_only:
        items = sources_common.unseen(items)

    if args.json:
        print(json.dumps({"window": window, "items": [i.to_json() for i in items]}))
        return 0

    if not items:
        print("nothing matched (try --min-duration / --max-duration)", file=sys.stderr)
        return 0
    print(f"{len(items)} candidate(s) — {window}", file=sys.stderr)
    for item in items:
        print(f"{item.url}\t{item.summary()}")
    return 0


def _stderr_progress(fraction: float, message: str) -> None:
    if fraction == -1:
        print(message, file=sys.stderr)


def cmd_edit(args: argparse.Namespace) -> int:
    """Per-clip editing verbs. All output is JSON on stdout for the app."""
    from pathlib import Path

    from .edits import render_clip as rc
    from .edits import store, visuals

    job = queue.get_job(args.job_id)
    if job is None:
        print(json.dumps({"ok": False, "error": f"no job {args.job_id}"}))
        return 2
    job_dir = Path(job.dir)

    if args.edit_cmd == "context":
        print(json.dumps({"ok": True, **rc.context_for_clip(job_dir, args.clip)}))
        return 0

    if args.edit_cmd == "suggest-visuals":
        score = json.loads((job_dir / "score.json").read_text(encoding="utf-8"))["data"]
        clip = score["clips"][args.clip]
        edit = store.edit_for_clip(job_dir, args.clip, clip)
        # plan against OUTPUT-time words = current bounds without dead-space
        # (suggestions land on the source-bounds timeline the UI shows)
        diarize = json.loads((job_dir / "diarize.json").read_text(encoding="utf-8"))["data"]
        words = [
            {"word": w["word"], "start": w["start"] - edit.start, "end": w["end"] - edit.start}
            for seg in diarize["segments"]
            for w in seg.get("words", [])
            if edit.start <= w["start"] < edit.end
        ]
        settings = config.Settings.from_json(json.loads(job.settings_json))
        try:
            suggestions = visuals.suggest(job_dir, words, settings.llm_mode, prefer=args.prefer)
        except Exception as err:  # noqa: BLE001 — surface, don't crash the app
            print(json.dumps({"ok": False, "error": str(err)}))
            return 1
        edits = store.load(job_dir)
        current = edits.get(str(args.clip), edit)
        known = {o.id for o in current.overlays}
        current.overlays.extend(o for o in suggestions if o.id not in known)
        edits[str(args.clip)] = current
        store.save(job_dir, edits)
        print(json.dumps({"ok": True, "edit": current.to_json()}))
        return 0

    if args.edit_cmd == "render-clip":
        emit = _progress_printer(args.jsonl)
        try:
            entry = rc.render_clip_edit(job_dir, args.clip, lambda f, m: emit("render", f, m))
        except Exception as err:  # noqa: BLE001
            _emit_result(args.jsonl, {"ok": False, "error": str(err)})
            return 1
        _emit_result(args.jsonl, {"ok": True, "output": entry})
        return 0
    return 2


def cmd_ig(args: argparse.Namespace) -> int:
    from .insights import calibration, instagram

    if args.ig_cmd == "connect":
        conn = instagram.connect(args.app_id, args.app_secret)
        print(f"Connected as @{conn['username']} (user {conn['user_id']}).")
        return 0

    # App-facing commands: exactly one JSON line on stdout (the shell's
    # ig_tool parses the last JSON line, same contract as edit_tool).
    if args.ig_cmd == "sync":
        summary = calibration.sync()
        print(json.dumps(summary))
        return 0 if summary.get("ok") else 1

    if args.ig_cmd == "overview":
        print(json.dumps(calibration.overview()))
        return 0

    if args.ig_cmd == "link":
        job = queue.get_job(args.job_id)
        if job is None:
            print(json.dumps({"ok": False, "error": f"no job {args.job_id}"}))
            return 2
        score_data = queue.read_checkpoint(job, "score", 1)
        if not score_data:
            print(json.dumps({"ok": False, "error": "job has no score checkpoint"}))
            return 2
        clips = score_data["clips"]
        if not 0 <= args.clip < len(clips):
            print(json.dumps({"ok": False, "error": f"clip index out of range (0..{len(clips) - 1})"}))
            return 2
        calibration.link_clip(
            args.job_id, args.clip, args.media_id, clips[args.clip],
            link_source=args.source,
            config_version=score_data.get("scoring_config_version", 1),
        )
        print(json.dumps({"ok": True, "linked": {"job_id": args.job_id, "clip": args.clip, "media_id": args.media_id}}))
        return 0

    if args.ig_cmd == "unlink":
        removed = calibration.unlink(args.media_id)
        print(json.dumps({"ok": True, "removed": removed}))
        return 0

    if args.ig_cmd == "reject":
        calibration.reject_match(args.media_id, args.job_id, args.clip)
        print(json.dumps({"ok": True}))
        return 0

    # Human/legacy commands.
    conn = instagram.load_connection()
    if args.ig_cmd in ("media", "pull") and conn is None:
        print("Not connected. Run: publikclip ig connect --app-id ... --app-secret ...", file=sys.stderr)
        return 2
    if conn is not None:
        conn = instagram.refresh_if_needed(conn)

    if args.ig_cmd == "media":
        for m in instagram.recent_media(conn):
            if m.get("media_product_type") == "REELS" or m.get("media_type") == "VIDEO":
                caption = (m.get("caption") or "")[:60].replace("\n", " ")
                print(f"{m['id']}  {m.get('timestamp', '')[:10]}  {caption}")
        return 0

    if args.ig_cmd == "pull":
        rows = calibration.tracked()
        if not rows:
            print("No linked clips yet. Post an exported clip, then: publikclip ig link ...")
            return 0
        for row in rows:
            if not row["ig_media_id"]:
                continue
            try:
                metrics = instagram.media_insights(conn, row["ig_media_id"])
            except instagram.IgError as err:
                print(f"{row['ig_media_id']}: {err}", file=sys.stderr)
                continue
            calibration.store_metrics(row["ig_media_id"], metrics)
            views = metrics.get("views")
            watch = metrics.get("ig_reels_avg_watch_time")
            print(
                f"{row['ig_media_id']}  score {row['score']:.0f} → views {views}, "
                f"avg watch {round(watch / 1000, 1) if watch else '?'}s"
            )
        return 0

    if args.ig_cmd == "report":
        print(json.dumps(calibration.report(args.metric), indent=2))
        return 0
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="publikclip")
    parser.add_argument("--jsonl", action="store_true", help="machine-readable progress on stdout")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="process a YouTube URL or local video file")
    p_run.add_argument("source")
    p_run.add_argument("--llm", choices=llm_mod.available_modes(), default=None)
    p_run.add_argument("--captions", default=None, help="caption preset name")
    p_run.add_argument("--camera", choices=["cut", "pan", "locked"], default=None)
    p_run.set_defaults(fn=cmd_run)

    p_resume = sub.add_parser("resume", help="resume a job from its checkpoints")
    p_resume.add_argument("job_id")
    p_resume.add_argument("--llm", choices=llm_mod.available_modes(), default=None)
    p_resume.add_argument("--captions", default=None, help="caption preset name")
    p_resume.add_argument("--camera", choices=["cut", "pan", "locked"], default=None)
    p_resume.set_defaults(fn=cmd_resume)

    p_jobs = sub.add_parser("jobs", help="list jobs")
    jobs_sub = p_jobs.add_subparsers(dest="jobs_cmd")
    p_prune = jobs_sub.add_parser(
        "prune", help="reclaim source video and analysis audio from old jobs"
    )
    p_prune.add_argument("job_id", nargs="?", help="one job, instead of a sweep")
    p_prune.add_argument(
        "--older-than", type=float, default=prune_mod.DEFAULT_MIN_AGE_DAYS,
        metavar="DAYS", help="leave jobs younger than this alone",
    )
    p_prune.add_argument(
        "--apply", action="store_true",
        help="actually delete; without it, only report what would go",
    )
    p_prune.add_argument("--json", action="store_true")
    p_jobs.set_defaults(fn=cmd_jobs)

    p_edit = sub.add_parser("edit", help="per-clip editing (context / visuals / render)")
    edit_sub = p_edit.add_subparsers(dest="edit_cmd", required=True)
    p_ctx = edit_sub.add_parser("context")
    p_ctx.add_argument("job_id")
    p_ctx.add_argument("clip", type=int)
    p_sv = edit_sub.add_parser("suggest-visuals")
    p_sv.add_argument("job_id")
    p_sv.add_argument("clip", type=int)
    p_sv.add_argument("--prefer", choices=["pexels", "gemini"], default="pexels")
    p_rcl = edit_sub.add_parser("render-clip")
    p_rcl.add_argument("job_id")
    p_rcl.add_argument("clip", type=int)
    p_edit.set_defaults(fn=cmd_edit)

    p_auto = sub.add_parser("auto", help="discover → clip → publish, unattended")
    auto_src = p_auto.add_mutually_exclusive_group(required=True)
    auto_src.add_argument("--youtube", metavar="CHANNEL", help="@handle, channel id, or URL")
    auto_src.add_argument("--twitch", metavar="CHANNEL", help="Twitch channel name")
    auto_src.add_argument(
        "--twitch-roster", metavar="FILE",
        help="file of Twitch channels; their clips are pooled and ranked together",
    )
    p_auto.add_argument(
        "--per-channel", type=int, default=5,
        help="clips to consider from each rostered channel before ranking",
    )
    p_auto.add_argument("--limit", type=int, default=1, help="how many sources to process")
    p_auto.add_argument("--clips", type=int, default=3, help="clips kept per source")
    p_auto.add_argument(
        "--min-score", type=float, default=40.0,
        help="composite floor on the 0-100 scale; 0 disables it",
    )
    p_auto.add_argument(
        "--publish", default="dry-run", choices=["dry-run", "zernio", "composio", "postiz"],
        help="dry-run posts nothing and reports what would go out (default)",
    )
    p_auto.add_argument(
        "--platforms", default="instagram",
        help="comma-separated: instagram, tiktok, youtube",
    )
    p_auto.add_argument(
        "--visibility", default="private", choices=["private", "unlisted", "public"],
        help="private by default; instagram has no private mode and will refuse it",
    )
    p_auto.add_argument("--llm", choices=llm_mod.available_modes(), default=None)
    p_auto.add_argument("--captions", default=None)
    p_auto.add_argument(
        "--include-seen", action="store_true",
        help="re-process sources the job queue already has",
    )
    p_auto.add_argument("--jsonl", action="store_true")
    p_auto.set_defaults(fn=cmd_auto)

    p_pub = sub.add_parser(
        "publish", help="post the clips of a job that is already rendered"
    )
    p_pub.add_argument("job_id")
    p_pub.add_argument("--dir", help="the job directory, if it is not the local one")
    p_pub.add_argument("--clips", type=int, default=3)
    p_pub.add_argument("--min-score", type=float, default=select_mod.DEFAULT_MIN_SCORE)
    p_pub.add_argument("--platforms", default="tiktok")
    p_pub.add_argument(
        "--publish", choices=["dry-run", "zernio", "composio", "postiz"],
        default="dry-run",
    )
    p_pub.add_argument(
        "--visibility", choices=["private", "unlisted", "public"], default="private"
    )
    p_pub.set_defaults(fn=cmd_publish)

    p_sources = sub.add_parser("sources", help="discover what to clip (downloads nothing)")
    src_sub = p_sources.add_subparsers(dest="source_cmd", required=True)

    p_yt = src_sub.add_parser("youtube", help="a channel's recent uploads (no API key)")
    p_yt.add_argument("channel", help="@handle, channel id, or a youtube.com URL")
    p_yt.add_argument("--limit", type=int, default=10)
    p_yt.add_argument("--min-duration", type=float, default=120.0, help="seconds; 0 disables")
    p_yt.add_argument("--max-duration", type=float, default=4 * 3600.0, help="seconds; 0 disables")

    p_tw = src_sub.add_parser("twitch", help="clips from a channel (no key) or a category (needs one)")
    p_tw.add_argument("channel", nargs="?", help="channel name; omit when using --category")
    p_tw.add_argument("--category", help="Twitch category name, e.g. 'Just Chatting' (needs API credentials)")
    p_tw.add_argument("--days", type=int, default=7, help="category mode only")
    p_tw.add_argument("--limit", type=int, default=10)
    p_tw.add_argument("--min-duration", type=float, default=20.0, help="seconds; 0 disables")
    p_tw.add_argument("--max-duration", type=float, default=600.0, help="seconds; 0 disables")

    p_probe = src_sub.add_parser(
        "probe",
        help="is a creator's material face-driven enough to clip? (samples video)",
    )
    p_probe.add_argument("channel", help="@handle, channel id, or URL")
    p_probe.add_argument(
        "--videos", type=int, default=2,
        help="how many recent uploads to sample (a minute of each)",
    )

    p_roster = src_sub.add_parser(
        "roster", help="pool the clips of many Twitch channels and rank them"
    )
    p_roster.add_argument("file", help="one channel per line; # comments allowed")
    p_roster.add_argument("--per-channel", type=int, default=5)
    p_roster.add_argument("--top", type=int, default=20, help="how many to print")

    p_scan = src_sub.add_parser(
        "scan", help="how crowded the clip scene around a creator looks (heuristic)"
    )
    p_scan.add_argument("creator", help="creator name as an audience would search it")
    p_scan.add_argument("--per-query", type=int, default=12)
    p_scan.add_argument("--channel", help="their YouTube channel, if the name does not resolve to it")
    p_scan.add_argument(
        "--no-audience", action="store_true",
        help="skip the reach measurement (faster, but 'open' stops being meaningful)",
    )

    for parser_ in (p_yt, p_tw, p_scan, p_probe, p_roster):
        parser_.add_argument("--new-only", action="store_true", help="drop what the job queue already has")
        parser_.add_argument("--json", action="store_true")
    p_sources.set_defaults(fn=cmd_sources)

    p_ig = sub.add_parser("ig", help="Instagram feedback loop (your own Meta app)")
    ig_sub = p_ig.add_subparsers(dest="ig_cmd", required=True)
    p_connect = ig_sub.add_parser("connect", help="OAuth against your own Meta app")
    p_connect.add_argument("--app-id", required=True)
    p_connect.add_argument("--app-secret", required=True)
    ig_sub.add_parser("sync", help="one sync pass: media + thumbnails + insights ladder + auto-fit (JSON)")
    ig_sub.add_parser("overview", help="everything the Loop screen renders (JSON)")
    ig_sub.add_parser("media", help="list your recent Reels to link against")
    p_link = ig_sub.add_parser("link", help="link a rendered clip to a posted Reel (JSON)")
    p_link.add_argument("job_id")
    p_link.add_argument("clip", type=int)
    p_link.add_argument("media_id")
    p_link.add_argument("--source", default="manual", choices=["manual", "match_confirmed"])
    p_unlink = ig_sub.add_parser("unlink", help="remove a clip↔Reel link (JSON)")
    p_unlink.add_argument("media_id")
    p_reject = ig_sub.add_parser("reject", help="'not this' — never suggest this pair again (JSON)")
    p_reject.add_argument("media_id")
    p_reject.add_argument("job_id")
    p_reject.add_argument("clip", type=int)
    ig_sub.add_parser("pull", help="fetch metrics for every linked clip")
    p_report = ig_sub.add_parser("report", help="score-vs-outcome calibration report")
    p_report.add_argument("--metric", default="views")
    p_ig.set_defaults(fn=cmd_ig)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
