"""Clip renderer: one ffmpeg filter_complex per clip.

The sendcmd architecture (vendored from mutonby/openshorts punch_in.py +
reframe_v2.py, MIT): the director's per-frame trajectory array becomes a
deduped sendcmd command file driving a labeled crop filter — hard cuts are
just discontinuities in the same array, pans are smooth regions, punch-ins
already live in the w/h values. One decode, one encode:

    sendcmd → crop@c → scale 1080x1920 → subtitles burn → loudnorm

Deduping to change-points matters: a 45 s clip at 25 fps is 1125 frames and
writing every parameter every frame slows the filter measurably (openshorts'
own comment). Even dimensions everywhere — x264/NVENC reject odd ones.

Encoder tiers follow openshorts ffmpeg_utils.py: try hardware
(h264_videotoolbox on macOS), fall back to libx264, mapping quality between
CRF and the hardware encoder's bitrate model.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from . import ffmpeg_bin

OUT_W = 1080
OUT_H = 1920
X264_CRF = 19
VT_BITRATE = "10M"

# Wide clips are letterboxed rather than stretched, and the bars are filled
# with a blurred, over-scaled copy of the same frame. Black bars are the
# honest alternative and cost nothing, but they read as an upload mistake on
# a feed where every neighbouring post fills the screen. Sigma is high
# enough that no detail survives to compete with the picture.
BLUR_SIGMA = 30

_vt_checked: bool | None = None


def videotoolbox_available() -> bool:
    """Probe once: encode 0.2 s of black through h264_videotoolbox."""
    global _vt_checked
    if _vt_checked is None:
        proc = subprocess.run(
            [
                ffmpeg_bin.ffmpeg(), "-v", "error", "-f", "lavfi", "-i", "color=black:s=320x240:d=0.2",
                "-c:v", "h264_videotoolbox", "-f", "null", "-",
            ],
            capture_output=True, timeout=60,
        )
        _vt_checked = proc.returncode == 0
    return _vt_checked


def crop_boxes(frames: list[list[float]], src_w: int, src_h: int) -> list[tuple[int, int, int, int]]:
    """Director frames [x, y, w, h] → even-int (w, h, x, y) crop boxes,
    clamped in-bounds (openshorts crop_boxes rounding rules)."""
    boxes: list[tuple[int, int, int, int]] = []
    for x, y, w, h in frames:
        wi = max(2, min(int(w) - int(w) % 2, src_w))
        hi = max(2, min(int(h) - int(h) % 2, src_h))
        xi = max(0, min(int(round(x)), src_w - wi))
        yi = max(0, min(int(round(y)), src_h - hi))
        boxes.append((wi, hi, xi - xi % 2, yi - yi % 2))
    return boxes


def sendcmd_lines(boxes: list[tuple[int, int, int, int]], fps: float, target: str = "crop@c") -> list[str]:
    """Per-frame w/h/x/y commands, deduped to change-points (openshorts)."""
    lines: list[str] = []
    prev: tuple[int, int, int, int] | None = None
    for i, box in enumerate(boxes):
        if box == prev:
            continue
        t = i / fps
        w, h, x, y = box
        pw, ph, px, py = prev if prev else (None, None, None, None)
        if w != pw:
            lines.append(f"{t:.4f} {target} w {w};")
        if h != ph:
            lines.append(f"{t:.4f} {target} h {h};")
        if x != px:
            lines.append(f"{t:.4f} {target} x {x};")
        if y != py:
            lines.append(f"{t:.4f} {target} y {y};")
        prev = box
    return lines


def _join(parts: list[str]) -> str:
    """Assemble filter chain segments into one filtergraph string.

    ffmpeg separates filters within a chain by ',' and whole chains by ';'.
    The vertical path is a single chain and only ever needs commas; the wide
    path splits into labelled branches and needs both. Rather than have
    every caller track which is which, the labels themselves say it: a
    segment that opens with '[' starts a new chain, and one that closes with
    ']' ends the previous one.
    """
    out = ""
    for part in parts:
        if not out:
            out = part
            continue
        sep = ";" if (part.startswith("[") or out.endswith("]")) else ","
        out += sep + part
    return out


def _compose(mode: str, first_box: tuple[int, int, int, int]) -> list[str]:
    """The filter chain from decoded frames to a full 1080x1920 canvas.

    Vertical is a straight scale: the crop already has the canvas aspect.
    Wide has to place a 4:3 picture inside a 9:16 frame, which means one
    decode feeding two branches — the picture itself, and the blurred fill
    behind it. Both branches read the SAME cropped rect, so sendcmd still
    drives a single crop filter and the two stay in lockstep by
    construction.
    """
    w0, h0, x0, y0 = first_box
    head = [
        f"crop@c=w={w0}:h={h0}:x={x0}:y={y0}",
    ]
    if mode != "wide":
        return head + [f"scale={OUT_W}:{OUT_H}:flags=lanczos", "setsar=1"]

    pic_h = int(round(OUT_W * 3 / 4))
    pic_h -= pic_h % 2
    top = (OUT_H - pic_h) // 2
    return head + [
        "split=2[wide_fg][wide_bg]",
        f"[wide_bg]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
        f"crop={OUT_W}:{OUT_H},gblur=sigma={BLUR_SIGMA}[wide_bgb]",
        f"[wide_fg]scale={OUT_W}:{pic_h}:flags=lanczos[wide_fgs]",
        f"[wide_bgb][wide_fgs]overlay=0:{top}",
        "setsar=1",
    ]


def _q(path: str) -> str:
    """ffmpeg filter-option quoting: single quotes make the value literal;
    an embedded quote closes, escapes, reopens ('\\'').

    Windows adds two wrinkles the mac path never sees: backslash is
    ffmpeg's escape character even inside quotes (av_get_token), and the
    drive-letter colon reads as an option separator on some parse levels.
    Forward slashes (fine for libass and every filter) plus an escaped
    colon is the canonical portable form: 'C\\:/Users/…/clip.ass'."""
    text = str(path)
    if os.name == "nt":
        text = text.replace("\\", "/").replace(":", "\\:")
    return "'" + text.replace("'", "'\\''") + "'"


def render_clip(
    media_path: str,
    out_path: Path,
    clip_start: float,
    clip_end: float,
    trajectory: dict,
    ass_path: Path | None,
    fonts_dir: Path | None,
    lufs: float = -14.0,
    true_peak: float = -1.0,
    src_w: int = 1920,
    src_h: int = 1080,
    timeout: float = 1800.0,
    mode: str = "vertical",
) -> None:
    duration = clip_end - clip_start
    boxes = crop_boxes(trajectory["frames"], src_w, src_h)
    if not boxes:
        boxes = [(src_h * 9 // 16 // 2 * 2, src_h - src_h % 2, 0, 0)]
    fps = float(trajectory.get("fps", 25))

    cmd_path = out_path.with_suffix(".cmd")
    cmd_path.write_text("\n".join(sendcmd_lines(boxes, fps)) + "\n", encoding="utf-8")

    vf_parts = [f"sendcmd=f={_q(cmd_path)}"] + _compose(mode, boxes[0])
    if ass_path is not None:
        sub = f"subtitles=filename={_q(ass_path)}"
        if fonts_dir is not None:
            sub += f":fontsdir={_q(fonts_dir)}"
        vf_parts.append(sub)

    # Chain segments are joined with ',' except where a segment already ends
    # a labelled branch — those are separate graph statements and take ';'.
    graph = _join(vf_parts)

    if videotoolbox_available():
        vcodec = ["-c:v", "h264_videotoolbox", "-b:v", VT_BITRATE, "-allow_sw", "1"]
    else:
        vcodec = ["-c:v", "libx264", "-preset", "medium", "-crf", str(X264_CRF)]

    args = [
        ffmpeg_bin.ffmpeg(), "-y", "-v", "error",
        "-ss", f"{clip_start:.3f}", "-t", f"{duration:.3f}",
        "-i", media_path,
        "-vf", graph,
        "-af", f"loudnorm=I={lufs}:TP={true_peak}:LRA=11",
        *vcodec,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        "-map_metadata", "-1",  # metadata scrub (openshorts ffmpeg_utils)
        str(out_path),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    cmd_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Render failed: {(proc.stderr or '')[-800:]}")


def verify_output(out_path: Path, expected_duration: float) -> dict:
    """Post-render sanity: exists, has both streams, duration within 1.5 s."""
    proc = subprocess.run(
        [
            ffmpeg_bin.ffprobe(), "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(out_path),
        ],
        capture_output=True, text=True, timeout=120,
    )
    import json

    info = json.loads(proc.stdout or "{}")
    streams = info.get("streams", [])
    has_v = any(s.get("codec_type") == "video" for s in streams)
    has_a = any(s.get("codec_type") == "audio" for s in streams)
    duration = float(info.get("format", {}).get("duration", 0.0))
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    return {
        "ok": has_v and has_a and abs(duration - expected_duration) < 1.5,
        "duration": duration,
        "width": video.get("width"),
        "height": video.get("height"),
    }
