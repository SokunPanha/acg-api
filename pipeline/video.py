import os
import re
import subprocess
import time
from typing import Callable, Optional

from .config import PipelineConfig, _hex_to_ass

LogFn      = Optional[Callable[[str, str], None]]
ProgressFn = Optional[Callable[[float, str], None]]

# sizing convention — the frontend preview MUST mirror these so preview == render
FONT_DIVISOR    = 600    # fontPx   = height * font_size / 600
OUTLINE_DIVISOR = 1000   # outline  = height * outline   / 1000
MARGIN_FRAC     = 0.06   # vertical margin from edge = 6% of height


def _srt_time_to_ass(t: str) -> str:
    # "00:01:02,500" → "0:01:02.50"
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    cs = int(ms) // 10
    return f"{int(h)}:{int(m):02d}:{int(s):02d}.{cs:02d}"


def _parse_srt(path: str):
    with open(path, "r", encoding="utf-8") as f:
        blocks = re.split(r"\n\s*\n", f.read().strip())
    cues = []
    for b in blocks:
        lines = [l for l in b.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        ti = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if ti is None:
            continue
        start, end = [x.strip() for x in lines[ti].split("-->")]
        text = "\\N".join(lines[ti + 1:])
        cues.append((_srt_time_to_ass(start), _srt_time_to_ass(end), text))
    return cues


def _build_ass(config: PipelineConfig, ass_path: str) -> None:
    """Render the SRT into an ASS file whose PlayRes equals the video — so font size,
    outline and position are exact pixels and match the on-screen preview."""
    w, h    = config.dimensions
    font_px = max(8, round(h * config.subtitle_font_size / FONT_DIVISOR))
    outline = round(h * config.subtitle_outline / OUTLINE_DIVISOR)
    margin  = round(h * MARGIN_FRAC)
    align   = {"bottom": 2, "middle": 5, "top": 8}.get(config.subtitle_position, 2)
    fontname = config.subtitle_font or "Arial"
    primary  = _hex_to_ass(config.subtitle_color)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {w}\nPlayResY: {h}\n"
        "WrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{fontname},{font_px},{primary},&H000000FF,&H00000000,&H00000000,"
        f"-1,0,0,0,100,100,0,0,1,{outline},0,{align},60,60,{margin},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [f"Dialogue: 0,{s},{e},Default,,0,0,0,,{t}" for s, e, t in _parse_srt(config.srt_path)]
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines) + "\n")


def render_video(
    config: PipelineConfig,
    on_log: LogFn = None,
    on_progress: ProgressFn = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> bool:
    def log(msg, level="info"):
        if on_log:
            on_log(msg, level)

    def progress(val, text=""):
        if on_progress:
            on_progress(val, text)

    log("Rendering video…", "info")
    progress(0.66, "Rendering…")

    try:
        cmd = _build_ffmpeg_cmd(config)
        log("Running ffmpeg…")

        # Popen + poll so the render can be cancelled mid-encode (terminates ffmpeg)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while proc.poll() is None:
            if should_cancel and should_cancel():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                log("Render cancelled.", "warn")
                # remove the partial/corrupt output
                if os.path.exists(config.output_path):
                    try: os.remove(config.output_path)
                    except OSError: pass
                return False
            time.sleep(0.3)

        if proc.returncode != 0:
            err = (proc.stderr.read() or "")[-400:]
            log(f"ffmpeg error:\n{err}", "error")
            return False

        log(f"Video saved: {config.output_path}", "ok")
        progress(1.0, "Done!")
        return True

    except Exception as exc:
        log(f"Error: {exc}", "error")
        return False


def _build_ffmpeg_cmd(config: PipelineConfig) -> list[str]:
    w, h = config.dimensions
    is_video = config.background.endswith(".mp4")

    # cover the frame without stretching: scale up to fill (lanczos = sharper), then crop to exact size
    scale = f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}"
    if config.include_subtitles and os.path.exists(config.srt_path):
        ass_path = os.path.join(config.session_dir, "subtitles.ass")
        _build_ass(config, ass_path)
        fontsdir = f":fontsdir='{config.fonts_dir}'" if config.fonts_dir else ""
        vf = f"{scale},ass='{ass_path}'{fontsdir}"
    else:
        vf = scale

    cmd = ["ffmpeg", "-y"]

    if is_video:
        cmd += ["-stream_loop", "-1", "-i", config.background]
    else:
        cmd += ["-loop", "1", "-i", config.background]

    # static image never moves → low fps means far fewer frames to encode (huge speedup).
    # real video → keep 30fps for smooth motion.
    fps = "30" if is_video else "10"

    cmd += [
        "-i", config.merged_audio,
        # take the video from the background (input 0) and the AUDIO from the voice (input 1).
        # explicit mapping is essential for video backgrounds: otherwise ffmpeg may pick the
        # background clip's own audio track instead of the generated voice.
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-vf", vf,
        "-r", fps,
        "-c:v", "libx264",
        "-preset", "veryfast",      # ~5-10x faster than "slow"; minimal quality loss at same CRF
        "-crf", "20",               # high quality (lower = better; 18 was overkill/slow)
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",                # loop the bg video forever, but stop when the voice ends
    ]
    # static-image backgrounds: tune for stills so bitrate isn't wasted chasing fake motion
    if not is_video:
        cmd += ["-tune", "stillimage"]

    cmd += [config.output_path]
    return cmd
