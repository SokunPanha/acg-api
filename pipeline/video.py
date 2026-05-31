import os
import re
import subprocess
import time
from typing import Callable, Optional

from .config import PipelineConfig, _hex_to_ass

LogFn      = Optional[Callable[[str, str], None]]
ProgressFn = Optional[Callable[[float, str], None]]

# GPU encoding. Default forces NVIDIA NVENC; override with VIDEO_ENCODER=libx264 etc.
VIDEO_ENCODER = os.environ.get("VIDEO_ENCODER", "h264_nvenc")
# allow GPU→CPU fallback if the GPU/driver isn't available (set "0" to fail hard instead)
ENCODER_FALLBACK = os.environ.get("VIDEO_ENCODER_FALLBACK", "1") != "0"


def _encoder_args(encoder: str) -> list[str]:
    """Codec + rate-control flags for the chosen encoder."""
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
                "-rc", "vbr", "-cq", "21", "-b:v", "0", "-pix_fmt", "yuv420p"]
    if encoder == "hevc_nvenc":
        return ["-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "23", "-b:v", "0", "-pix_fmt", "yuv420p"]
    if encoder == "h264_videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-q:v", "55", "-pix_fmt", "yuv420p"]
    # CPU fallback / default
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]

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

    # try the configured (GPU) encoder; fall back to CPU libx264 if the GPU isn't usable
    encoders = [VIDEO_ENCODER]
    if ENCODER_FALLBACK and VIDEO_ENCODER != "libx264":
        encoders.append("libx264")

    # ffmpeg writes a continuous progress stream to stderr. Capturing that via
    # subprocess.PIPE without draining it deadlocks long renders: the ~64 KB OS pipe
    # buffer fills (after ~15 min of output), ffmpeg blocks on its next write and never
    # exits. Redirect to a log FILE instead — it can't fill — and read its tail on error.
    log_path = os.path.join(config.session_dir, "ffmpeg.log")

    for i, encoder in enumerate(encoders):
        try:
            cmd = _build_ffmpeg_cmd(config, encoder)
            log(f"Running ffmpeg ({encoder})…")

            # Popen + poll so the render can be cancelled mid-encode (terminates ffmpeg)
            with open(log_path, "wb") as logf:
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=logf)
                while proc.poll() is None:
                    if should_cancel and should_cancel():
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        log("Render cancelled.", "warn")
                        if os.path.exists(config.output_path):
                            try: os.remove(config.output_path)
                            except OSError: pass
                        return False
                    time.sleep(0.3)

            if proc.returncode == 0:
                log(f"Video saved: {config.output_path}", "ok")
                progress(1.0, "Done!")
                return True

            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    err = f.read()[-600:]
            except OSError:
                err = ""
            # GPU encoder failed (no GPU/driver in this environment) → try CPU next
            if i + 1 < len(encoders):
                log(f"{encoder} failed — falling back to CPU (libx264). Reason:\n{err}", "warn")
                continue
            log(f"ffmpeg error:\n{err}", "error")
            return False

        except Exception as exc:
            log(f"Error: {exc}", "error")
            return False

    return False


def _build_ffmpeg_cmd(config: PipelineConfig, encoder: str = VIDEO_ENCODER) -> list[str]:
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

    # -nostats/-loglevel keep stderr small (no per-frame progress spam), so the log file
    # stays tiny and we still capture genuine warnings/errors for diagnostics.
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "warning", "-y"]

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
    ]
    cmd += _encoder_args(encoder)
    cmd += [
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",                # loop the bg video forever, but stop when the voice ends
    ]
    # static-image backgrounds with libx264: tune for stills (nvenc has no such tune)
    if not is_video and encoder == "libx264":
        cmd += ["-tune", "stillimage"]

    cmd += [config.output_path]
    return cmd
