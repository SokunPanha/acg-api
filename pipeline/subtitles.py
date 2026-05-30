import json
import os
import re
import subprocess
from typing import Callable, Optional

from pydub import AudioSegment

from .config import PipelineConfig

LogFn      = Optional[Callable[[str, str], None]]
ProgressFn = Optional[Callable[[float, str], None]]


def build_subtitles(
    sections: list[str],
    config: PipelineConfig,
    on_log: LogFn = None,
    on_progress: ProgressFn = None,
) -> bool:
    def log(msg, level="info"):
        if on_log:
            on_log(msg, level)

    def progress(val, text=""):
        if on_progress:
            on_progress(val, text)

    log("Building subtitles…", "info")
    progress(0.33, "Building SRT…")

    try:
        # prefer per-section mp3s; fall back to cached durations (mp3s are pruned after merge)
        audio_files = _sorted_mp3s(config.mp3_dir)
        if audio_files:
            durations = [len(AudioSegment.from_file(a)) / 1000 for a in audio_files]
        elif os.path.exists(config.durations_path):
            with open(config.durations_path) as f:
                durations = json.load(f)
        else:
            log("No audio durations available. Run Step 1 first.", "error")
            return False

        srt = _build_srt(sections, durations)
        with open(config.srt_path, "w", encoding="utf-8") as f:
            f.write(srt)
        entry_count = srt.count("\n\n")
        log(f"SRT created with {entry_count} entries", "ok")

        # merge only if it hasn't been done already (it normally has, right after voice gen)
        if not os.path.exists(config.merged_audio):
            if not audio_files:
                log("No audio files to merge.", "error")
                return False
            _write_concat(config.concat_file, audio_files)
            r = subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", config.concat_file, "-c", "copy", config.merged_audio],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                log(f"ffmpeg merge failed:\n{r.stderr[-400:]}", "error")
                return False

        return True

    except Exception as exc:
        log(f"Error: {exc}", "error")
        return False


# ── helpers ────────────────────────────────────────────────────────────────────

def _sorted_mp3s(mp3_dir: str) -> list[str]:
    if not os.path.isdir(mp3_dir):
        return []
    return sorted(
        [os.path.join(mp3_dir, f) for f in os.listdir(mp3_dir) if f.endswith(".mp3")],
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )


MAX_WORDS = 10


def _chunk_words(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Break a string into chunks of at most `max_words` words."""
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)] or [text]


def _build_srt(sections: list[str], durations: list[float]) -> str:
    srt = ""
    t   = 0.0
    idx = 1
    for block, dur in zip(sections, durations):
        # split into sentences, then cap each at MAX_WORDS words per cue
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', block) if s.strip()]
        cues: list[str] = []
        for sent in sentences:
            cues.extend(_chunk_words(sent))
        if not cues:
            continue

        # distribute the block's audio time across cues by character length
        total_ch = sum(len(c) for c in cues)
        for cue in cues:
            ratio = len(cue) / total_ch if total_ch else 1 / len(cues)
            end   = t + dur * ratio
            srt  += f"{idx}\n{_fmt(t)} --> {_fmt(end)}\n{cue}\n\n"
            t, idx = end, idx + 1
    return srt


def _write_concat(concat_file: str, audio_files: list[str]) -> None:
    with open(concat_file, "w") as f:
        for a in audio_files:
            f.write(f"file '{a}'\n")


def _fmt(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s   = divmod(rem, 60)
    ms     = int((sec - int(sec)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"
