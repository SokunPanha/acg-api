import asyncio
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime
from typing import AsyncGenerator

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from ai import generate_script, chat_with_retry
from pipeline import PipelineConfig, build_subtitles, generate_voice, render_video

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR      = os.path.join(BASE_DIR, "assets")
VOICES_DIR      = os.path.join(BASE_DIR, "assets", "voices")
BACKGROUNDS_DIR = os.path.join(BASE_DIR, "assets", "backgrounds")
SESSIONS_DIR    = os.path.join(BASE_DIR, "output", "sessions")
SCRIPT_FILE     = os.path.join(BASE_DIR, "script.txt")
SOURCES         = ("single", "bulk")   # output is split into separate buckets

os.makedirs(ASSETS_DIR,      exist_ok=True)
os.makedirs(VOICES_DIR,      exist_ok=True)
os.makedirs(BACKGROUNDS_DIR, exist_ok=True)
for _s in SOURCES:
    os.makedirs(os.path.join(SESSIONS_DIR, _s), exist_ok=True)


def _sess_dir(sid: str, source: str | None = None) -> str:
    """Path to a session folder. With `source`, returns the bucket path directly;
    otherwise searches the buckets (falls back to the legacy flat path)."""
    if source:
        return os.path.join(SESSIONS_DIR, source, sid)
    for src in SOURCES:
        p = os.path.join(SESSIONS_DIR, src, sid)
        if os.path.isdir(p):
            return p
    return os.path.join(SESSIONS_DIR, sid)


def _source_of(sid: str) -> str:
    for src in SOURCES:
        if os.path.isdir(os.path.join(SESSIONS_DIR, src, sid)):
            return src
    return "single"


def _all_sessions() -> list[str]:
    """All session ids across buckets (+ legacy flat)."""
    ids: list[str] = []
    for src in SOURCES:
        d = os.path.join(SESSIONS_DIR, src)
        if os.path.isdir(d):
            ids += [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
    return ids

BG_EXTS = (".jpg", ".jpeg", ".png", ".mp4")

FONTS_DIR = os.path.join(BASE_DIR, "assets", "fonts")
os.makedirs(FONTS_DIR, exist_ok=True)


_FONT_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def _resolve_font(family: str) -> tuple[str, str]:
    """Ensure the Google Font TTF is downloaded & cached. Returns (font_name, fonts_dir) or ('','')."""
    import urllib.request
    family = (family or "").strip()
    if not family:
        return "", ""
    path = os.path.join(FONTS_DIR, family.replace(" ", "") + ".ttf")
    if not os.path.exists(path):
        try:
            fid = family.lower().replace(" ", "-")
            req = urllib.request.Request(f"https://gwfh.mranftl.com/api/fonts/{fid}?subsets=latin", headers=_FONT_UA)
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.load(r)
            variants = data.get("variants", [])
            v = next((x for x in variants if x.get("id") in ("regular", "700", "800")), variants[0] if variants else None)
            ttf = v.get("ttf") if v else None
            if not ttf:
                return "", ""
            with urllib.request.urlopen(urllib.request.Request(ttf, headers=_FONT_UA), timeout=30) as r, open(path, "wb") as f:
                shutil.copyfileobj(r, f)
        except Exception:
            return "", ""   # fall back to default font on any failure
    return family, FONTS_DIR

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# keep strong references to background work so a client disconnect (page refresh)
# doesn't let the event loop garbage-collect & cancel an in-flight pipeline/render
_bg_tasks: set = set()

# session ids the user has asked to cancel; the voice loop checks this cooperatively
_cancel_flags: set[str] = set()


def _spawn(coro):
    t = asyncio.create_task(coro)
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)
    return t


@app.on_event("startup")
def _reconcile_sessions():
    """On boot: migrate legacy flat sessions into the 'single' bucket, then fix any stuck 'running'."""
    # migrate legacy output/sessions/<id> → output/sessions/single/<id>
    if os.path.isdir(SESSIONS_DIR):
        for name in os.listdir(SESSIONS_DIR):
            p = os.path.join(SESSIONS_DIR, name)
            if os.path.isdir(p) and name not in SOURCES and os.path.exists(os.path.join(p, "meta.json")):
                dst = os.path.join(SESSIONS_DIR, "single", name)
                if not os.path.exists(dst):
                    shutil.move(p, dst)

    for sid in _all_sessions():
        meta = _session_meta(sid)
        if meta.get("status") == "running":
            has_audio = os.path.exists(os.path.join(_sess_dir(sid), "merged.mp3"))
            _save_meta(sid, {
                "status":    "done" if has_audio else "error",
                "has_audio": has_audio,
                "error":     None if has_audio else "Interrupted (page refresh or restart)",
            })


def _require_groq_key() -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set in environment.")
    return GROQ_API_KEY


def _make_config(session_id: str, voice_filename: str = "", background_filename: str = "", source: str | None = None, **kwargs) -> PipelineConfig:
    if background_filename and os.path.exists(os.path.join(BACKGROUNDS_DIR, background_filename)):
        bg_path = os.path.join(BACKGROUNDS_DIR, background_filename)
    else:
        bg_path = next(
            (os.path.join(BACKGROUNDS_DIR, f) for f in sorted(os.listdir(BACKGROUNDS_DIR))
             if f.lower().endswith(BG_EXTS)),
            "",
        )
    voice_path = (
        os.path.join(VOICES_DIR, voice_filename)
        if voice_filename and os.path.exists(os.path.join(VOICES_DIR, voice_filename))
        else next((os.path.join(VOICES_DIR, f) for f in os.listdir(VOICES_DIR) if f.endswith((".mp3", ".wav"))), "")
    )
    src = source or _source_of(session_id)
    return PipelineConfig(
        work_dir    = BASE_DIR,
        session_id  = session_id,
        source      = src,
        voice_ref   = voice_path,
        background  = bg_path,
        output_path = os.path.join(SESSIONS_DIR, src, session_id, "final.mp4"),
        **kwargs,
    )


def _sse(event: str, data: dict) -> str:
    return f"data: {json.dumps({'event': event, **data})}\n\n"


def _session_meta(session_id: str) -> dict:
    meta_path = os.path.join(_sess_dir(session_id), "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        return json.load(f)


def _save_meta(session_id: str, data: dict) -> None:
    session_dir = _sess_dir(session_id)
    os.makedirs(session_dir, exist_ok=True)
    meta_path = os.path.join(session_dir, "meta.json")
    existing = _session_meta(session_id)
    existing.update(data)
    with open(meta_path, "w") as f:
        json.dump(existing, f)


# ── script ─────────────────────────────────────────────────────────────────────

@app.post("/api/script/generate")
async def script_generate(topic: str = Form(...), duration: int = Form(...)):
    script = await asyncio.to_thread(generate_script, _require_groq_key(), topic, duration)
    with open(SCRIPT_FILE, "w", encoding="utf-8") as f:
        f.write(script)
    return {"script": script}


@app.get("/api/script")
def script_get():
    if not os.path.exists(SCRIPT_FILE):
        return {"script": ""}
    with open(SCRIPT_FILE, "r", encoding="utf-8") as f:
        return {"script": f.read()}


@app.post("/api/script")
async def script_save(script: str = Form(...)):
    with open(SCRIPT_FILE, "w", encoding="utf-8") as f:
        f.write(script)
    return {"ok": True}


# ── file uploads ───────────────────────────────────────────────────────────────

@app.get("/api/voices")
def voices_list():
    files = [
        f for f in os.listdir(VOICES_DIR)
        if f.lower().endswith((".mp3", ".wav"))
    ]
    return sorted(files)


@app.get("/api/voices/{filename}")
def voice_preview(filename: str):
    path = os.path.join(VOICES_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Voice not found.")
    ext = os.path.splitext(filename)[1].lower()
    media_type = "audio/wav" if ext == ".wav" else "audio/mpeg"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.post("/api/voices")
async def upload_voice(file: UploadFile = File(...), name: str = Form("")):
    ext      = os.path.splitext(file.filename or "voice.mp3")[1] or ".mp3"
    base     = (name.strip()[:15] or os.path.splitext(os.path.basename(file.filename or "voice"))[0][:15])
    safe_name = base + ext
    dest = os.path.join(VOICES_DIR, safe_name)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": safe_name}


@app.put("/api/voices/{filename}/rename")
async def voice_rename(filename: str, new_name: str = Form(...)):
    ext      = os.path.splitext(filename)[1]
    safe     = new_name.strip()[:15]
    if not safe:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    new_filename = safe + ext
    src = os.path.join(VOICES_DIR, filename)
    dst = os.path.join(VOICES_DIR, new_filename)
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail="Voice not found.")
    if os.path.exists(dst) and src != dst:
        raise HTTPException(status_code=409, detail="A voice with that name already exists.")
    os.rename(src, dst)
    return {"filename": new_filename}


@app.delete("/api/voices/{filename}")
def voice_delete(filename: str):
    path = os.path.join(VOICES_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
    return {"ok": True}


# ── background library ─────────────────────────────────────────────────────────

@app.get("/api/backgrounds")
def backgrounds_list():
    files = [f for f in os.listdir(BACKGROUNDS_DIR) if f.lower().endswith(BG_EXTS)]
    return sorted(files)


@app.get("/api/backgrounds/{filename}")
def background_preview(filename: str):
    path = os.path.join(BACKGROUNDS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Background not found.")
    ext = os.path.splitext(filename)[1].lower()
    media_type = "video/mp4" if ext == ".mp4" else f"image/{'jpeg' if ext in ('.jpg', '.jpeg') else 'png'}"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.post("/api/backgrounds")
async def upload_background(file: UploadFile = File(...), name: str = Form("")):
    ext       = os.path.splitext(file.filename or "background.jpg")[1] or ".jpg"
    base      = (name.strip()[:15] or os.path.splitext(os.path.basename(file.filename or "background"))[0][:15])
    safe_name = base + ext
    dest = os.path.join(BACKGROUNDS_DIR, safe_name)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": safe_name}


@app.put("/api/backgrounds/{filename}/rename")
async def background_rename(filename: str, new_name: str = Form(...)):
    ext  = os.path.splitext(filename)[1]
    safe = new_name.strip()[:15]
    if not safe:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    new_filename = safe + ext
    src = os.path.join(BACKGROUNDS_DIR, filename)
    dst = os.path.join(BACKGROUNDS_DIR, new_filename)
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail="Background not found.")
    if os.path.exists(dst) and src != dst:
        raise HTTPException(status_code=409, detail="A background with that name already exists.")
    os.rename(src, dst)
    return {"filename": new_filename}


@app.delete("/api/backgrounds/{filename}")
def background_delete(filename: str):
    path = os.path.join(BACKGROUNDS_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
    return {"ok": True}


# ── pipeline ───────────────────────────────────────────────────────────────────

@app.get("/api/pipeline/run")
async def pipeline_run(
    steps: str = "1,2,3",
    title: str = "Untitled",
    voice: str = "",
    instruction: str = "Gentle and calm voice tone",
    cfg: float = 2.0,
    denoise: bool = False,
    normalize: bool = False,
    include_subtitles: bool = True,
    background: str = "",
    resolution: str = "16:9",
    sub_font_size: int = 24,
    sub_color: str = "#FFFFFF",
    sub_position: str = "bottom",
    sub_outline: int = 2,
    sub_font: str = "",
):
    step_list  = [int(s) for s in steps.split(",") if s.strip()]
    session_id = str(uuid.uuid4())
    font_name, fonts_dir = _resolve_font(sub_font)
    config     = _make_config(
        session_id,
        source="single",
        voice_filename=voice,
        background_filename=background,
        control_instruction=instruction,
        cfg_value=cfg,
        denoise=denoise,
        normalize=normalize,
        include_subtitles=include_subtitles,
        resolution=resolution,
        subtitle_font_size=sub_font_size,
        subtitle_color=sub_color,
        subtitle_position=sub_position,
        subtitle_outline=sub_outline,
        subtitle_font=font_name,
        fonts_dir=fonts_dir,
    )
    os.makedirs(config.session_dir, exist_ok=True)

    if not os.path.exists(SCRIPT_FILE):
        async def _err():
            yield _sse("error", {"message": "No script found. Generate or save one first."})
        return StreamingResponse(_err(), media_type="text/event-stream")

    with open(SCRIPT_FILE, "r", encoding="utf-8") as f:
        sections = [s.strip() for s in f.read().split("#") if s.strip()]

    # save initial metadata
    _save_meta(session_id, {
        "id":        session_id,
        "title":     title,
        "created_at": datetime.utcnow().isoformat(),
        "steps":     step_list,
        "status":    "running",
    })

    # copy script into session folder
    shutil.copy(SCRIPT_FILE, os.path.join(config.session_dir, "script.txt"))

    _cancel_flags.discard(session_id)
    should_cancel = lambda: session_id in _cancel_flags

    async def stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        log_queue: asyncio.Queue = asyncio.Queue()

        def on_log(msg: str, level: str = "info"):
            loop.call_soon_threadsafe(log_queue.put_nowait, ("log", msg, level))

        def on_progress(value: float, text: str = ""):
            loop.call_soon_threadsafe(log_queue.put_nowait, ("progress", value, text))

        async def run_pipeline():
            if 1 in step_list:
                on_log("▶ Step 1: Generate Voice", "info")
                ok = await asyncio.to_thread(generate_voice, sections, config, on_log, on_progress, should_cancel)
                if should_cancel():
                    _cancel_flags.discard(session_id)
                    _save_meta(session_id, {"status": "cancelled"})
                    loop.call_soon_threadsafe(log_queue.put_nowait, ("step", 0, "error"))
                    loop.call_soon_threadsafe(log_queue.put_nowait, ("done", False, session_id))
                    return
                loop.call_soon_threadsafe(log_queue.put_nowait, ("step", 0, "done" if ok else "error"))
                if not ok:
                    _save_meta(session_id, {"status": "error"})
                    loop.call_soon_threadsafe(log_queue.put_nowait, ("done", False, session_id))
                    return
                await asyncio.to_thread(_merge_audio, config, on_log)
                _save_meta(session_id, {"has_audio": True})

            if 2 in step_list:
                on_log("▶ Step 2: Build Subtitles", "info")
                ok = await asyncio.to_thread(build_subtitles, sections, config, on_log, on_progress)
                loop.call_soon_threadsafe(log_queue.put_nowait, ("step", 1, "done" if ok else "error"))
                if not ok:
                    _save_meta(session_id, {"status": "error"})
                    loop.call_soon_threadsafe(log_queue.put_nowait, ("done", False, session_id))
                    return
                _save_meta(session_id, {"has_subtitle": True})

            if 3 in step_list:
                on_log("▶ Step 3: Render Video", "info")
                ok = await asyncio.to_thread(render_video, config, on_log, on_progress)
                loop.call_soon_threadsafe(log_queue.put_nowait, ("step", 2, "done" if ok else "error"))
                if ok:
                    _save_meta(session_id, {"has_video": True, "resolution": resolution})

            _save_meta(session_id, {"status": "done"})
            loop.call_soon_threadsafe(log_queue.put_nowait, ("done", True, session_id))

        _spawn(run_pipeline())

        yield _sse("session_id", {"session_id": session_id})

        while True:
            item = await log_queue.get()
            if item[0] == "log":
                yield _sse("log", {"message": item[1], "level": item[2]})
            elif item[0] == "progress":
                yield _sse("progress", {"value": item[1], "text": item[2]})
            elif item[0] == "step":
                yield _sse("step", {"index": item[1], "status": item[2]})
            elif item[0] == "done":
                yield _sse("done", {"success": item[1], "session_id": item[2]})
                break

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── helpers ────────────────────────────────────────────────────────────────────

def _download_name(session_id: str, ext: str) -> str:
    """Return a human-readable download filename from the session title."""
    import re
    title = _session_meta(session_id).get("title", "").strip()
    if not title:
        title = f"voice_{datetime.utcnow().strftime('%Y%m%d')}"
    safe = re.sub(r"[^\w\s-]", "", title)
    safe = re.sub(r"[\s]+", "-", safe).strip("-").lower()
    return f"{safe[:60]}{ext}"


def _merge_audio(config: PipelineConfig, on_log=None) -> None:
    import glob
    from pydub import AudioSegment

    mp3s = sorted(
        glob.glob(os.path.join(config.mp3_dir, "*.mp3")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    if not mp3s:
        return

    # cache per-section durations BEFORE merging, so subtitles can be built
    # later even after the individual section mp3s are pruned
    try:
        section_durations = [len(AudioSegment.from_file(p)) / 1000 for p in mp3s]
        with open(config.durations_path, "w") as f:
            json.dump(section_durations, f)
    except Exception:
        section_durations = []

    with open(config.concat_file, "w") as f:
        for p in mp3s:
            f.write(f"file '{p}'\n")
    r = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", config.concat_file, "-c", "copy", config.merged_audio],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        try:
            duration_ms = len(AudioSegment.from_file(config.merged_audio))
            _save_meta(config.session_id, {"duration": round(duration_ms / 1000)})
        except Exception:
            pass
        # prune per-section mp3s + concat list now that they're merged (saves storage)
        try:
            shutil.rmtree(config.mp3_dir, ignore_errors=True)
            if os.path.exists(config.concat_file):
                os.remove(config.concat_file)
        except Exception:
            pass
        if on_log:
            on_log("Audio merged — ready to download.", "ok")
    elif on_log:
        on_log(f"Audio merge warning: {r.stderr[-200:]}", "warn")


def _render_session(session_id: str, resolution: str, background: str, subtitles: bool,
                    on_log=None, on_progress=None, sub_style: dict | None = None) -> bool:
    """Render a video for an existing session. Builds subtitles on demand. Cancellable via _cancel_flags."""
    should_cancel = lambda: session_id in _cancel_flags
    sub_style = sub_style or {}

    font_name, fonts_dir = _resolve_font(sub_style.get("font", ""))
    config = _make_config(
        session_id,
        background_filename = background,
        include_subtitles   = subtitles,
        resolution          = resolution,
        subtitle_font_size  = sub_style.get("font_size", 24),
        subtitle_color      = sub_style.get("color", "#FFFFFF"),
        subtitle_position   = sub_style.get("position", "bottom"),
        subtitle_outline    = sub_style.get("outline", 2),
        subtitle_font       = font_name,
        fonts_dir           = fonts_dir,
    )

    if should_cancel():
        return False
    if not os.path.exists(config.merged_audio):
        if on_log: on_log("No audio for this session — generate voice first.", "error")
        return False
    if not config.background:
        if on_log: on_log("No background available — upload one first.", "error")
        return False

    script_path = os.path.join(config.session_dir, "script.txt")
    if subtitles and not os.path.exists(config.srt_path) and os.path.exists(script_path):
        with open(script_path, "r", encoding="utf-8") as f:
            sections = [s.strip() for s in f.read().split("#") if s.strip()]
        if build_subtitles(sections, config, on_log, on_progress):
            _save_meta(session_id, {"has_subtitle": True})

    ok = render_video(config, on_log, on_progress, should_cancel)
    if ok:
        _save_meta(session_id, {"has_video": True, "resolution": resolution})
    return ok


# ── sessions ───────────────────────────────────────────────────────────────────

@app.get("/api/sessions")
def sessions_list(source: str = ""):
    # list a single bucket when `source` is given, otherwise all
    buckets = [source] if source in SOURCES else list(SOURCES)
    sessions = []
    for src in buckets:
        d = os.path.join(SESSIONS_DIR, src)
        if not os.path.isdir(d):
            continue
        for sid in os.listdir(d):
            meta = _session_meta(sid)
            if meta:
                sessions.append(meta)
    sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return sessions


@app.post("/api/sessions/{session_id}/cancel")
def session_cancel(session_id: str):
    """Request cancellation of an in-progress generation. The voice loop checks this cooperatively."""
    _cancel_flags.add(session_id)
    return {"ok": True}


@app.post("/api/sessions/mark-downloaded")
async def sessions_mark_downloaded(ids: str = Form(...)):
    for sid in json.loads(ids):
        if os.path.isdir(_sess_dir(sid)):
            _save_meta(sid, {"downloaded": True})
    return {"ok": True}


@app.post("/api/sessions/download-audio")
async def sessions_download_audio(ids: str = Form(...)):
    import io, zipfile, re

    id_list = json.loads(ids)
    if not id_list:
        raise HTTPException(status_code=400, detail="No sessions selected.")

    buf = io.BytesIO()
    used: set[str] = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for sid in id_list:
            path = os.path.join(_sess_dir(sid), "merged.mp3")
            if not os.path.exists(path):
                continue
            name = _download_name(sid, ".mp3")
            # avoid collisions when two titles sanitize to the same name
            if name in used:
                base, ext = os.path.splitext(name)
                name = f"{base}-{sid[:8]}{ext}"
            used.add(name)
            zf.write(path, arcname=name)
            _save_meta(sid, {"downloaded": True})

    buf.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="voices.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


@app.post("/api/sessions/download-video")
async def sessions_download_video(ids: str = Form(...)):
    import io, zipfile

    id_list = json.loads(ids)
    if not id_list:
        raise HTTPException(status_code=400, detail="No sessions selected.")

    buf = io.BytesIO()
    used: set[str] = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for sid in id_list:
            path = os.path.join(_sess_dir(sid), "final.mp4")
            if not os.path.exists(path):
                continue
            name = _download_name(sid, ".mp4")
            if name in used:
                base, ext = os.path.splitext(name)
                name = f"{base}-{sid[:8]}{ext}"
            used.add(name)
            zf.write(path, arcname=name)
            _save_meta(sid, {"downloaded": True})

    buf.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="videos.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


@app.get("/api/sessions/{session_id}/audio")
def session_audio(session_id: str, download: bool = False):
    path = os.path.join(_sess_dir(session_id), "merged.mp3")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Audio not found.")
    # inline by default (so <audio> plays it); attachment only for explicit downloads
    if download:
        return FileResponse(path, media_type="audio/mpeg", filename=_download_name(session_id, ".mp3"))
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/api/sessions/{session_id}/video")
def session_video(session_id: str, download: bool = False):
    path = os.path.join(_sess_dir(session_id), "final.mp4")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found.")
    # inline by default (so <video> plays it); attachment only for explicit downloads
    if download:
        return FileResponse(path, media_type="video/mp4", filename=_download_name(session_id, ".mp4"))
    return FileResponse(path, media_type="video/mp4")


@app.delete("/api/sessions/{session_id}/video")
def session_video_delete(session_id: str):
    """Delete ONLY the rendered video — the voice (merged.mp3) and the session stay intact."""
    path = os.path.join(_sess_dir(session_id), "final.mp4")
    if os.path.exists(path):
        os.remove(path)
    _save_meta(session_id, {"has_video": False, "resolution": None})
    return {"ok": True}


@app.get("/api/sessions/{session_id}/subtitles")
def session_subtitles(session_id: str):
    """Download existing .srt — or generate on-demand if not built yet."""
    config      = _make_config(session_id)
    script_path = os.path.join(_sess_dir(session_id), "script.txt")

    if not os.path.exists(config.srt_path):
        if not os.path.exists(script_path):
            raise HTTPException(status_code=404, detail="No script found for this session.")
        with open(script_path, "r", encoding="utf-8") as f:
            sections = [s.strip() for s in f.read().split("#") if s.strip()]
        ok = build_subtitles(sections, config)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to generate subtitles.")
        _save_meta(session_id, {"has_subtitle": True})

    return FileResponse(config.srt_path, media_type="text/plain", filename=_download_name(session_id, ".srt"))


@app.get("/api/sessions/{session_id}/render/stream")
async def session_render(
    session_id: str,
    resolution: str = "16:9",
    background: str = "",
    subtitles: bool = True,
    sub_font_size: int = 24,
    sub_color: str = "#FFFFFF",
    sub_position: str = "bottom",
    sub_outline: int = 2,
    sub_font: str = "",
):
    if not os.path.isdir(_sess_dir(session_id)):
        raise HTTPException(status_code=404, detail="Session not found.")
    sub_style = {"font_size": sub_font_size, "color": sub_color, "position": sub_position, "outline": sub_outline, "font": sub_font}

    _cancel_flags.discard(session_id)
    _save_meta(session_id, {"status": "running"})

    async def stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def on_log(msg, level="info"):
            loop.call_soon_threadsafe(q.put_nowait, ("log", {"message": msg, "level": level}))

        def on_progress(val, text=""):
            loop.call_soon_threadsafe(q.put_nowait, ("progress", {"value": val, "text": text}))

        async def run():
            ok = await asyncio.to_thread(_render_session, session_id, resolution, background, subtitles, on_log, on_progress, sub_style)
            cancelled = session_id in _cancel_flags
            _cancel_flags.discard(session_id)
            _save_meta(session_id, {"status": "cancelled" if cancelled else "done"})
            loop.call_soon_threadsafe(q.put_nowait, ("done", {"success": ok, "session_id": session_id}))

        _spawn(run())

        while True:
            event, data = await q.get()
            yield _sse(event, data)
            if event == "done":
                break

    return StreamingResponse(stream(), media_type="text/event-stream")


def _next_titled(title: str) -> str:
    """Given a title (possibly already 'Base (n)'), return the next free 'Base (k)'."""
    import re
    m    = re.match(r"^(.*?)\s*\((\d+)\)\s*$", title)
    base = (m.group(1) if m else title).strip()
    maxn = 0
    for sid in _all_sessions():
        t = _session_meta(sid).get("title", "")
        mm = re.match(r"^(.*?)\s*\((\d+)\)\s*$", t)
        if mm and mm.group(1).strip() == base:
            maxn = max(maxn, int(mm.group(2)))
    return f"{base} ({maxn + 1})"


@app.get("/api/sessions/{session_id}/rerender/stream")
async def session_rerender(
    session_id: str,
    resolution: str = "16:9",
    background: str = "",
    subtitles: bool = True,
    sub_font_size: int = 24,
    sub_color: str = "#FFFFFF",
    sub_position: str = "bottom",
    sub_outline: int = 2,
    sub_font: str = "",
):
    """Render a NEW video from an existing session's voice — keeps the original, names it 'Title (N)'."""
    src_dir = _sess_dir(session_id)
    src_audio = os.path.join(src_dir, "merged.mp3")
    if not os.path.exists(src_audio):
        raise HTTPException(status_code=404, detail="Source voice not found.")

    src_meta = _session_meta(session_id)
    new_id   = str(uuid.uuid4())
    new_dir  = _sess_dir(new_id, source=_source_of(session_id))
    os.makedirs(new_dir, exist_ok=True)

    # copy what the render needs: voice, durations, script, existing subtitles
    for name in ("merged.mp3", "durations.json", "script.txt", "subtitles.srt"):
        sp = os.path.join(src_dir, name)
        if os.path.exists(sp):
            shutil.copy(sp, os.path.join(new_dir, name))

    title = _next_titled(src_meta.get("title", "Video"))
    _save_meta(new_id, {
        "id":         new_id,
        "title":      title,
        "created_at": datetime.utcnow().isoformat(),
        "steps":      [3],
        "status":     "running",
        "has_audio":  True,
        "has_subtitle": os.path.exists(os.path.join(new_dir, "subtitles.srt")),
        "duration":   src_meta.get("duration"),
    })

    sub_style = {"font_size": sub_font_size, "color": sub_color, "position": sub_position, "outline": sub_outline, "font": sub_font}
    _cancel_flags.discard(new_id)

    async def stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def on_log(msg, level="info"):
            loop.call_soon_threadsafe(q.put_nowait, ("log", {"message": msg, "level": level}))

        def on_progress(val, text=""):
            loop.call_soon_threadsafe(q.put_nowait, ("progress", {"value": val, "text": text}))

        async def run():
            ok = await asyncio.to_thread(_render_session, new_id, resolution, background, subtitles, on_log, on_progress, sub_style)
            cancelled = new_id in _cancel_flags
            _cancel_flags.discard(new_id)
            _save_meta(new_id, {"status": "cancelled" if cancelled else "done"})
            loop.call_soon_threadsafe(q.put_nowait, ("done", {"success": ok, "session_id": new_id}))

        _spawn(run())

        # tell the client the new session id up front so it can show a card
        yield _sse("session_id", {"session_id": new_id, "title": title})
        while True:
            event, data = await q.get()
            yield _sse(event, data)
            if event == "done":
                break

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/sessions/delete")
async def sessions_delete_many(ids: str = Form(...)):
    count = 0
    for sid in json.loads(ids):
        p = _sess_dir(sid)
        if os.path.isdir(p):
            shutil.rmtree(p)
            count += 1
    return {"ok": True, "deleted": count}


@app.delete("/api/sessions")
def sessions_delete_all(source: str = ""):
    # delete one bucket when `source` is given, otherwise all
    buckets = [source] if source in SOURCES else list(SOURCES)
    count = 0
    for src in buckets:
        d = os.path.join(SESSIONS_DIR, src)
        if not os.path.isdir(d):
            continue
        for sid in os.listdir(d):
            p = os.path.join(d, sid)
            if os.path.isdir(p):
                shutil.rmtree(p)
                count += 1
    return {"ok": True, "deleted": count}


@app.delete("/api/sessions/{session_id}")
def session_delete(session_id: str):
    path = _sess_dir(session_id)
    if os.path.isdir(path):
        shutil.rmtree(path)
    return {"ok": True}


# ── topic generation ──────────────────────────────────────────────────────────

@app.post("/api/topics/generate")
async def topics_generate(content: str = Form(...), count: int = Form(...)):
    from groq import Groq
    client = Groq(api_key=_require_groq_key())
    prompt = f"""Generate {count} unique and compelling video script topic titles about: "{content}"

Rules:
- Each title must be concise (3-8 words)
- Make them specific, engaging, and varied
- Output ONLY the titles, one per line, no numbering, no bullets, no extra text"""

    response = await asyncio.to_thread(
        chat_with_retry,
        client,
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    lines = [l.strip() for l in response.choices[0].message.content.strip().splitlines() if l.strip()]
    return {"topics": lines[:count]}


# ── bulk ───────────────────────────────────────────────────────────────────────

BULK_TOPIC_CONCURRENCY = 2
_bulk_jobs: dict[str, list] = {}   # job_id → list of item state dicts


@app.post("/api/bulk/run")
async def bulk_run(
    items: str = Form(...),          # JSON: [{topic, duration}]
    voice: str = Form(""),
    instruction: str = Form("Gentle and calm voice tone, slow and peaceful delivery"),
    cfg: float = Form(2.0),
    denoise: bool = Form(False),
    normalize: bool = Form(False),
):
    parsed: list[dict] = json.loads(items)
    if not parsed:
        raise HTTPException(status_code=400, detail="No items provided.")

    job_id = str(uuid.uuid4())
    _bulk_jobs[job_id] = [
        {**it, "index": i, "status": "queued", "session_id": None, "error": None}
        for i, it in enumerate(parsed)
    ]
    return {"job_id": job_id}


@app.get("/api/bulk/{job_id}/stream")
async def bulk_stream(
    job_id: str,
    voice: str = "",
    instruction: str = "Gentle and calm voice tone, slow and peaceful delivery",
    cfg: float = 2.0,
    denoise: bool = False,
    normalize: bool = False,
    make_video: bool = False,
    resolution: str = "9:16",
    background: str = "",
    subtitles: bool = True,
    sub_font_size: int = 24,
    sub_color: str = "#FFFFFF",
    sub_position: str = "bottom",
    sub_outline: int = 2,
    sub_font: str = "",
):
    if job_id not in _bulk_jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    items = _bulk_jobs[job_id]

    async def stream() -> AsyncGenerator[str, None]:
        loop      = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        semaphore = asyncio.Semaphore(BULK_TOPIC_CONCURRENCY)

        def put(event: str, data: dict):
            loop.call_soon_threadsafe(queue.put_nowait, (event, data))

        async def process_item(item: dict):
            idx   = item["index"]
            topic = item["topic"]
            dur   = item["duration"]

            async with semaphore:
                # ── 1. generate script ──────────────────────────────────────
                item["status"] = "script"
                put("item_update", {"index": idx, "status": "script", "topic": topic})
                try:
                    script = await asyncio.to_thread(generate_script, _require_groq_key(), topic, dur)
                except Exception as exc:
                    item["status"] = "error"
                    item["error"]  = str(exc)
                    put("item_update", {"index": idx, "status": "error", "error": str(exc)})
                    return

                # ── 2. generate voice ───────────────────────────────────────
                session_id  = str(uuid.uuid4())
                # per-item overrides fall back to global settings
                config      = _make_config(
                    session_id,
                    source               = "bulk",
                    voice_filename       = item.get("voice") or voice,
                    control_instruction  = item.get("instruction") or instruction,
                    cfg_value            = item.get("cfg", cfg),
                    denoise              = item.get("denoise", denoise),
                    normalize            = item.get("normalize", normalize),
                )
                os.makedirs(config.session_dir, exist_ok=True)
                sections    = [s.strip() for s in script.split("#") if s.strip()]

                # save script to session
                with open(os.path.join(config.session_dir, "script.txt"), "w") as f:
                    f.write(script)

                _save_meta(session_id, {
                    "id": session_id, "title": topic,
                    "created_at": datetime.utcnow().isoformat(),
                    "steps": [1], "status": "running",
                })

                item["status"]     = "voice"
                item["session_id"] = session_id
                put("item_update", {"index": idx, "status": "voice", "session_id": session_id})

                def on_log(msg, level="info"):
                    put("item_log", {"index": idx, "message": msg, "level": level})

                def on_progress(val, text=""):
                    put("item_progress", {"index": idx, "value": val, "text": text})

                ok = await asyncio.to_thread(
                    generate_voice, sections, config, on_log, on_progress,
                    lambda: session_id in _cancel_flags,
                )
                if session_id in _cancel_flags:
                    _cancel_flags.discard(session_id)
                    item["status"] = "error"
                    item["error"]  = "Cancelled"
                    _save_meta(session_id, {"status": "cancelled"})
                    put("item_update", {"index": idx, "status": "error", "error": "Cancelled"})
                    return
                if not ok:
                    item["status"] = "error"
                    item["error"]  = "Voice generation failed"
                    _save_meta(session_id, {"status": "error"})
                    put("item_update", {"index": idx, "status": "error", "error": "Voice generation failed"})
                    return

                await asyncio.to_thread(_merge_audio, config, None)
                _save_meta(session_id, {"has_audio": True})

                # ── 3. render video (optional) ──────────────────────────────
                if make_video:
                    item["status"] = "video"
                    put("item_update", {"index": idx, "status": "video", "session_id": session_id})
                    item_bg   = item.get("background") or background
                    item_res  = item.get("resolution") or resolution
                    item_subs = item["subtitles"] if "subtitles" in item and item["subtitles"] is not None else subtitles
                    # per-item subtitle style override falls back to the global style
                    g_style   = {"font_size": sub_font_size, "color": sub_color, "position": sub_position, "outline": sub_outline, "font": sub_font}
                    o_sub     = item.get("sub")
                    sub_style = {
                        "font_size": o_sub.get("font_size", g_style["font_size"]) if o_sub else g_style["font_size"],
                        "color":     o_sub.get("color", g_style["color"])         if o_sub else g_style["color"],
                        "position":  o_sub.get("position", g_style["position"])   if o_sub else g_style["position"],
                        "outline":   o_sub.get("outline", g_style["outline"])     if o_sub else g_style["outline"],
                        "font":      o_sub.get("font", g_style["font"])           if o_sub else g_style["font"],
                    }
                    vok = await asyncio.to_thread(
                        _render_session, session_id, item_res, item_bg, item_subs, on_log, on_progress, sub_style
                    )
                    if not vok:
                        item["status"] = "error"
                        item["error"]  = "Video render failed"
                        _save_meta(session_id, {"status": "error"})
                        put("item_update", {"index": idx, "status": "error", "error": "Video render failed"})
                        return

                _save_meta(session_id, {"status": "done"})
                item["status"] = "done"
                put("item_update", {"index": idx, "status": "done", "session_id": session_id})

        # launch all items concurrently (semaphore limits to BULK_TOPIC_CONCURRENCY)
        tasks = [asyncio.create_task(process_item(item)) for item in items]

        # stream events until all done
        pending = len(tasks)
        finished = 0

        async def wait_all():
            await asyncio.gather(*tasks, return_exceptions=True)
            loop.call_soon_threadsafe(queue.put_nowait, ("__done__", {}))

        _spawn(wait_all())

        while True:
            event, data = await queue.get()
            if event == "__done__":
                success = sum(1 for it in items if it["status"] == "done")
                yield _sse("done", {"success": success, "total": len(items)})
                break
            yield _sse(event, data)

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── bulk render ────────────────────────────────────────────────────────────────

_render_jobs: dict[str, list] = {}   # job_id → list of {session_id, status}


@app.post("/api/bulk/render")
async def bulk_render_create(
    items:       str = Form(...),    # JSON: [{session_id, resolution?, background?, subtitles?}]
    resolution:  str = Form("16:9"),  # global fallbacks
    background:   str = Form(""),
    subtitles:   bool = Form(True),
    sub_font_size: int = Form(24),
    sub_color:   str = Form("#FFFFFF"),
    sub_position: str = Form("bottom"),
    sub_outline: int = Form(2),
    sub_font:    str = Form(""),
):
    parsed: list[dict] = json.loads(items)
    if not parsed:
        raise HTTPException(status_code=400, detail="No sessions selected.")
    job_id = str(uuid.uuid4())
    _render_jobs[job_id] = [
        {
            "session_id": it["session_id"],
            "resolution": it.get("resolution") or resolution,
            "background": it.get("background") or background,
            "subtitles":  it["subtitles"] if "subtitles" in it else subtitles,
            "sub_style":  {"font_size": sub_font_size, "color": sub_color, "position": sub_position, "outline": sub_outline, "font": sub_font},
            "status": "queued",
        }
        for it in parsed
    ]
    return {"job_id": job_id}


@app.get("/api/bulk/render/{job_id}/stream")
async def bulk_render_stream(job_id: str):
    if job_id not in _render_jobs:
        raise HTTPException(status_code=404, detail="Render job not found.")
    items = _render_jobs[job_id]

    async def stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        q: asyncio.Queue = asyncio.Queue()

        def put(event, data):
            loop.call_soon_threadsafe(q.put_nowait, (event, data))

        async def run_all():
            # sequential — ffmpeg is CPU-bound
            for it in items:
                sid = it["session_id"]
                # if this session was cancelled (incl. queued ones), skip it
                if sid in _cancel_flags:
                    _cancel_flags.discard(sid)
                    it["status"] = "cancelled"
                    put("item_update", {"session_id": sid, "status": "cancelled"})
                    continue

                it["status"] = "rendering"
                put("item_update", {"session_id": sid, "status": "rendering"})

                def on_log(msg, level="info", _sid=sid):
                    put("item_log", {"session_id": _sid, "message": msg, "level": level})

                def on_progress(val, text="", _sid=sid):
                    put("item_progress", {"session_id": _sid, "value": val, "text": text})

                ok = await asyncio.to_thread(
                    _render_session, sid, it["resolution"], it["background"], it["subtitles"], on_log, on_progress, it.get("sub_style")
                )
                cancelled = sid in _cancel_flags
                _cancel_flags.discard(sid)
                it["status"] = "cancelled" if cancelled else ("done" if ok else "error")
                put("item_update", {"session_id": sid, "status": it["status"]})

            put("__done__", {})

        _spawn(run_all())

        while True:
            event, data = await q.get()
            if event == "__done__":
                success = sum(1 for it in items if it["status"] == "done")
                yield _sse("done", {"success": success, "total": len(items)})
                break
            yield _sse(event, data)

    return StreamingResponse(stream(), media_type="text/event-stream")
