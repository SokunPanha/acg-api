import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from gradio_client import Client, handle_file

from .config import PipelineConfig

LogFn      = Optional[Callable[[str, str], None]]
ProgressFn = Optional[Callable[[float, str], None]]

CONCURRENCY = 3
MAX_RETRIES = 5          # the free VoxCPM Space drops requests under load — retry generously
RETRY_BACKOFF = 3.0      # seconds, grows linearly per attempt


def _generate_one(
    index: int,
    text: str,
    config: PipelineConfig,
    semaphore: threading.Semaphore,
    cancelled: Optional[Callable[[], bool]] = None,
) -> str:
    """Generate audio for one section. Returns the output file path. Retries transient failures with backoff."""
    with semaphore:
        if cancelled and cancelled():
            raise RuntimeError(f"Section {index} cancelled")
        last_exc = None
        for attempt in range(MAX_RETRIES + 1):
            if cancelled and cancelled():
                raise RuntimeError(f"Section {index} cancelled")
            try:
                client = Client("openbmb/VoxCPM-Demo")  # fresh client each attempt — avoids stale queue handles
                result = client.predict(
                    text_input=text,
                    control_instruction=config.control_instruction,
                    reference_wav_path_input=handle_file(config.voice_ref),
                    use_prompt_text=False,
                    prompt_text_input="",
                    cfg_value_input=config.cfg_value,
                    do_normalize=config.normalize,
                    denoise=config.denoise,
                    api_name="/generate",
                )
                out_path = os.path.join(config.mp3_dir, f"{index}.mp3")
                shutil.move(result, out_path)
                return out_path
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    # sleep in 0.5s chunks so a cancel is picked up quickly
                    waited = 0.0
                    delay  = RETRY_BACKOFF * (attempt + 1)   # 3s, 6s, 9s, …
                    while waited < delay:
                        if cancelled and cancelled():
                            raise RuntimeError(f"Section {index} cancelled")
                        time.sleep(0.5)
                        waited += 0.5
        raise RuntimeError(f"Section {index} failed after {MAX_RETRIES + 1} attempts: {last_exc}")


def generate_voice(
    sections: list[str],
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

    def cancelled() -> bool:
        return bool(should_cancel and should_cancel())

    if not sections:
        log("Script is empty.", "error")
        return False

    os.makedirs(config.mp3_dir, exist_ok=True)
    total = len(sections)
    log(f"Generating {total} section(s) with concurrency={CONCURRENCY}…", "info")

    semaphore = threading.Semaphore(CONCURRENCY)
    completed = 0
    failed    = False
    lock      = threading.Lock()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {
            pool.submit(_generate_one, i + 1, text, config, semaphore, cancelled): i + 1
            for i, text in enumerate(sections)
        }

        for future in as_completed(futures):
            if cancelled():
                log("Cancelled by user.", "warn")
                pool.shutdown(wait=False, cancel_futures=True)
                return False
            idx = futures[future]
            try:
                future.result()
                with lock:
                    completed += 1
                    log(f"Section {idx}/{total} done", "ok")
                    progress(completed / total * 0.33, f"Audio {completed}/{total}")
            except Exception as exc:
                log(str(exc), "error")
                with lock:
                    failed = True

    if cancelled():
        log("Cancelled by user.", "warn")
        return False
    if failed:
        return False

    log("Voice generation complete!", "ok")
    return True
