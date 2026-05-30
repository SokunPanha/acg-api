import re
import time


def chat_with_retry(client, *, max_retries: int = 6, **kwargs):
    """
    Call Groq chat.completions.create, retrying on 429 (rate limit / TPM).

    Groq's free tier caps tokens-per-minute; bulk runs hit it constantly. The
    error message carries a "try again in Xs" hint — we honor it, otherwise back off.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:  # groq.RateLimitError or generic with status 429
            msg = str(exc)
            status = getattr(exc, "status_code", None)
            is_rate_limit = status == 429 or "rate_limit" in msg or "Rate limit" in msg
            if not is_rate_limit or attempt >= max_retries:
                raise
            last_exc = exc
            # parse the suggested wait ("try again in 590ms" / "in 12.3s"), else back off
            wait = _suggested_wait(msg)
            if wait is None:
                wait = min(20.0, 2.0 * (attempt + 1))
            time.sleep(wait + 0.25)   # small cushion
    raise last_exc  # pragma: no cover


def _suggested_wait(msg: str) -> float | None:
    m = re.search(r"try again in\s+([\d.]+)\s*(ms|s)", msg, re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    return val / 1000.0 if m.group(2).lower() == "ms" else val
