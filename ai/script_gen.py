import math
import re

from .groq_client import chat_with_retry

WORDS_PER_MINUTE = 145
MAX_WORDS_PER_SECTION = 200
BATCH_SECTIONS = 8          # sections generated per LLM call (keeps each call within output limits)
MAX_BATCHES = 30            # safety cap so we never loop forever


def _word_count(sections: list[str]) -> int:
    return sum(len(s.split()) for s in sections)


def calc_sections(duration_minutes: int) -> tuple[int, int]:
    """Returns (num_sections, words_per_section)."""
    total_words = max(1, duration_minutes) * WORDS_PER_MINUTE
    num_sections = math.ceil(total_words / MAX_WORDS_PER_SECTION)
    words_per_section = total_words // num_sections
    return num_sections, words_per_section


def generate_script(api_key: str, topic: str, duration_minutes: int) -> str:
    """
    Generate a script with # section separators.

    A single LLM completion can't produce a long (e.g. 60-minute / ~8700-word)
    script — the model stops after a couple thousand words. So we generate in
    batches of BATCH_SECTIONS and stitch them together, giving each call the tail
    of the previous output for continuity.
    """
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("groq not installed. Run: pip install groq")

    client = Groq(api_key=api_key)
    _, words_per_section = calc_sections(duration_minutes)
    target_words = max(1, duration_minutes) * WORDS_PER_MINUTE

    sections: list[str] = []
    batches = 0
    while _word_count(sections) < target_words and batches < MAX_BATCHES:
        batches += 1
        # size this batch to what's still needed, so we don't overshoot the target
        remaining_words = target_words - _word_count(sections)
        batch = max(1, min(BATCH_SECTIONS, math.ceil(remaining_words / words_per_section)))

        position = (
            "This is the OPENING of the script." if not sections
            else "Continue the script seamlessly from where it left off; "
                 "do NOT restart or re-introduce the topic."
        )
        continuity = ""
        if sections:
            tail = sections[-1][-400:]
            continuity = f'\n\nThe script so far ends with:\n"""{tail}"""\nContinue naturally without repeating it.'

        # this batch is expected to finish the script
        is_final = batch < BATCH_SECTIONS or remaining_words <= BATCH_SECTIONS * words_per_section
        ending = (
            " The final section should deliver a powerful closing call-to-action."
            if is_final else " Do NOT conclude or wrap up yet — the script continues after this."
        )

        min_words = max(1, words_per_section - 15)
        prompt = f"""You are writing one long {duration_minutes}-minute motivational video script about: "{topic}".
{position}

Write exactly {batch} section(s). Rules:
- LENGTH IS CRITICAL: each section must be a FULL passage of {min_words}–{MAX_WORDS_PER_SECTION} words. Never write short or terse sections — expand every idea with vivid imagery, concrete examples, stories, and rhetorical repetition until it reaches the word count.
- Separate sections with a single line containing only the character: #
- Powerful motivational speaking style — speak directly to the viewer.
- No headers, titles, numbering, or labels of any kind.{ending}
- Output ONLY the raw script text with # separators, nothing else.{continuity}"""

        response = chat_with_retry(
            client,
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.9,
        )
        text = response.choices[0].message.content.strip()
        parts = [p.strip() for p in text.split("#") if p.strip()]
        if not parts:
            break  # model returned nothing usable — stop rather than loop
        sections.extend(parts)

    script = "\n#\n".join(sections)
    # tidy stray separators
    script = re.sub(r"\s*\n\s*#\s*\n\s*", "\n#\n", script)
    script = re.sub(r"\s*#\s*$", "", script).strip()
    return script
