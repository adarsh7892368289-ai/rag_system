"""Generic text utilities shared across the pipeline."""

import re
from typing import List


# Abbreviations that contain a period and would otherwise trip the
# sentence splitter. Order matters only for readability.
_ABBREV_PATTERN = re.compile(
    r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|etc|vs|e\.g|i\.e)\.',
    flags=re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
_DOT_PLACEHOLDER = '\x00DOT\x00'  # NUL-delimited so it cannot occur in source text.


def estimate_tokens(text: str) -> int:
    """Rough token estimate for budgeting LLM context windows.

    1 token ≈ 0.75 words for English, so we use 1.3 as the inverse multiplier.
    Good enough for sizing; not a substitute for the model's tokenizer.
    """
    return int(len(text.split()) * 1.3)


def clean_text(text: str) -> str:
    """Collapse whitespace and drop non-ASCII characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving common abbreviations.

    Uses a placeholder swap so periods in tokens like "Dr." don't end a sentence.
    """
    if not text:
        return []

    masked = _ABBREV_PATTERN.sub(
        lambda m: m.group(0).replace('.', _DOT_PLACEHOLDER), text
    )
    sentences = _SENTENCE_SPLIT.split(masked)
    return [s.replace(_DOT_PLACEHOLDER, '.').strip() for s in sentences if s.strip()]


def calculate_similarity(text1: str, text2: str) -> float:
    """Jaccard similarity over word sets — cheap, no embeddings needed."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0
