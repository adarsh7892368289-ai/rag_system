"""Multi-strategy document chunking.

Each document is chunked four different ways in parallel — each strategy is
tuned for a different retrieval pattern (sentence-aware for QA, semantic for
topic queries, paragraph for long-form, fixed-size as a baseline). Downstream
search fuses results across strategies via RRF.
"""

import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List

from config.settings import CHUNKING
from utils.helpers import calculate_similarity, split_sentences


# Heuristic for filtering short noise sentences from chunks.
_MIN_SENTENCE_WORDS = 3

# Per-chunk semantic-mode hard cap on sentences before forcing a split.
# Without this, a long monotonic document collapses into a single chunk.
_SEMANTIC_MAX_SENTENCES = 5


@dataclass
class Chunk:
    text: str
    index: int
    char_count: int
    word_count: int
    metadata: Dict = field(default_factory=dict)


class AdaptiveKeywordExtractor:
    """TF-IDF-style keyword extractor with no external corpus.

    IDF is computed against the document itself (single-doc TF-IDF), which is
    a deliberate choice: we want keywords that *characterize* this chunk, not
    keywords that distinguish it from a corpus we don't have at chunking time.
    """

    _STOPWORD_PATTERN = re.compile(
        r'\b(the|a|an|and|or|but|in|on|at|to|for|of|with|by|'
        r'is|are|was|were|be|been|being|have|has|had|'
        r'this|that|these|those|what|which|who|when|where|why|how)\b',
        re.IGNORECASE,
    )
    _WORD_PATTERN = re.compile(r'\b[a-zA-Z]{4,}\b')

    def extract(self, text: str, top_k: int = 5) -> List[str]:
        words = self._WORD_PATTERN.findall(text.lower())
        if not words:
            return []

        word_freq = Counter(words)
        total_words = len(words)

        scored = []
        for word, count in word_freq.items():
            if self._STOPWORD_PATTERN.fullmatch(word):
                continue

            tf = count / total_words
            idf = math.log(total_words / count + 1)
            length_bonus = min(len(word) / 15, 1.0)

            # Weights tuned empirically for English prose; keep them adjustable
            # at this level rather than promoting to config.
            score = 0.4 * tf * 100 + 0.4 * idf + 0.2 * length_bonus
            scored.append((word, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in scored[:top_k]]


class ChunkQualityScorer:
    """Score chunks 0–1 on four signals: length, keyword density, coherence, info content.

    Boilerplate (nav links, footers, copyright lines) is detected and zeroed
    out so it never makes it into search results.
    """

    _NAV_PATTERNS = (
        re.compile(r'jump to (content|navigation|main)', re.IGNORECASE),
        re.compile(r'skip to', re.IGNORECASE),
        re.compile(r'(menu|navigation|footer|header)\s*$', re.IGNORECASE),
        re.compile(r'^\s*(home|about|contact|search|login|sign up)\s*$', re.IGNORECASE),
        re.compile(r'copyright.*all rights reserved', re.IGNORECASE),
        re.compile(r'(privacy policy|terms of service|cookie)', re.IGNORECASE),
        re.compile(r'^\s*\d+\s*$'),
    )

    def __init__(self):
        self._keyword_extractor = AdaptiveKeywordExtractor()

    def score_chunk(self, chunk: Chunk, document_stats: Dict = None) -> float:
        if self._is_boilerplate(chunk.text):
            return 0.0

        if document_stats is None:
            document_stats = {'avg_chunk_words': 300, 'std_chunk_words': 100}

        length_score = self._score_length(
            chunk.word_count,
            document_stats.get('avg_chunk_words', 300),
            document_stats.get('std_chunk_words', 100),
        )
        keyword_score = self._score_keyword_density(chunk)
        coherence_score = self._score_coherence(chunk.text)
        info_score = self._score_information(chunk.text)

        quality = (
            0.30 * length_score
            + 0.25 * keyword_score
            + 0.25 * coherence_score
            + 0.20 * info_score
        )
        return round(quality, 3)

    def _score_length(self, word_count: int, avg: float, std: float) -> float:
        if std == 0:
            return 1.0 if word_count > 50 else 0.3

        z = abs((word_count - avg) / std)
        if z <= 1:
            return 1.0 - (z * 0.2)
        if z <= 2:
            return 0.8 - ((z - 1) * 0.3)
        return max(0.3, 0.5 - ((z - 2) * 0.1))

    def _score_keyword_density(self, chunk: Chunk) -> float:
        if chunk.word_count == 0:
            return 0.0

        keywords = chunk.metadata.get('keywords', [])
        density = len(keywords) / chunk.word_count

        # Optimal density tapers off in longer chunks: a 1000-word chunk
        # naturally has more diluted keywords than a 100-word one.
        if chunk.word_count < 100:
            optimal = 0.08
        elif chunk.word_count < 300:
            optimal = 0.05
        else:
            optimal = 0.03

        ratio = density / optimal if optimal > 0 else 0
        if 0.8 <= ratio <= 1.2:
            return 1.0
        if ratio < 0.5:
            return 0.3
        return max(0.5, 1.0 - abs(1.0 - ratio) * 0.5)

    def _score_coherence(self, text: str) -> float:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if len(sentences) < 2:
            return 0.6

        lengths = [len(s.split()) for s in sentences]
        avg = sum(lengths) / len(lengths)
        if avg <= 0:
            return 0.5

        variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
        std = math.sqrt(variance)
        consistency = 1.0 / (1.0 + std / avg)

        # Penalize chunks dominated by sentence fragments (under 5 words).
        fragment_ratio = sum(1 for l in lengths if l < 5) / len(lengths)
        return max(0.3, consistency - fragment_ratio * 0.3)

    def _score_information(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0

        unique_ratio = len(set(words)) / len(words)
        diversity = min(unique_ratio * 1.5, 1.0)

        numeric_count = len(re.findall(r'\b\d+\b', text))
        numeric_score = min(numeric_count / len(words), 0.2) * 5

        entities = re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', text)
        entity_score = min(len(entities) / len(words), 0.15) * 6.67

        return 0.5 * diversity + 0.25 * numeric_score + 0.25 * entity_score

    def _is_boilerplate(self, text: str) -> bool:
        for pattern in self._NAV_PATTERNS:
            if pattern.search(text):
                return True

        # Very short text with all-unique words is almost always navigation.
        words = text.split()
        if 0 < len(words) < 20 and len(set(words)) / len(words) > 0.9:
            return True

        return False


class EnsembleChunker:
    """Run all four chunking strategies in parallel and tag each chunk.

    Strategies are independent — failures in one (e.g., a regex blowup on a
    pathological doc) don't poison the others; we log and return empty for
    that strategy only.
    """

    _STRATEGY_NAMES = ('sentence_aware', 'semantic', 'paragraph', 'fixed_size')

    def __init__(self):
        self._config = CHUNKING
        self._quality_scorer = ChunkQualityScorer()
        self._keyword_extractor = AdaptiveKeywordExtractor()

    def chunk_all_strategies(self, text: str) -> Dict[str, List[Chunk]]:
        if not text or len(text.strip()) < 10:
            return {name: [] for name in self._STRATEGY_NAMES}

        strategies = {
            'sentence_aware': self._sentence_aware_chunk,
            'semantic': self._semantic_chunk,
            'paragraph': self._paragraph_chunk,
            'fixed_size': self._fixed_size_chunk,
        }

        results: Dict[str, List[Chunk]] = {}
        with ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            futures = {
                executor.submit(fn, text): name
                for name, fn in strategies.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    chunks = future.result()
                    for chunk in chunks:
                        chunk.metadata['chunking_strategy'] = name
                    results[name] = chunks
                except Exception as e:
                    print(f"   ⚠️  {name} chunking failed: {e}")
                    results[name] = []

        for name, chunks in results.items():
            if not chunks:
                continue
            stats = self._calculate_document_stats(chunks)
            for chunk in chunks:
                chunk.metadata['quality_score'] = self._quality_scorer.score_chunk(
                    chunk, stats
                )

        return results

    def _calculate_document_stats(self, chunks: List[Chunk]) -> Dict:
        word_counts = [c.word_count for c in chunks]
        if not word_counts:
            return {'avg_chunk_words': 300, 'std_chunk_words': 100}

        avg = sum(word_counts) / len(word_counts)
        variance = sum((wc - avg) ** 2 for wc in word_counts) / len(word_counts)
        std = math.sqrt(variance)

        # Floor std at 50 to keep length scoring well-conditioned for very
        # uniform documents (otherwise z-scores explode).
        return {
            'avg_chunk_words': avg,
            'std_chunk_words': max(std, 50.0),
            'total_chunks': len(chunks),
        }

    def _sentence_aware_chunk(self, text: str) -> List[Chunk]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current: List[str] = []
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())
            if sentence_words < _MIN_SENTENCE_WORDS:
                continue

            # Flush when adding this sentence would push us over the target.
            if current and current_words + sentence_words > self._config.target_words:
                chunks.append(self._create_chunk(' '.join(current), len(chunks)))
                overlap = self._tail_within_word_budget(current, self._config.overlap_words)
                current = overlap + [sentence]
                current_words = sum(len(s.split()) for s in current)
            else:
                current.append(sentence)
                current_words += sentence_words

        if current and current_words >= self._config.min_chunk_words:
            chunks.append(self._create_chunk(' '.join(current), len(chunks)))

        return chunks

    def _semantic_chunk(self, text: str) -> List[Chunk]:
        sentences = split_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current: List[str] = []

        for sentence in sentences:
            if not current:
                current.append(sentence)
                continue

            similarity = calculate_similarity(current[-1], sentence)
            should_split = (
                similarity < self._config.similarity_threshold
                or len(current) >= _SEMANTIC_MAX_SENTENCES
                or sum(len(s.split()) for s in current) > self._config.target_words
            )

            if should_split:
                chunks.append(self._create_chunk(' '.join(current), len(chunks)))
                current = [sentence]
            else:
                current.append(sentence)

        if current:
            chunks.append(self._create_chunk(' '.join(current), len(chunks)))

        return chunks

    def _paragraph_chunk(self, text: str) -> List[Chunk]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            return []

        chunks: List[Chunk] = []
        current: List[str] = []
        current_words = 0

        for para in paragraphs:
            para_words = len(para.split())
            if current and current_words + para_words > self._config.target_words:
                chunks.append(self._create_chunk('\n\n'.join(current), len(chunks)))
                current = [para]
                current_words = para_words
            else:
                current.append(para)
                current_words += para_words

        if current:
            chunks.append(self._create_chunk('\n\n'.join(current), len(chunks)))

        return chunks

    def _fixed_size_chunk(self, text: str) -> List[Chunk]:
        words = text.split()
        if not words:
            return []

        target = self._config.target_words
        overlap = self._config.overlap_words

        # Guard against pathological config that would cause an infinite loop
        # or zero forward progress.
        step = max(target - overlap, 1)

        chunks: List[Chunk] = []
        for i in range(0, len(words), step):
            window = words[i : i + target]
            if len(window) >= self._config.min_chunk_words:
                chunks.append(self._create_chunk(' '.join(window), len(chunks)))

        return chunks

    def _create_chunk(self, text: str, index: int) -> Chunk:
        keywords = self._keyword_extractor.extract(text, top_k=5)
        sentence_count = sum(1 for s in re.split(r'[.!?]+', text) if s.strip())

        return Chunk(
            text=text,
            index=index,
            char_count=len(text),
            word_count=len(text.split()),
            metadata={
                'keywords': keywords,
                'sentence_count': sentence_count,
            },
        )

    def _tail_within_word_budget(self, sentences: List[str], budget: int) -> List[str]:
        """Return the suffix of `sentences` whose word count fits in `budget`.

        Used to carry context into the next chunk so chunk boundaries don't
        sever sentences from their referents.
        """
        tail: List[str] = []
        used = 0
        for sentence in reversed(sentences):
            sentence_words = len(sentence.split())
            if used + sentence_words > budget:
                break
            tail.insert(0, sentence)
            used += sentence_words
        return tail
