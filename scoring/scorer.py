"""Adaptive scoring utilities used by all search strategies.

Score normalization here uses percentile ranking rather than fixed
score ranges so it works across BM25, cross-encoder, and dot-product
distributions without per-strategy tuning.
"""

import numpy as np
from typing import Dict, List, Optional, Any


class UnifiedScorer:

    # Cap on how much metadata signals can boost a base relevance score.
    # Keeps boosts from dominating the underlying ranker.
    MAX_METADATA_BOOST = 0.15

    def normalize_distance(self, distance: float) -> float:
        """ChromaDB cosine distance [0, 2] -> similarity [0, 1]."""
        return max(0.0, min(1.0, 1.0 - (distance / 2.0)))

    def normalize_bm25(self, scores: np.ndarray) -> np.ndarray:
        """Percentile-rank BM25 scores into [0.2, 1.0].

        Floor at 0.2 so low-but-nonzero matches still contribute to fusion.
        """
        return self._percentile_normalize(scores, low=0.2, high=1.0, uniform_value=0.5)

    def normalize_cross_encoder_scores(self, scores: np.ndarray) -> np.ndarray:
        """Percentile-rank cross-encoder scores into [0.3, 1.0].

        Higher floor than BM25: rerank candidates were already top-K filtered.
        """
        return self._percentile_normalize(scores, low=0.3, high=1.0, uniform_value=0.6)

    def calculate_metadata_boost(
        self,
        metadata: Dict[str, Any],
        query: Optional[str] = None,
        result_pool: Optional[List[Dict]] = None,
    ) -> float:
        """Compute additive boost from metadata signals, capped at MAX_METADATA_BOOST.

        Two signals contribute:
          1. Quality score percentile (relative to result pool when provided).
          2. Query/title and query/keyword overlap.
        """
        boost = 0.0

        quality_score = float(metadata.get('quality_score', 0) or 0)

        if result_pool and len(result_pool) > 1:
            qualities = [
                float(r.get('metadata', {}).get('quality_score', 0) or 0)
                for r in result_pool
            ]
            quality_percentile = self._calculate_percentile(quality_score, qualities)
            if quality_percentile >= 0.75:
                boost += 0.08
            elif quality_percentile >= 0.5:
                boost += 0.05
        else:
            # Fallback when no pool is available — use absolute thresholds.
            if quality_score > 0.8:
                boost += 0.08
            elif quality_score > 0.6:
                boost += 0.05

        if query:
            boost += self._calculate_query_relevance(metadata, query)

        return min(boost, self.MAX_METADATA_BOOST)

    def compute_final_score(
        self,
        base_score: float,
        metadata: Dict[str, Any],
        query: Optional[str] = None,
        result_pool: Optional[List[Dict]] = None,
    ) -> float:
        """Apply metadata boost to a base score and clip to [0, 1]."""
        boost = self.calculate_metadata_boost(metadata, query, result_pool)
        return min(base_score + boost, 1.0)

    def _percentile_normalize(
        self,
        scores: np.ndarray,
        low: float,
        high: float,
        uniform_value: float,
    ) -> np.ndarray:
        """Map raw scores to [low, high] by percentile rank.

        Ties resolve in argsort order — acceptable because callers only use
        relative ordering, not absolute equality.
        """
        scores = np.asarray(scores, dtype=float)
        n = len(scores)
        if n == 0:
            return scores
        if np.std(scores) < 1e-10:
            return np.full_like(scores, uniform_value)

        denom = max(n - 1, 1)
        percentiles = np.empty_like(scores)
        for rank, idx in enumerate(np.argsort(scores)):
            percentiles[idx] = rank / denom

        normalized = low + percentiles * (high - low)
        return np.clip(normalized, 0.0, 1.0)

    def _calculate_percentile(self, value: float, values: List[float]) -> float:
        if not values or len(values) == 1:
            return 0.5
        rank = sum(1 for v in values if v <= value)
        return rank / len(values)

    def _calculate_query_relevance(self, metadata: Dict[str, Any], query: str) -> float:
        boost = 0.0
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.0

        title = str(metadata.get('title', '')).lower()
        if title:
            title_terms = set(title.split())
            overlap = len(query_terms & title_terms)
            if overlap:
                overlap_ratio = overlap / len(query_terms)
                boost += min(overlap_ratio * 0.05, 0.05)

        keywords = metadata.get('keywords', [])
        if keywords:
            keyword_set = {str(kw).lower() for kw in keywords}
            if query_terms & keyword_set:
                boost += 0.02

        return boost
