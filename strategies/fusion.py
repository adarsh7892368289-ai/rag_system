"""Reciprocal Rank Fusion + MMR diversification.

RRF reference: Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods" (SIGIR 2009).

The k constant in RRF (60) flattens contribution from low-ranked documents;
this is the value from the original paper and is widely used as-is.
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


# Standard RRF constant from the original paper. Higher k = flatter weighting.
_RRF_K = 60

# MMR balance: 0.7 keeps relevance dominant while still penalizing redundancy.
_MMR_LAMBDA = 0.7


class ResultFusion:

    def __init__(self, embedding_generator=None):
        # Optional — only needed for diversity / dedup. RRF works without it.
        self._embedding_generator = embedding_generator

    def reciprocal_rank_fusion(
        self,
        strategy_results: Dict[str, List[Dict]],
        top_n: int = 5,
    ) -> List[Dict]:
        """Merge ranked lists from multiple strategies using RRF + MMR.

        Each document's RRF score is the sum of 1/(k + rank) across strategies
        that ranked it. Confidence is derived from how many strategies agreed
        and how consistent their rankings were.
        """
        if not strategy_results:
            return []

        doc_scores: Dict[str, Dict] = defaultdict(self._fresh_doc_entry)

        for strategy_name, results in strategy_results.items():
            for rank, result in enumerate(results, start=1):
                doc_id = result['id']
                entry = doc_scores[doc_id]

                entry['rrf_score'] += 1.0 / (_RRF_K + rank)
                entry['strategies'].append(strategy_name)
                entry['strategy_scores'][strategy_name] = result.get('final_score', 0)
                entry['strategy_ranks'][strategy_name] = rank

                if entry['result_data'] is None:
                    entry['result_data'] = result

        num_strategies = len(strategy_results)
        # Theoretical RRF max: every strategy ranks the doc #1.
        max_possible_rrf = num_strategies / (_RRF_K + 1)

        fused_results: List[Dict] = []
        for entry in doc_scores.values():
            result = entry['result_data'].copy()
            result['fusion_score'] = entry['rrf_score']
            result['strategies_used'] = entry['strategies']
            result['strategy_scores'] = entry['strategy_scores']
            result['strategy_ranks'] = entry['strategy_ranks']
            result['confidence'] = self._calculate_confidence(entry, num_strategies)

            if max_possible_rrf > 0:
                result['final_score'] = min(entry['rrf_score'] / max_possible_rrf, 1.0)
            else:
                result['final_score'] = 0.0

            fused_results.append(result)

        fused_results.sort(key=lambda r: r['fusion_score'], reverse=True)
        return self._apply_diversity(fused_results, top_n)[:top_n]

    def deduplicate_results(
        self, results: List[Dict], threshold: float = 0.95
    ) -> List[Dict]:
        """Drop near-duplicate results by embedding cosine similarity."""
        if not results or not self._embedding_generator:
            return results

        contents = [r['content'] for r in results]
        embeddings = self._embedding_generator.encode_batch(contents, show_progress=False)
        if embeddings.size == 0:
            return results

        unique_results: List[Dict] = []
        unique_embeddings: List[np.ndarray] = []

        for result, embedding in zip(results, embeddings):
            if any(
                self._cosine_similarity(embedding, ue) > threshold
                for ue in unique_embeddings
            ):
                continue
            unique_results.append(result)
            unique_embeddings.append(embedding)

        return unique_results

    # ------------------------------------------------------------------ private

    @staticmethod
    def _fresh_doc_entry() -> Dict:
        return {
            'rrf_score': 0.0,
            'strategies': [],
            'strategy_scores': {},
            'strategy_ranks': {},
            'result_data': None,
        }

    def _calculate_confidence(self, entry: Dict, num_strategies: int) -> float:
        """Three-component confidence: coverage, average rank quality, rank agreement."""
        ranks = entry['strategy_ranks']
        if not ranks:
            return 0.0

        coverage = (
            min(len(entry['strategies']) / num_strategies, 1.0)
            if num_strategies > 0
            else 0.0
        )

        # Quality: 1.0 at rank 1, decays smoothly with worse ranks.
        avg_rank = sum(ranks.values()) / len(ranks)
        quality = 1.0 / (1.0 + avg_rank / 10)

        if len(ranks) > 1:
            mean_rank = avg_rank
            variance = sum((r - mean_rank) ** 2 for r in ranks.values()) / len(ranks)
            std_dev = variance ** 0.5
            agreement = 1.0 / (1.0 + std_dev / 5)
        else:
            # Single strategy gets a perfect "agreement" value but is penalized
            # via the coverage component, so this isn't a free pass.
            agreement = 1.0

        confidence = 0.40 * coverage + 0.30 * quality + 0.30 * agreement
        return min(confidence, 1.0)

    def _apply_diversity(self, results: List[Dict], top_n: int) -> List[Dict]:
        """MMR diversification over already-RRF-ranked results.

        Skip if we have fewer results than requested or no embedder available.
        """
        if len(results) <= top_n or not self._embedding_generator:
            return results

        contents = [r['content'] for r in results]
        embeddings = self._embedding_generator.encode_batch(contents, show_progress=False)
        if embeddings.size == 0:
            return results

        selected_indices = [0]
        selected_embeddings = [embeddings[0]]
        remaining = list(range(1, len(results)))

        while len(selected_indices) < top_n and remaining:
            best_score = -float('inf')
            best_idx: Optional[int] = None

            for idx in remaining:
                relevance = results[idx]['fusion_score']
                similarities = self._cosine_similarity_batch(
                    embeddings[idx], np.array(selected_embeddings)
                )
                max_sim = float(np.max(similarities))
                mmr = _MMR_LAMBDA * relevance - (1 - _MMR_LAMBDA) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = idx

            if best_idx is None:
                break
            selected_indices.append(best_idx)
            selected_embeddings.append(embeddings[best_idx])
            remaining.remove(best_idx)

        return [results[i] for i in selected_indices]

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = float(np.linalg.norm(vec1))
        norm2 = float(np.linalg.norm(vec2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    @staticmethod
    def _cosine_similarity_batch(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Cosine similarity between a 1-D vector and each row of a 2-D matrix."""
        dot = matrix @ vec
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec)
        # Avoid divide-by-zero for zero vectors.
        norms = np.where(norms == 0, 1e-10, norms)
        return dot / norms
