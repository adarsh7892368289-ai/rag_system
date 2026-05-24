"""Composable search strategies over the multi-collection store.

We expose four primitives — semantic, BM25, hybrid, MMR, rerank — and two
fan-out orchestrators. The orchestrators run primitives in parallel against
multiple chunking strategies and return per-chunking-strategy result lists
ready for fusion.
"""

import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import SEARCH
from scoring.scorer import UnifiedScorer
from strategies.fusion import ResultFusion


logger = logging.getLogger(__name__)


_CHUNKING_STRATEGIES = ('sentence_aware', 'semantic', 'paragraph', 'fixed_size')

# Diversity filter threshold: drop a result if its Jaccard word overlap with an
# already-selected result exceeds this.
_DIVERSITY_THRESHOLD = 0.85

# Per-strategy timeout (seconds) for a single search call.
_SEARCH_TIMEOUT = 30
# Outer-orchestrator timeout when fanning out across all chunking strategies.
_PARALLEL_SEARCH_TIMEOUT = 60


class AdvancedSearchStrategies:

    def __init__(self, db_manager):
        self._db_manager = db_manager
        self._scorer = UnifiedScorer()
        self._fusion = ResultFusion(db_manager.embedding_model)

        # Lazy-loaded: cross-encoder is ~120MB and not needed unless rerank runs.
        self._cross_encoder: Optional[CrossEncoder] = None
        self._cross_encoder_lock = threading.Lock()

        # Per-chunking-strategy BM25 indexes, keyed by content fingerprint so
        # we rebuild only when the underlying corpus actually changes.
        self._bm25_indices: Dict[str, BM25Okapi] = {}
        self._bm25_corpus_fingerprints: Dict[str, str] = {}
        self._bm25_lock = threading.Lock()

    # ----------------------------------------------------------------- internals

    @property
    def cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None:
            with self._cross_encoder_lock:
                if self._cross_encoder is None:
                    self._cross_encoder = CrossEncoder(SEARCH.rerank_model)
        return self._cross_encoder

    def get_bm25(self, strategy: str) -> Optional[BM25Okapi]:
        """Return a BM25 index for the strategy, rebuilding only on corpus change."""
        cache = self._db_manager.documents_cache.get(strategy, [])
        fingerprint = self._corpus_fingerprint(cache)

        with self._bm25_lock:
            if (
                strategy in self._bm25_indices
                and self._bm25_corpus_fingerprints.get(strategy) == fingerprint
            ):
                return self._bm25_indices[strategy]

            if not cache:
                return None

            tokenized = [doc['content'].lower().split() for doc in cache]
            self._bm25_indices[strategy] = BM25Okapi(tokenized)
            self._bm25_corpus_fingerprints[strategy] = fingerprint
            return self._bm25_indices[strategy]

    def invalidate_caches(self):
        with self._bm25_lock:
            self._bm25_indices.clear()
            self._bm25_corpus_fingerprints.clear()

    @staticmethod
    def _corpus_fingerprint(cache: List[Dict]) -> str:
        """Cheap content-aware fingerprint: hash of (count, joined ids).

        Catches both growth and replacement; doesn't hash full content because
        that's O(N) and ids already change on content updates.
        """
        if not cache:
            return 'empty'
        joined = f"{len(cache)}|" + '|'.join(str(d.get('id', '')) for d in cache)
        return hashlib.md5(joined.encode('utf-8')).hexdigest()

    # ---------------------------------------------------------------- primitives

    def semantic_search(
        self, query: str, strategy: str, n_results: int = 5
    ) -> List[Dict]:
        # Over-fetch so the diversity filter has slack to drop near-duplicates.
        raw_results = self._db_manager.search(query, strategy, n_results=n_results * 2)

        for result in raw_results:
            result['final_score'] = self._scorer.compute_final_score(
                base_score=result['similarity_score'],
                metadata=result['metadata'],
                query=query,
            )
            result['search_strategy'] = 'semantic'

        raw_results.sort(key=lambda r: r['final_score'], reverse=True)
        return self._diversify_results(raw_results)[:n_results]

    def bm25_search(
        self, query: str, strategy: str, n_results: int = 5
    ) -> List[Dict]:
        cache = self._db_manager.documents_cache.get(strategy, [])
        if not cache:
            return []

        bm25 = self.get_bm25(strategy)
        if bm25 is None:
            return []

        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        normalized_scores = self._scorer.normalize_bm25(bm25_scores)
        top_indices = np.argsort(bm25_scores)[::-1][: n_results * 2]

        results: List[Dict] = []
        for idx in top_indices:
            doc = cache[idx]
            base_score = float(normalized_scores[idx])
            final_score = self._scorer.compute_final_score(
                base_score=base_score,
                metadata=doc.get('metadata', {}),
                query=query,
            )
            results.append({
                'id': doc['id'],
                'content': doc['content'],
                'metadata': doc.get('metadata', {}),
                'bm25_score': base_score,
                'final_score': final_score,
                'query': query,
                'chunking_strategy': strategy,
                'search_strategy': 'bm25',
            })

        results.sort(key=lambda r: r['final_score'], reverse=True)
        return self._diversify_results(results)[:n_results]

    def hybrid_search(
        self,
        query: str,
        strategy: str,
        n_results: int = 5,
        alpha: Optional[float] = None,
    ) -> List[Dict]:
        """Linear combination of semantic + BM25 scores.

        alpha weights the vector score; (1 - alpha) goes to BM25.
        """
        if alpha is None:
            alpha = SEARCH.hybrid_alpha

        cache = self._db_manager.documents_cache.get(strategy, [])
        if not cache:
            return []

        # Pull more vector candidates than we need so we have a wider pool to
        # combine with BM25 scores below.
        vector_results = self._db_manager.search(query, strategy, n_results=n_results * 3)
        if not vector_results:
            return []

        bm25 = self.get_bm25(strategy)
        if bm25 is None:
            return vector_results[:n_results]

        # Build id -> cache index once instead of doing O(n) lookup per result.
        id_to_index = {doc['id']: i for i, doc in enumerate(cache)}

        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        normalized_bm25 = self._scorer.normalize_bm25(bm25_scores)

        hybrid_results: List[Dict] = []
        for result in vector_results:
            idx = id_to_index.get(result['id'])
            if idx is None:
                continue

            bm25_score = float(normalized_bm25[idx])
            vector_score = result['similarity_score']
            hybrid_base = alpha * vector_score + (1 - alpha) * bm25_score

            final_score = self._scorer.compute_final_score(
                base_score=hybrid_base,
                metadata=result['metadata'],
                query=query,
            )

            result.update({
                'bm25_score': bm25_score,
                'vector_score': vector_score,
                'hybrid_score': hybrid_base,
                'final_score': final_score,
                'search_strategy': 'hybrid',
            })
            hybrid_results.append(result)

        hybrid_results.sort(key=lambda r: r['final_score'], reverse=True)
        return self._diversify_results(hybrid_results)[:n_results]

    def mmr_search(
        self,
        query: str,
        strategy: str,
        n_results: int = 5,
        lambda_param: Optional[float] = None,
    ) -> List[Dict]:
        """Maximal Marginal Relevance: rerank for relevance + diversity."""
        if lambda_param is None:
            lambda_param = SEARCH.mmr_lambda

        candidate_count = min(
            n_results * SEARCH.mmr_candidates_multiplier, SEARCH.max_top_k
        )
        candidates = self._db_manager.search(query, strategy, n_results=candidate_count)
        if not candidates:
            return []

        for c in candidates:
            c['base_score'] = self._scorer.compute_final_score(
                base_score=c['similarity_score'],
                metadata=c['metadata'],
                query=query,
            )
            c['search_strategy'] = 'mmr'

        # Sort by relevance so the seed pick is the most relevant document,
        # which the original MMR formulation requires.
        candidates.sort(key=lambda c: c['base_score'], reverse=True)

        candidate_texts = [c['content'] for c in candidates]
        candidate_embeddings = self._db_manager.embedding_model.encode_batch(
            candidate_texts, show_progress=False
        )
        if candidate_embeddings.size == 0:
            return candidates[:n_results]

        selected = [0]
        selected_embeddings = [candidate_embeddings[0]]
        remaining = list(range(1, len(candidates)))

        while len(selected) < n_results and remaining:
            best_score = -float('inf')
            best_idx: Optional[int] = None

            selected_matrix = np.array(selected_embeddings)
            for idx in remaining:
                relevance = candidates[idx]['base_score']
                cand_emb = candidate_embeddings[idx].reshape(1, -1)
                similarities = cosine_similarity(cand_emb, selected_matrix)[0]
                max_sim = float(np.max(similarities))
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is None:
                break
            selected.append(best_idx)
            selected_embeddings.append(candidate_embeddings[best_idx])
            remaining.remove(best_idx)

        results = [candidates[i] for i in selected]
        for r in results:
            r['mmr_score'] = r['base_score']
            r['final_score'] = r['base_score']
        return results

    def rerank_search(
        self,
        query: str,
        strategy: str,
        n_results: int = 5,
        initial_results: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Cross-encoder rerank over a candidate pool from semantic search."""
        if initial_results is None:
            candidate_count = min(SEARCH.rerank_top_k, SEARCH.max_top_k)
            initial_results = self._db_manager.search(
                query, strategy, n_results=candidate_count
            )
        if not initial_results:
            return []

        pairs = [[query, r['content']] for r in initial_results]
        raw_scores = self.cross_encoder.predict(pairs)
        normalized_scores = self._scorer.normalize_cross_encoder_scores(raw_scores)

        for i, result in enumerate(initial_results):
            result['rerank_raw_score'] = float(raw_scores[i])
            base_score = float(normalized_scores[i])
            result['rerank_score'] = base_score
            result['final_score'] = self._scorer.compute_final_score(
                base_score=base_score,
                metadata=result['metadata'],
                query=query,
            )
            result['search_strategy'] = 'rerank'

        reranked = sorted(initial_results, key=lambda r: r['final_score'], reverse=True)
        return self._diversify_results(reranked)[:n_results]

    # -------------------------------------------------------------- orchestrators

    def parallel_search_single_strategy(
        self, query: str, chunking_strategy: str, n_results: int = 5
    ) -> List[Dict]:
        """Run all four search strategies in parallel against one chunking strategy."""
        strategies_to_run: Dict[str, Callable[[], List[Dict]]] = {
            'semantic': lambda: self.semantic_search(query, chunking_strategy, n_results * 3),
            'hybrid': lambda: self.hybrid_search(query, chunking_strategy, n_results * 3),
            'mmr': lambda: self.mmr_search(query, chunking_strategy, n_results * 3),
            'rerank': lambda: self.rerank_search(query, chunking_strategy, n_results * 3),
        }

        strategy_results = self._run_in_parallel(
            strategies_to_run, timeout=_SEARCH_TIMEOUT
        )

        if not any(strategy_results.values()):
            return []

        fused = self._fusion.reciprocal_rank_fusion(strategy_results, top_n=n_results)
        for result in fused:
            result['chunking_strategy'] = chunking_strategy
        return fused

    def parallel_search_all(
        self, query: str, n_results: int = 5
    ) -> Dict[str, List[Dict]]:
        """Fan out across all chunking strategies × all search strategies (16 searches)."""
        tasks: Dict[str, Callable[[], List[Dict]]] = {
            strategy: (
                lambda s=strategy: self.parallel_search_single_strategy(
                    query, s, n_results * 2
                )
            )
            for strategy in _CHUNKING_STRATEGIES
        }
        return self._run_in_parallel(tasks, timeout=_PARALLEL_SEARCH_TIMEOUT)

    def single_strategy_all_chunking(
        self, query: str, search_strategy: str, n_results: int = 5
    ) -> Dict[str, List[Dict]]:
        """Run one search strategy in parallel against every chunking strategy."""
        strategy_map: Dict[str, Callable] = {
            'semantic': self.semantic_search,
            'bm25': self.bm25_search,
            'hybrid': self.hybrid_search,
            'mmr': self.mmr_search,
            'rerank': self.rerank_search,
        }
        search_func = strategy_map.get(search_strategy)
        if search_func is None:
            logger.warning(
                "Unknown search strategy '%s', falling back to semantic", search_strategy
            )
            search_func = self.semantic_search

        tasks: Dict[str, Callable[[], List[Dict]]] = {
            chunking: (lambda c=chunking: search_func(query, c, n_results * 2))
            for chunking in _CHUNKING_STRATEGIES
        }
        return self._run_in_parallel(tasks, timeout=_SEARCH_TIMEOUT)

    # ---------------------------------------------------------------- internals

    def _run_in_parallel(
        self,
        tasks: Dict[str, Callable[[], List[Dict]]],
        timeout: int,
    ) -> Dict[str, List[Dict]]:
        results: Dict[str, List[Dict]] = {name: [] for name in tasks}
        if not tasks:
            return results

        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_name = {executor.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result(timeout=timeout)
                except Exception as e:
                    logger.warning("Strategy '%s' failed: %s", name, e)
                    print(f"   ⚠️  {name} failed: {e}")
                    results[name] = []
        return results

    @staticmethod
    def _diversify_results(
        results: List[Dict], threshold: float = _DIVERSITY_THRESHOLD
    ) -> List[Dict]:
        """Drop near-duplicate results using Jaccard word overlap.

        Cheap O(n²) but bounded by small input (top_k * 2 or 3). For larger
        candidate pools, swap to embedding-based dedup in fusion.deduplicate_results.
        """
        if not results:
            return results

        diverse = [results[0]]
        for candidate in results[1:]:
            cand_words = set(candidate['content'].lower().split())
            if not cand_words:
                diverse.append(candidate)
                continue

            is_diverse = True
            for selected in diverse:
                sel_words = set(selected['content'].lower().split())
                if not sel_words:
                    continue
                union = len(cand_words | sel_words)
                if union == 0:
                    continue
                jaccard = len(cand_words & sel_words) / union
                if jaccard > threshold:
                    is_diverse = False
                    break

            if is_diverse:
                diverse.append(candidate)
        return diverse
