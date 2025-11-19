import numpy as np
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from config.settings import SEARCH
from scoring.scorer import UnifiedScorer
from strategies.fusion import ResultFusion


class AdvancedSearchStrategies:
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.scorer = UnifiedScorer()
        self.fusion = ResultFusion(db_manager.embedding_model)
        self._cross_encoder = None
        self._bm25_indices = {}
        self._bm25_corpus_hashes = {}
    
    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(SEARCH.rerank_model)
        return self._cross_encoder
    
    def get_bm25(self, strategy: str):
        current_hash = len(self.db_manager.documents_cache.get(strategy, []))
        
        if strategy not in self._bm25_indices or self._bm25_corpus_hashes.get(strategy) != current_hash:
            corpus = [doc['content'] for doc in self.db_manager.documents_cache.get(strategy, [])]
            if corpus:
                tokenized_corpus = [doc.lower().split() for doc in corpus]
                self._bm25_indices[strategy] = BM25Okapi(tokenized_corpus)
                self._bm25_corpus_hashes[strategy] = current_hash
        
        return self._bm25_indices.get(strategy)
    
    def semantic_search(self, query: str, strategy: str, n_results: int = 5) -> List[Dict]:
        raw_results = self.db_manager.search(query, strategy, n_results=n_results * 2)
        
        for result in raw_results:
            result['final_score'] = self.scorer.compute_final_score(
                base_score=result['similarity_score'],
                metadata=result['metadata'],
                query=query
            )
            result['search_strategy'] = 'semantic'
        
        raw_results.sort(key=lambda x: x['final_score'], reverse=True)
        diverse_results = self._diversify_results(raw_results)
        
        return diverse_results[:n_results]
    
    def bm25_search(self, query: str, strategy: str, n_results: int = 5) -> List[Dict]:
        cache = self.db_manager.documents_cache.get(strategy, [])
        if not cache:
            return []
        
        bm25 = self.get_bm25(strategy)
        if not bm25:
            return []
        
        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        normalized_scores = self.scorer.normalize_bm25(bm25_scores)
        
        top_indices = np.argsort(bm25_scores)[::-1][:n_results * 2]
        
        results = []
        for idx in top_indices:
            doc = cache[idx]
            base_score = float(normalized_scores[idx])
            
            final_score = self.scorer.compute_final_score(
                base_score=base_score,
                metadata=doc.get('metadata', {}),
                query=query
            )
            
            results.append({
                'id': doc['id'],
                'content': doc['content'],
                'metadata': doc.get('metadata', {}),
                'bm25_score': base_score,
                'final_score': final_score,
                'query': query,
                'chunking_strategy': strategy,
                'search_strategy': 'bm25'
            })
        
        results.sort(key=lambda x: x['final_score'], reverse=True)
        diverse_results = self._diversify_results(results)
        
        return diverse_results[:n_results]
    
    def hybrid_search(self, query: str, strategy: str, n_results: int = 5, alpha: float = None) -> List[Dict]:
        if alpha is None:
            alpha = SEARCH.hybrid_alpha
        
        cache = self.db_manager.documents_cache.get(strategy, [])
        if not cache:
            return []
        
        vector_results = self.db_manager.search(query, strategy, n_results=n_results * 3)
        
        if not vector_results:
            return []
        
        bm25 = self.get_bm25(strategy)
        if not bm25:
            return vector_results[:n_results]
        
        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)
        normalized_bm25 = self.scorer.normalize_bm25(bm25_scores)
        
        hybrid_results = []
        for result in vector_results:
            doc_idx = next(
                (i for i, doc in enumerate(cache) if doc['id'] == result['id']),
                None
            )
            
            if doc_idx is not None:
                bm25_score = float(normalized_bm25[doc_idx])
                vector_score = result['similarity_score']
                hybrid_base_score = alpha * vector_score + (1 - alpha) * bm25_score
                
                final_score = self.scorer.compute_final_score(
                    base_score=hybrid_base_score,
                    metadata=result['metadata'],
                    query=query
                )
                
                result['bm25_score'] = bm25_score
                result['vector_score'] = vector_score
                result['hybrid_score'] = hybrid_base_score
                result['final_score'] = final_score
                result['search_strategy'] = 'hybrid'
                hybrid_results.append(result)
        
        hybrid_results.sort(key=lambda x: x['final_score'], reverse=True)
        diverse_results = self._diversify_results(hybrid_results)
        
        return diverse_results[:n_results]
    
    def mmr_search(self, query: str, strategy: str, n_results: int = 5, lambda_param: float = None) -> List[Dict]:
        if lambda_param is None:
            lambda_param = SEARCH.mmr_lambda
        
        candidates = self.db_manager.search(
            query,
            strategy,
            n_results=min(n_results * SEARCH.mmr_candidates_multiplier, SEARCH.max_top_k)
        )
        
        if not candidates:
            return []
        
        for candidate in candidates:
            candidate['base_score'] = self.scorer.compute_final_score(
                base_score=candidate['similarity_score'],
                metadata=candidate['metadata'],
                query=query
            )
            candidate['search_strategy'] = 'mmr'
        
        candidate_texts = [c['content'] for c in candidates]
        candidate_embeddings = self.db_manager.embedding_model.encode_batch(
            candidate_texts, show_progress=False
        )
        
        selected = []
        selected_embeddings = []
        remaining = list(range(len(candidates)))
        
        selected.append(0)
        selected_embeddings.append(candidate_embeddings[0])
        remaining.remove(0)
        
        while len(selected) < n_results and remaining:
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining:
                relevance = candidates[idx]['base_score']
                
                candidate_emb = candidate_embeddings[idx].reshape(1, -1)
                selected_embs = np.array(selected_embeddings)
                similarities = cosine_similarity(candidate_emb, selected_embs)[0]
                max_sim = float(np.max(similarities))
                
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                selected_embeddings.append(candidate_embeddings[best_idx])
                remaining.remove(best_idx)
        
        mmr_results = [candidates[i] for i in selected]
        for result in mmr_results:
            result['mmr_score'] = result['base_score']
            result['final_score'] = result['base_score']
        
        return mmr_results
    
    def rerank_search(self, query: str, strategy: str, n_results: int = 5,
                     initial_results: Optional[List[Dict]] = None) -> List[Dict]:
        if initial_results is None:
            initial_results = self.db_manager.search(
                query,
                strategy,
                n_results=min(SEARCH.rerank_top_k, SEARCH.max_top_k)
            )

        if not initial_results:
            return []

        pairs = [[query, result['content']] for result in initial_results]
        raw_scores = self.cross_encoder.predict(pairs)
        normalized_scores = self.scorer.normalize_cross_encoder_scores(raw_scores)

        for i, result in enumerate(initial_results):
            result['rerank_raw_score'] = float(raw_scores[i])
            base_score = float(normalized_scores[i])
            
            final_score = self.scorer.compute_final_score(
                base_score=base_score,
                metadata=result['metadata'],
                query=query
            )
            
            result['rerank_score'] = base_score
            result['final_score'] = final_score
            result['search_strategy'] = 'rerank'

        reranked = sorted(initial_results, key=lambda x: x['final_score'], reverse=True)
        diverse_results = self._diversify_results(reranked)

        return diverse_results[:n_results]
    
    def parallel_search_single_strategy(self, query: str, chunking_strategy: str, n_results: int = 5) -> List[Dict]:
        strategies_to_run = {
            'semantic': lambda: self.semantic_search(query, chunking_strategy, n_results * 3),
            'hybrid': lambda: self.hybrid_search(query, chunking_strategy, n_results * 3),
            'mmr': lambda: self.mmr_search(query, chunking_strategy, n_results * 3),
            'rerank': lambda: self.rerank_search(query, chunking_strategy, n_results * 3)
        }
        
        strategy_results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_strategy = {
                executor.submit(strategy_func): name
                for name, strategy_func in strategies_to_run.items()
            }
            
            for future in as_completed(future_to_strategy):
                strategy_name = future_to_strategy[future]
                try:
                    results = future.result(timeout=30)
                    strategy_results[strategy_name] = results
                except Exception as e:
                    strategy_results[strategy_name] = []
        
        if not any(strategy_results.values()):
            return []
        
        fused_results = self.fusion.reciprocal_rank_fusion(
            strategy_results,
            top_n=n_results
        )
        
        for result in fused_results:
            result['chunking_strategy'] = chunking_strategy
        
        return fused_results
    
    def parallel_search_all(self, query: str, n_results: int = 5) -> Dict[str, List[Dict]]:
        chunking_strategies = ['sentence_aware', 'semantic', 'paragraph', 'fixed_size']
        
        all_chunking_results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_chunking = {
                executor.submit(self.parallel_search_single_strategy, query, strategy, n_results * 2): strategy
                for strategy in chunking_strategies
            }
            
            for future in as_completed(future_to_chunking):
                chunking_strategy = future_to_chunking[future]
                try:
                    results = future.result(timeout=60)
                    all_chunking_results[chunking_strategy] = results
                except Exception as e:
                    print(f"   ⚠️  {chunking_strategy} search failed: {e}")
                    all_chunking_results[chunking_strategy] = []
        
        return all_chunking_results
    
    def single_strategy_all_chunking(self, query: str, search_strategy: str, n_results: int = 5) -> Dict[str, List[Dict]]:
        chunking_strategies = ['sentence_aware', 'semantic', 'paragraph', 'fixed_size']
        
        strategy_map = {
            'semantic': self.semantic_search,
            'bm25': self.bm25_search,
            'hybrid': self.hybrid_search,
            'mmr': self.mmr_search,
            'rerank': self.rerank_search
        }
        
        search_func = strategy_map.get(search_strategy, self.semantic_search)
        
        all_chunking_results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_chunking = {
                executor.submit(search_func, query, chunking, n_results * 2): chunking
                for chunking in chunking_strategies
            }
            
            for future in as_completed(future_to_chunking):
                chunking_strategy = future_to_chunking[future]
                try:
                    results = future.result(timeout=30)
                    all_chunking_results[chunking_strategy] = results
                except Exception as e:
                    print(f"   ⚠️  {chunking_strategy} failed: {e}")
                    all_chunking_results[chunking_strategy] = []
        
        return all_chunking_results
    
    def _diversify_results(self, results: List[Dict], threshold: float = 0.85) -> List[Dict]:
        if not results:
            return results
        
        diverse_results = [results[0]]
        
        for result in results[1:]:
            is_diverse = True
            result_words = set(result['content'].lower().split())
            
            for selected in diverse_results:
                selected_words = set(selected['content'].lower().split())
                
                if not result_words or not selected_words:
                    continue
                
                intersection = len(result_words & selected_words)
                union = len(result_words | selected_words)
                jaccard_sim = intersection / union if union > 0 else 0
                
                if jaccard_sim > threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(result)
        
        return diverse_results
    
    def invalidate_caches(self):
        self._bm25_corpus_hashes = {}