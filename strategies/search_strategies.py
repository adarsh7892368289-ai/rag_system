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
        self._bm25_index = None
        self._bm25_corpus_hash = None
    
    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(SEARCH.rerank_model)
        return self._cross_encoder
    
    @property
    def bm25(self):
        current_hash = self._get_corpus_hash()
        
        if self._bm25_index is None or self._bm25_corpus_hash != current_hash:
            corpus = [doc['content'] for doc in self.db_manager.documents_cache]
            tokenized_corpus = [doc.lower().split() for doc in corpus]
            self._bm25_index = BM25Okapi(tokenized_corpus)
            self._bm25_corpus_hash = current_hash
        
        return self._bm25_index
    
    def semantic_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Pure vector similarity search"""
        raw_results = self.db_manager.search(query, n_results=n_results * 2)
        
        for result in raw_results:
            result['final_score'] = self.scorer.compute_final_score(
                base_score=result['similarity_score'],
                metadata=result['metadata'],
                query=query
            )
        
        raw_results.sort(key=lambda x: x['final_score'], reverse=True)
        diverse_results = self._diversify_results(raw_results)
        
        return diverse_results[:n_results]
    
    def bm25_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Keyword-based search using BM25"""
        if not self.db_manager.documents_cache:
            return []
        
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        normalized_scores = self.scorer.normalize_bm25(bm25_scores)
        
        top_indices = np.argsort(bm25_scores)[::-1][:n_results * 2]
        
        results = []
        for idx in top_indices:
            doc = self.db_manager.documents_cache[idx]
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
                'query': query
            })
        
        results.sort(key=lambda x: x['final_score'], reverse=True)
        diverse_results = self._diversify_results(results)
        
        return diverse_results[:n_results]
    
    def hybrid_search(self, query: str, n_results: int = 5, alpha: float = None) -> List[Dict]:
        """Combined semantic + BM25 search"""
        if alpha is None:
            alpha = SEARCH.hybrid_alpha
        
        if not self.db_manager.documents_cache:
            return []
        
        vector_results = self.db_manager.search(query, n_results=n_results * 3)
        
        if not vector_results:
            return []
        
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        normalized_bm25 = self.scorer.normalize_bm25(bm25_scores)
        
        hybrid_results = []
        for result in vector_results:
            doc_idx = next(
                (i for i, doc in enumerate(self.db_manager.documents_cache)
                 if doc['id'] == result['id']),
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
                hybrid_results.append(result)
        
        hybrid_results.sort(key=lambda x: x['final_score'], reverse=True)
        diverse_results = self._diversify_results(hybrid_results)
        
        return diverse_results[:n_results]
    
    def mmr_search(self, query: str, n_results: int = 5, lambda_param: float = None) -> List[Dict]:
        """Maximal Marginal Relevance search for diversity"""
        if lambda_param is None:
            lambda_param = SEARCH.mmr_lambda
        
        candidates = self.db_manager.search(
            query,
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
    
    def rerank_search(self, query: str, n_results: int = 5,
                     initial_results: Optional[List[Dict]] = None) -> List[Dict]:
        """Cross-encoder reranking for high accuracy"""
        if initial_results is None:
            initial_results = self.db_manager.search(
                query,
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

        reranked = sorted(initial_results, key=lambda x: x['final_score'], reverse=True)
        diverse_results = self._diversify_results(reranked)

        return diverse_results[:n_results]
    
    def parallel_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Execute multiple strategies in parallel and fuse with RRF
        
        NEW: Uses Reciprocal Rank Fusion for true top-N ranking
        """
        strategies_to_run = {
            'semantic': self.semantic_search,
            'hybrid': self.hybrid_search,
            'mmr': self.mmr_search,
            'rerank': self.rerank_search
        }
        
        strategy_results = {}
        
        # Execute strategies in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_strategy = {
                executor.submit(strategy_func, query, n_results * 3): name
                for name, strategy_func in strategies_to_run.items()
            }
            
            for future in as_completed(future_to_strategy):
                strategy_name = future_to_strategy[future]
                try:
                    results = future.result(timeout=30)
                    strategy_results[strategy_name] = results
                except Exception as e:
                    print(f"   ⚠️  {strategy_name} failed: {e}")
                    strategy_results[strategy_name] = []
        
        # Apply RRF fusion
        if not any(strategy_results.values()):
            return []
        
        fused_results = self.fusion.reciprocal_rank_fusion(
            strategy_results,
            top_n=n_results
        )
        
        return fused_results
    
    def _diversify_results(self, results: List[Dict], threshold: float = 0.85) -> List[Dict]:
        """Remove near-duplicate results using Jaccard similarity"""
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
    
    def _get_corpus_hash(self):
        return len(self.db_manager.documents_cache)
    
    def invalidate_caches(self):
        """Clear caches when database changes"""
        self._bm25_corpus_hash = None