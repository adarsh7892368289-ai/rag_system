import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from config.settings import SEARCH

class AdvancedSearchStrategies:
    """Advanced search strategies with uniform score normalization"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
        # Cache models and indices
        self._cross_encoder = None
        self._bm25_index = None
        self._bm25_corpus_hash = None
        
    @property
    def cross_encoder(self):
        """Lazy load and cache cross-encoder model"""
        if self._cross_encoder is None:
            print(f"🔄 Loading Cross-Encoder: {SEARCH.rerank_model}")
            self._cross_encoder = CrossEncoder(SEARCH.rerank_model)
            print("✅ Cross-Encoder loaded and cached")
        return self._cross_encoder
    
    @property
    def bm25(self):
        """Lazy load and cache BM25 index"""
        current_hash = self._get_corpus_hash()
        
        if self._bm25_index is None or self._bm25_corpus_hash != current_hash:
            print("🔄 Building BM25 index...")
            corpus = [doc['content'] for doc in self.db_manager.documents_cache]
            tokenized_corpus = [doc.lower().split() for doc in corpus]
            self._bm25_index = BM25Okapi(tokenized_corpus)
            self._bm25_corpus_hash = current_hash
            print(f"✅ BM25 index built ({len(corpus)} documents)")
        
        return self._bm25_index
    
    def _get_corpus_hash(self):
        """Get hash of current corpus for cache validation"""
        return len(self.db_manager.documents_cache)
    
    def semantic_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Pure vector similarity search - scores naturally in [0,1] range"""
        results = self.db_manager.search(query, n_results=n_results)
        
        # Cosine similarity is already [0,1], but often concentrated in [0.5, 0.9]
        # No additional normalization needed - this is our baseline
        for r in results:
            r['final_score'] = r['similarity_score']
        
        return results
    
    def bm25_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Keyword-based BM25 search with normalized scores"""
        if not self.db_manager.documents_cache:
            return []
        
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores to match semantic search range
        # BM25 scores are unbounded, so we use percentile-based normalization
        normalized_scores = self._percentile_normalize(bm25_scores)
        
        # Get top N results
        top_indices = np.argsort(bm25_scores)[::-1][:n_results]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            doc = self.db_manager.documents_cache[idx]
            results.append({
                'id': doc['id'],
                'content': doc['content'],
                'metadata': doc.get('metadata', {}),
                'bm25_score': float(normalized_scores[idx]),
                'final_score': float(normalized_scores[idx]),
                'rank': rank,
                'query': query
            })
        
        return results
    
    def hybrid_search(self, query: str, n_results: int = 5, alpha: float = None) -> List[Dict]:
        """Combines semantic and BM25 search"""
        if alpha is None:
            alpha = SEARCH.hybrid_alpha
        
        if not self.db_manager.documents_cache:
            return []
        
        # Get semantic results (already in good range)
        vector_results = self.db_manager.search(query, n_results=n_results * 2)
        
        if not vector_results:
            return []
        
        # Get BM25 scores for all documents
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        normalized_bm25 = self._percentile_normalize(bm25_scores)
        
        # Combine scores
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
                
                # Weighted combination
                hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score
                
                result['bm25_score'] = bm25_score
                result['vector_score'] = vector_score
                result['hybrid_score'] = hybrid_score
                result['final_score'] = hybrid_score
                hybrid_results.append(result)
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(hybrid_results[:n_results], 1):
            result['rank'] = i
        
        return hybrid_results[:n_results]
    
    def mmr_search(self, query: str, n_results: int = 5, lambda_param: float = None) -> List[Dict]:
        """Maximal Marginal Relevance for diverse results"""
        if lambda_param is None:
            lambda_param = SEARCH.mmr_lambda
        
        # Get more candidates than needed
        candidates = self.db_manager.search(
            query,
            n_results=min(n_results * SEARCH.mmr_candidates_multiplier, SEARCH.max_top_k)
        )
        
        if not candidates:
            return []
        
        # Get embeddings for candidates
        candidate_texts = [c['content'] for c in candidates]
        candidate_embeddings = self.db_manager.embedding_model.encode_batch(candidate_texts)
        
        selected = []
        selected_embeddings = []
        remaining = list(range(len(candidates)))
        
        # Select first document (highest relevance)
        selected.append(0)
        selected_embeddings.append(candidate_embeddings[0])
        remaining.remove(0)
        
        # Iteratively select diverse documents
        while len(selected) < n_results and remaining:
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining:
                # Relevance score (already in good range)
                relevance = candidates[idx]['similarity_score']
                
                # Calculate max similarity to selected documents
                candidate_emb = candidate_embeddings[idx].reshape(1, -1)
                selected_embs = np.array(selected_embeddings)
                similarities = cosine_similarity(candidate_emb, selected_embs)[0]
                max_sim = float(np.max(similarities))
                
                # MMR score: balance relevance and diversity
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                selected_embeddings.append(candidate_embeddings[best_idx])
                remaining.remove(best_idx)
        
        # Return selected documents
        mmr_results = [candidates[i] for i in selected]
        for i, result in enumerate(mmr_results, 1):
            result['mmr_score'] = result['similarity_score']
            result['final_score'] = result['similarity_score']
            result['rank'] = i
        
        return mmr_results
    
    def rerank_search(self, query: str, n_results: int = 5,
                     initial_results: Optional[List[Dict]] = None) -> List[Dict]:
        """Cross-encoder reranking with calibrated score normalization"""
        # Get initial candidates if not provided
        if initial_results is None:
            initial_results = self.db_manager.search(
                query,
                n_results=min(SEARCH.rerank_top_k, SEARCH.max_top_k)
            )

        if not initial_results:
            return []

        # Prepare pairs for cross-encoder
        pairs = [[query, result['content']] for result in initial_results]
        
        # Get reranking scores (uses cached model)
        raw_scores = self.cross_encoder.predict(pairs)

        # CRITICAL FIX: Use percentile-based normalization instead of sigmoid
        # This matches the score range of other techniques (typically 0.3-0.8)
        normalized_scores = self._percentile_normalize(raw_scores, target_range=(0.2, 0.85))

        # Assign scores
        for i, result in enumerate(initial_results):
            result['rerank_raw_score'] = float(raw_scores[i])
            result['rerank_score'] = float(normalized_scores[i])
            result['final_score'] = float(normalized_scores[i])

        # Sort by rerank score
        reranked = sorted(initial_results, key=lambda x: x['rerank_score'], reverse=True)

        # Update ranks
        for i, result in enumerate(reranked[:n_results], 1):
            result['rank'] = i

        return reranked[:n_results]
    
    def _percentile_normalize(self, scores: np.ndarray, 
                             target_range: tuple = (0.2, 0.9)) -> np.ndarray:
        """
        Percentile-based normalization for uniform score distribution
        
        This method ensures scores from different techniques fall in similar ranges
        by mapping the distribution to a target range while preserving relative ordering.
        
        Args:
            scores: Raw scores to normalize
            target_range: Desired output range (min, max)
        
        Returns:
            Normalized scores in target_range
        """
        scores = np.array(scores, dtype=float)
        
        if len(scores) == 0:
            return scores
        
        # Handle edge case where all scores are identical
        if np.std(scores) < 1e-10:
            mid_point = (target_range[0] + target_range[1]) / 2
            return np.full_like(scores, mid_point)
        
        # Use percentile rank to get uniform distribution
        # This maps scores to [0, 1] based on their rank in the distribution
        percentiles = np.zeros_like(scores)
        sorted_indices = np.argsort(scores)
        
        for rank, idx in enumerate(sorted_indices):
            percentiles[idx] = rank / (len(scores) - 1) if len(scores) > 1 else 0.5
        
        # Map percentiles to target range
        min_score, max_score = target_range
        normalized = min_score + percentiles * (max_score - min_score)
        
        return np.clip(normalized, target_range[0], target_range[1])
    
    def _minmax_normalize(self, scores: np.ndarray, 
                         target_range: tuple = (0.0, 1.0)) -> np.ndarray:
        """Min-max normalization (kept for compatibility)"""
        scores = np.array(scores)
        
        if len(scores) == 0:
            return scores
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            mid_point = (target_range[0] + target_range[1]) / 2
            return np.full_like(scores, mid_point)
        
        normalized = (scores - min_score) / (max_score - min_score)
        min_target, max_target = target_range
        normalized = min_target + normalized * (max_target - min_target)
        
        return np.clip(normalized, target_range[0], target_range[1])
    
    def invalidate_caches(self):
        """Force rebuild of cached indices (call after adding documents)"""
        self._bm25_corpus_hash = None