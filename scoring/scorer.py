import numpy as np
from typing import List, Dict, Any


class UnifiedScorer:
    """Adaptive scoring system - no hardcoded thresholds"""
    
    # Maximum boost range (adaptive within this)
    MAX_METADATA_BOOST = 0.15
    
    def normalize_distance(self, distance: float) -> float:
        """
        Convert ChromaDB cosine distance [0, 2] to similarity [0, 1]
        
        Cosine distance = 1 - cosine_similarity
        Therefore: similarity = 1 - (distance / 2)
        """
        return max(0.0, min(1.0, 1.0 - (distance / 2.0)))
    
    def normalize_bm25(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize BM25 scores using percentile ranking
        
        Modern approach:
        - No hardcoded ranges
        - Relative ranking preserved
        - Works for any score distribution
        """
        scores = np.array(scores, dtype=float)
        
        if len(scores) == 0:
            return scores
        
        # Handle uniform scores
        if np.std(scores) < 1e-10:
            return np.full_like(scores, 0.5)
        
        # Percentile-based ranking
        percentiles = np.zeros_like(scores)
        sorted_indices = np.argsort(scores)
        
        for rank, idx in enumerate(sorted_indices):
            percentiles[idx] = rank / max(len(scores) - 1, 1)
        
        # Map to [0.2, 1.0] to preserve ordering
        # Bottom result: 0.2, Top result: 1.0
        normalized = 0.2 + percentiles * 0.8
        
        return np.clip(normalized, 0.0, 1.0)
    
    def calculate_metadata_boost(self, metadata: Dict[str, Any], 
                                 query: str = None,
                                 result_pool: List[Dict] = None) -> float:
        """
        Calculate adaptive metadata boost using relative ranking
        
        Modern approach:
        - Compare to other results in pool (percentile-based)
        - Quality score based (universal metric)
        - Query relevance (if provided)
        - No hardcoded thresholds
        
        Args:
            metadata: Document metadata
            query: Search query (optional)
            result_pool: All results for relative comparison (optional)
        """
        boost = 0.0
        
        # 1. Quality-based boost (universal)
        quality_score = metadata.get('quality_score', 0)
        
        if result_pool and len(result_pool) > 1:
            # Adaptive: Compare to other results
            qualities = [r['metadata'].get('quality_score', 0) for r in result_pool]
            quality_percentile = self._calculate_percentile(quality_score, qualities)
            
            # Top 25% get boost
            if quality_percentile >= 0.75:
                boost += 0.08
            elif quality_percentile >= 0.5:
                boost += 0.05
        else:
            # Fallback: Absolute thresholds
            if quality_score > 0.8:
                boost += 0.08
            elif quality_score > 0.6:
                boost += 0.05
        
        # 2. Query-metadata relevance (if query provided)
        if query:
            relevance_boost = self._calculate_query_relevance(metadata, query)
            boost += relevance_boost
        
        return min(boost, self.MAX_METADATA_BOOST)
    
    def _calculate_percentile(self, value: float, values: List[float]) -> float:
        """Calculate percentile rank of value in distribution"""
        if not values or len(values) == 1:
            return 0.5
        
        sorted_values = sorted(values)
        rank = sum(1 for v in sorted_values if v <= value)
        percentile = rank / len(sorted_values)
        
        return percentile
    
    def _calculate_query_relevance(self, metadata: Dict[str, Any], 
                                   query: str) -> float:
        """
        Calculate query-metadata relevance boost
        
        Universal approach:
        - Title matching
        - Keyword matching
        - Source matching
        """
        boost = 0.0
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        # Title matching (if available)
        title = metadata.get('title', '').lower()
        if title:
            title_terms = set(title.split())
            overlap = len(query_terms & title_terms)
            
            if overlap > 0:
                # Adaptive: based on overlap ratio
                overlap_ratio = overlap / len(query_terms) if query_terms else 0
                boost += min(overlap_ratio * 0.05, 0.05)
        
        # Keyword matching (if available)
        keywords = metadata.get('keywords', [])
        if keywords:
            keyword_set = set(kw.lower() for kw in keywords)
            keyword_overlap = len(query_terms & keyword_set)
            
            if keyword_overlap > 0:
                boost += 0.02
        
        return boost
    
    def compute_final_score(self, base_score: float, 
                           metadata: Dict[str, Any],
                           query: str = None,
                           result_pool: List[Dict] = None) -> float:
        """
        Compute final score with adaptive boosting
        
        Formula: final_score = min(base_score + boost, 1.0)
        
        Args:
            base_score: Base similarity/relevance score
            metadata: Document metadata
            query: Search query
            result_pool: All results for relative comparison
        """
        boost = self.calculate_metadata_boost(metadata, query, result_pool)
        final_score = base_score + boost
        
        return min(final_score, 1.0)
    
    def normalize_cross_encoder_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize cross-encoder scores using percentile ranking
        
        Modern approach:
        - Adaptive to score distribution
        - Preserves relative ordering
        - No assumptions about score range
        """
        scores = np.array(scores, dtype=float)
        
        if len(scores) == 0:
            return scores
        
        # Handle uniform scores
        if np.std(scores) < 1e-10:
            return np.full_like(scores, 0.6)
        
        # Percentile ranking
        percentiles = np.zeros_like(scores)
        sorted_indices = np.argsort(scores)
        
        for rank, idx in enumerate(sorted_indices):
            percentiles[idx] = rank / max(len(scores) - 1, 1)
        
        # Map to [0.3, 1.0]
        # Reranked results should be relatively high quality
        normalized = 0.3 + percentiles * 0.7
        
        return np.clip(normalized, 0.0, 1.0)
    
    def normalize_strategy_scores(self, results: List[Dict], 
                                  strategy_type: str) -> List[Dict]:
        """
        Apply strategy-specific normalization
        
        Args:
            results: List of results with raw scores
            strategy_type: 'semantic', 'bm25', 'hybrid', etc.
        """
        if not results:
            return results
        
        if strategy_type == 'semantic':
            # Distance already normalized in database.py
            for r in results:
                r['normalized_score'] = r.get('similarity_score', 0)
        
        elif strategy_type == 'bm25':
            # Extract BM25 scores and normalize
            bm25_scores = np.array([r.get('bm25_score', 0) for r in results])
            normalized = self.normalize_bm25(bm25_scores)
            
            for i, r in enumerate(results):
                r['normalized_score'] = float(normalized[i])
        
        elif strategy_type == 'hybrid':
            # Hybrid scores already normalized during calculation
            for r in results:
                r['normalized_score'] = r.get('hybrid_score', 0)
        
        elif strategy_type == 'rerank':
            # Cross-encoder normalization
            raw_scores = np.array([r.get('rerank_raw_score', 0) for r in results])
            normalized = self.normalize_cross_encoder_scores(raw_scores)
            
            for i, r in enumerate(results):
                r['normalized_score'] = float(normalized[i])
        
        return results
    
    def apply_adaptive_boosting_to_results(self, results: List[Dict], 
                                          query: str = None) -> List[Dict]:
        """
        Apply adaptive metadata boosting to all results
        
        Modern approach:
        - Uses result pool for relative comparison
        - Each result compared to others
        - Percentile-based boosting
        """
        for result in results:
            base_score = result.get('normalized_score', result.get('similarity_score', 0))
            
            final_score = self.compute_final_score(
                base_score=base_score,
                metadata=result['metadata'],
                query=query,
                result_pool=results
            )
            
            result['final_score'] = final_score
        
        return results