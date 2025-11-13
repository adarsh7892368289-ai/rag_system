"""
Result Fusion using Reciprocal Rank Fusion (RRF)

Modern RAG approach for merging results from multiple search strategies
"""

import numpy as np
from typing import List, Dict, Any
from collections import defaultdict


class ResultFusion:
    """
    Fuse results from multiple search strategies
    
    Uses Reciprocal Rank Fusion (RRF) - state-of-the-art method
    Paper: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
    """
    
    def __init__(self, embedding_generator=None):
        self.embedding_generator = embedding_generator
        self.rrf_k = 60  # Standard RRF constant
    
    def reciprocal_rank_fusion(self, 
                               strategy_results: Dict[str, List[Dict]],
                               top_n: int = 5) -> List[Dict]:
        """
        Apply Reciprocal Rank Fusion to merge results from multiple strategies
        
        RRF Formula: score(doc) = Σ(1 / (k + rank_in_strategy_i))
        
        Args:
            strategy_results: {strategy_name: [results]}
            top_n: Number of final results to return
            
        Returns:
            Fused and ranked results with provenance
        """
        if not strategy_results:
            return []
        
        # Track document scores and provenance
        doc_scores = defaultdict(lambda: {
            'rrf_score': 0.0,
            'strategies': [],
            'strategy_scores': {},
            'strategy_ranks': {},
            'result_data': None
        })
        
        # Calculate RRF score for each document
        for strategy_name, results in strategy_results.items():
            for rank, result in enumerate(results, start=1):
                doc_id = result['id']
                
                # RRF formula: 1 / (k + rank)
                rrf_contribution = 1.0 / (self.rrf_k + rank)
                
                doc_scores[doc_id]['rrf_score'] += rrf_contribution
                doc_scores[doc_id]['strategies'].append(strategy_name)
                doc_scores[doc_id]['strategy_scores'][strategy_name] = result.get('final_score', 0)
                doc_scores[doc_id]['strategy_ranks'][strategy_name] = rank
                
                # Store first occurrence of result data
                if doc_scores[doc_id]['result_data'] is None:
                    doc_scores[doc_id]['result_data'] = result
        
        # Convert to list and calculate confidence
        fused_results = []
        for doc_id, data in doc_scores.items():
            result = data['result_data'].copy()
            
            # Add fusion metadata
            result['fusion_score'] = data['rrf_score']
            result['strategies_used'] = data['strategies']
            result['strategy_scores'] = data['strategy_scores']
            result['strategy_ranks'] = data['strategy_ranks']
            
            # Calculate confidence (strategy agreement)
            result['confidence'] = self._calculate_confidence(data)
            
            # Update final score to use RRF score (normalized)
            max_possible_rrf = len(strategy_results) / self.rrf_k
            normalized_rrf = min(data['rrf_score'] / max_possible_rrf, 1.0)
            result['final_score'] = normalized_rrf
            
            fused_results.append(result)
        
        # Sort by RRF score
        fused_results.sort(key=lambda x: x['fusion_score'], reverse=True)
        
        # Apply diversity filter
        diverse_results = self._apply_diversity(fused_results, top_n)
        
        return diverse_results[:top_n]
    
    def _calculate_confidence(self, doc_data: Dict) -> float:
        """
        Calculate confidence score based on strategy agreement
        
        High confidence = multiple strategies ranked it highly
        
        Components:
        - Coverage: How many strategies found it? (40%)
        - Quality: Average rank position (30%)
        - Agreement: Consistency of ranks (30%)
        """
        strategies = doc_data['strategies']
        ranks = doc_data['strategy_ranks']
        
        if not strategies:
            return 0.0
        
        # 1. Coverage score (how many strategies found it)
        num_strategies = len(strategies)
        coverage_score = min(num_strategies / 4, 1.0)  # Assume max 4 strategies
        
        # 2. Quality score (average rank position)
        avg_rank = sum(ranks.values()) / len(ranks)
        quality_score = 1.0 / (1.0 + avg_rank / 10)  # Normalize with decay
        
        # 3. Agreement score (rank consistency)
        if len(ranks) > 1:
            rank_values = list(ranks.values())
            mean_rank = sum(rank_values) / len(rank_values)
            variance = sum((r - mean_rank) ** 2 for r in rank_values) / len(rank_values)
            std_dev = variance ** 0.5
            
            # Low std_dev = high agreement
            agreement_score = 1.0 / (1.0 + std_dev / 5)
        else:
            agreement_score = 1.0  # Single strategy = perfect "agreement"
        
        # Weighted combination
        confidence = (
            0.40 * coverage_score +
            0.30 * quality_score +
            0.30 * agreement_score
        )
        
        return min(confidence, 1.0)
    
    def _apply_diversity(self, results: List[Dict], top_n: int) -> List[Dict]:
        """
        Apply diversity filter using Maximal Marginal Relevance (MMR)
        
        MMR = λ * Relevance - (1-λ) * max(Similarity to selected)
        
        Args:
            results: Ranked results
            top_n: Number of diverse results to select
            
        Returns:
            Diverse results
        """
        if not results or len(results) <= top_n:
            return results
        
        # If no embedding generator, return as-is
        if not self.embedding_generator:
            return results
        
        lambda_param = 0.7  # Balance relevance vs diversity
        
        # Generate embeddings for all results
        contents = [r['content'] for r in results]
        embeddings = self.embedding_generator.encode_batch(contents, show_progress=False)
        
        selected_indices = []
        selected_embeddings = []
        remaining = list(range(len(results)))
        
        # Select first result (highest RRF score)
        selected_indices.append(0)
        selected_embeddings.append(embeddings[0])
        remaining.remove(0)
        
        # Iteratively select diverse results
        while len(selected_indices) < top_n and remaining:
            best_mmr = -float('inf')
            best_idx = None
            
            for idx in remaining:
                # Relevance score (from RRF)
                relevance = results[idx]['fusion_score']
                
                # Calculate max similarity to already selected documents
                candidate_emb = embeddings[idx].reshape(1, -1)
                selected_embs = np.array(selected_embeddings)
                
                similarities = self._cosine_similarity_batch(candidate_emb, selected_embs)
                max_similarity = float(np.max(similarities))
                
                # MMR formula
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_embeddings.append(embeddings[best_idx])
                remaining.remove(best_idx)
        
        # Return selected results in order
        return [results[i] for i in selected_indices]
    
    def deduplicate_results(self, results: List[Dict], threshold: float = 0.95) -> List[Dict]:
        """
        Remove near-duplicate results using embedding similarity
        
        Args:
            results: List of results to deduplicate
            threshold: Similarity threshold (default: 0.95 = 95% similar)
            
        Returns:
            Deduplicated results
        """
        if not results or not self.embedding_generator:
            return results
        
        # Generate embeddings
        contents = [r['content'] for r in results]
        embeddings = self.embedding_generator.encode_batch(contents, show_progress=False)
        
        unique_results = []
        unique_embeddings = []
        
        for result, embedding in zip(results, embeddings):
            is_duplicate = False
            
            for unique_emb in unique_embeddings:
                similarity = self._cosine_similarity(embedding, unique_emb)
                
                if similarity > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
                unique_embeddings.append(embedding)
        
        return unique_results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _cosine_similarity_batch(self, vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between vector and matrix of vectors"""
        dot_products = np.dot(matrix, vec.T).flatten()
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vec)
        
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)
        
        return dot_products / norms