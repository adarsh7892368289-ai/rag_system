import time
import numpy as np
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.settings import SEARCH

class UnifiedSearchEngine:
    """Unified search engine with uniform scoring across all modes"""
    
    def __init__(self, db_manager, strategies):
        self.db_manager = db_manager
        self.strategies = strategies
    
    def search(self, query: str, top_n: int = None, mode: str = None, auto_route: bool = False) -> List[Dict]:
        """Execute search with uniform scoring"""
        if top_n is None:
            top_n = SEARCH.top_k
        
        # Auto-route if enabled
        if mode is None:
            if auto_route:
                from strategies.query_processor import QueryRouter
                mode = QueryRouter.route_query(query)
            else:
                mode = SEARCH.default_mode
        
        start_time = time.time()
        
        # Execute search based on mode
        if mode == 'parallel':
            results = self._parallel_search(query, top_n)
        elif mode == 'fast':
            results = self._fast_search(query, top_n)
        elif mode == 'accurate':
            results = self._accurate_search(query, top_n)
        elif mode == 'semantic':
            results = self.strategies.semantic_search(query, top_n)
        elif mode == 'hybrid':
            results = self.strategies.hybrid_search(query, top_n)
        elif mode == 'bm25':
            results = self.strategies.bm25_search(query, top_n)
        elif mode == 'mmr':
            results = self.strategies.mmr_search(query, top_n)
        elif mode == 'rerank':
            results = self.strategies.rerank_search(query, top_n)
        else:
            results = self._parallel_search(query, top_n)
        
        execution_time = time.time() - start_time
        
        # Apply threshold
        threshold = SEARCH.confidence_thresholds.get(mode, SEARCH.default_confidence_threshold)
        filtered_results = [
            r for r in results
            if r.get('confidence', r.get('final_score', 0)) >= threshold
        ]

        # Add search technique to results
        for result in filtered_results:
            result['search_technique'] = mode
            result['execution_time'] = execution_time

        return filtered_results
    
    def _parallel_search(self, query: str, top_n: int) -> List[Dict]:
        """Execute multiple strategies in parallel with improved fusion"""
        all_results = {}
        strategy_results = {}
        
        strategy_funcs = {
            'semantic': lambda: self.strategies.semantic_search(query, n_results=top_n * 2),
            'hybrid': lambda: self.strategies.hybrid_search(query, n_results=top_n * 2),
            'mmr': lambda: self.strategies.mmr_search(query, n_results=top_n * 2),
        }
        
        if SEARCH.enable_reranking:
            strategy_funcs['rerank'] = lambda: self.strategies.rerank_search(query, n_results=top_n * 2)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_strategy = {
                executor.submit(func): strategy
                for strategy, func in strategy_funcs.items()
            }
            
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    results = future.result()
                    strategy_results[strategy] = results
                    
                    for r in results:
                        doc_id = r['id']
                        if doc_id not in all_results:
                            all_results[doc_id] = {
                                'id': doc_id,
                                'content': r['content'],
                                'metadata': r['metadata'],
                                'scores': {},
                                'ranks': []
                            }
                        
                        # Store the final_score from each strategy
                        score_key = f"{strategy}_score"
                        all_results[doc_id]['scores'][score_key] = r.get('final_score', 
                                                                         r.get('similarity_score', 0))
                        all_results[doc_id]['ranks'].append(r['rank'])
                
                except Exception as e:
                    print(f"⚠️  Strategy '{strategy}' failed: {e}")
        
        # Fuse results using improved RRF + score combination
        fused_results = self._advanced_fusion(all_results)
        return fused_results[:top_n * 2]
    
    def _fast_search(self, query: str, top_n: int) -> List[Dict]:
        """Fast semantic search"""
        raw_results = self.strategies.semantic_search(query, n_results=top_n)
        for r in raw_results:
            r['confidence'] = r['similarity_score']
            r['final_score'] = r['similarity_score']
        return raw_results
    
    def _accurate_search(self, query: str, top_n: int) -> List[Dict]:
        """Accurate search using hybrid + reranking"""
        raw_results = self.strategies.rerank_search(
            query,
            n_results=top_n,
            initial_results=self.strategies.hybrid_search(query, n_results=top_n * 2)
        )
        for r in raw_results:
            r['confidence'] = r.get('rerank_score', r.get('hybrid_score', 0))
        return raw_results
    
    def _advanced_fusion(self, all_results: Dict) -> List[Dict]:
        """
        Advanced fusion combining RRF and score averaging
        
        This ensures parallel search produces scores in the same range as other methods
        """
        fused_results = []
        
        for doc_id, data in all_results.items():
            # Calculate RRF score (rank-based)
            rrf_score = sum(1 / (60 + rank) for rank in data['ranks'])
            
            # Calculate average score across strategies (score-based)
            strategy_scores = list(data['scores'].values())
            avg_score = np.mean(strategy_scores) if strategy_scores else 0
            
            # Combine RRF and average score (weighted)
            # RRF gives better relative ranking, avg_score maintains absolute scale
            data['raw_rrf_score'] = rrf_score
            data['avg_strategy_score'] = avg_score
            fused_results.append(data)
        
        if fused_results:
            # Normalize RRF scores to [0, 1]
            rrf_scores = [r['raw_rrf_score'] for r in fused_results]
            max_rrf = max(rrf_scores)
            min_rrf = min(rrf_scores)
            
            for result in fused_results:
                # Normalize RRF
                if max_rrf > min_rrf:
                    normalized_rrf = (result['raw_rrf_score'] - min_rrf) / (max_rrf - min_rrf)
                else:
                    normalized_rrf = 0.5
                
                # Combine normalized RRF (70%) with average score (30%)
                # This preserves the score range of individual strategies
                combined_score = 0.7 * normalized_rrf + 0.3 * result['avg_strategy_score']
                
                result['confidence'] = combined_score
                result['final_score'] = combined_score
        
        # Sort by combined score
        fused_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(fused_results, 1):
            result['rank'] = i
        
        return fused_results
    
    def _reciprocal_rank_fusion(self, all_results: Dict) -> List[Dict]:
        """Original RRF implementation (kept for compatibility)"""
        fused_results = []
        
        for doc_id, data in all_results.items():
            rrf_score = sum(1 / (60 + rank) for rank in data['ranks'])
            data['raw_rrf_score'] = rrf_score
            fused_results.append(data)
        
        if fused_results:
            raw_scores = [r['raw_rrf_score'] for r in fused_results]
            max_rrf = max(raw_scores)
            min_rrf = min(raw_scores)
            
            for result in fused_results:
                if max_rrf > min_rrf:
                    normalized = (result['raw_rrf_score'] - min_rrf) / (max_rrf - min_rrf)
                else:
                    normalized = 0.5
                
                result['confidence'] = normalized
                result['final_score'] = normalized
        
        fused_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        for i, result in enumerate(fused_results, 1):
            result['rank'] = i
        
        return fused_results