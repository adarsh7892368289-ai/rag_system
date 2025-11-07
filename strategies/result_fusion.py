import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.settings import SEARCH

class UnifiedSearchEngine:
    def __init__(self, db_manager, strategies):
        self.db_manager = db_manager
        self.strategies = strategies
        self.confidence_threshold = SEARCH.confidence_threshold
        print(f"🎯 Search engine ready (confidence threshold: {self.confidence_threshold})")
    
    def search(self, query: str, top_n: int = None, mode: str = None) -> List[Dict]:
        if top_n is None:
            top_n = SEARCH.top_k
        if mode is None:
            mode = SEARCH.default_mode
        
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"🔍 SEARCH: '{query}'")
        print(f"   Mode: {mode} | Top N: {top_n}")
        print('='*70)
        
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
        
        filtered_results = [r for r in results 
                           if r.get('confidence', r.get('final_score', 0)) >= self.confidence_threshold]
        
        print(f"\n✅ Search complete:")
        print(f"   Total candidates: {len(results)}")
        print(f"   After filtering: {len(filtered_results)}")
        print(f"   Execution time: {execution_time:.3f}s\n")
        
        return filtered_results
    
    def _parallel_search(self, query: str, top_n: int) -> List[Dict]:
        all_results = {}
        
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
                        
                        score_key = f"{strategy}_score"
                        all_results[doc_id]['scores'][score_key] = r.get('final_score', 
                                                                         r.get('similarity_score', 0))
                        all_results[doc_id]['ranks'].append(r['rank'])
                    
                    print(f"   ✓ {strategy}: {len(results)} results")
                except Exception as e:
                    print(f"   ✗ {strategy} failed: {e}")
        
        fused_results = self._reciprocal_rank_fusion(all_results)
        
        return fused_results[:top_n * 2]
    
    def _fast_search(self, query: str, top_n: int) -> List[Dict]:
        raw_results = self.strategies.semantic_search(query, n_results=top_n)
        for r in raw_results:
            r['confidence'] = r['similarity_score']
            r['final_score'] = r['similarity_score']
        return raw_results
    
    def _accurate_search(self, query: str, top_n: int) -> List[Dict]:
        raw_results = self.strategies.rerank_search(
            query,
            n_results=top_n,
            initial_results=self.strategies.hybrid_search(query, n_results=top_n * 2)
        )
        for r in raw_results:
            r['confidence'] = r.get('rerank_score', r.get('hybrid_score', 0))
        return raw_results
    
    def _reciprocal_rank_fusion(self, all_results: Dict) -> List[Dict]:
        fused_results = []
        
        for doc_id, data in all_results.items():
            rrf_score = sum(1 / (60 + rank) for rank in data['ranks'])
            data['confidence'] = rrf_score
            data['final_score'] = rrf_score
            fused_results.append(data)
        
        fused_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        for i, result in enumerate(fused_results, 1):
            result['rank'] = i
        
        return fused_results