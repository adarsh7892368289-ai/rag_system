from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.extraction import DocumentExtractor
from core.database import MultiCollectionManager
from strategies.search_strategies import AdvancedSearchStrategies
from strategies.query_router import QueryRouter
from strategies.fusion import ResultFusion
from strategies.result_tracker import ResultTracker


class RAGPipeline:
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.extractor = DocumentExtractor()
        self.db_manager = MultiCollectionManager()
        self.db_manager.initialize()
        self.search_strategies = AdvancedSearchStrategies(self.db_manager)
        self.query_router = QueryRouter()
        self.result_tracker = ResultTracker()
        self.fusion = ResultFusion(self.db_manager.embedding_model)
    
    def ingest(self, sources: List[str], reset: bool = False, 
               save_extracted: bool = True, update_mode: str = 'skip'):
        if self.verbose:
            print(f"\n📥 Ingesting {len(sources)} source(s)...")
        
        strategy_documents = self.extractor.extract_multiple(sources)
        
        if not any(strategy_documents.values()):
            print("⚠️  No documents extracted")
            return
        
        if save_extracted:
            self.extractor.save_documents(strategy_documents)
        
        if reset:
            self.db_manager.initialize(reset=reset)
        
        self.db_manager.add_documents_by_strategy(strategy_documents, update_mode=update_mode)
        self.search_strategies.invalidate_caches()
        
        if self.verbose:
            stats = self.db_manager.get_stats()
            print(f"✅ Ingestion complete: {stats['total']} total chunks\n")
    
    def load_from_json(self, folder_path: str, reset: bool = False, 
                      update_mode: str = 'skip'):
        if self.verbose:
            print(f"\n📂 Loading from {folder_path}...")
        
        strategy_documents = self.extractor.load_from_folder(folder_path)
        
        if not any(strategy_documents.values()):
            print("⚠️  No documents loaded")
            return
        
        if reset:
            self.db_manager.initialize(reset=reset)
        
        self.db_manager.add_documents_by_strategy(strategy_documents, update_mode=update_mode)
        self.search_strategies.invalidate_caches()
        
        if self.verbose:
            stats = self.db_manager.get_stats()
            print(f"✅ Loaded {stats['total']} total chunks\n")
    
    def query(self, query_text: str, top_k: int = 5, mode: str = None, 
             auto_route: bool = True, min_confidence: float = 0.0,
             save_results: bool = True) -> List[Dict[str, Any]]:
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if mode is None and auto_route:
            mode = self.query_router.route(query_text)
        elif mode is None:
            mode = 'parallel'
        
        if mode == 'parallel':
            if self.verbose:
                print(f"\n🔍 Running PARALLEL mode (16 searches: 4 chunking × 4 search strategies)")
            
            chunking_results = self.search_strategies.parallel_search_all(query_text, top_k)
            
            if save_results:
                self.result_tracker.save_intermediate_results(
                    query_text, chunking_results, mode, timestamp
                )
            
            final_results = self.fusion.reciprocal_rank_fusion(
                chunking_results,
                top_n=top_k
            )
        
        else:
            if self.verbose:
                print(f"\n🔍 Running {mode.upper()} mode (4 searches: 4 chunking × 1 search strategy)")
            
            chunking_results = self.search_strategies.single_strategy_all_chunking(
                query_text, mode, top_k
            )
            
            if save_results:
                self.result_tracker.save_intermediate_results(
                    query_text, chunking_results, mode, timestamp
                )
            
            final_results = self.fusion.reciprocal_rank_fusion(
                chunking_results,
                top_n=top_k
            )
        
        if min_confidence > 0.0:
            final_results = [r for r in final_results if r.get('confidence', 1.0) >= min_confidence]
        
        if save_results:
            saved_path = self.result_tracker.save_final_results(
                query_text, final_results, mode, timestamp
            )
            if self.verbose:
                print(f"\n💾 Results saved to: {saved_path}")
        
        if self.verbose:
            self._print_results_summary(final_results, chunking_results)
        
        return final_results
    
    def query_batch(self, queries: List[str], top_k: int = 5, mode: str = None,
                    auto_route: bool = True, max_workers: int = 4,
                    save_results: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        if self.verbose:
            print(f"\n🔄 Processing {len(queries)} queries in parallel...")
        
        results_dict = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {
                executor.submit(self.query, query, top_k, mode, auto_route, 0.0, save_results): query
                for query in queries
            }
            
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results = future.result(timeout=120)
                    results_dict[query] = results
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Query '{query}' failed: {e}")
                    results_dict[query] = []
        
        if self.verbose:
            print(f"✅ Batch complete\n")
        
        return results_dict
    
    def build_llm_context(self, results: List[Dict[str, Any]], 
                         max_tokens: int = 4000,
                         include_metadata: bool = True,
                         min_confidence: float = 0.0) -> Dict[str, Any]:
        
        filtered_results = [
            r for r in results 
            if r.get('confidence', 1.0) >= min_confidence
        ]
        
        if not filtered_results:
            return {
                'context': '',
                'chunks_used': 0,
                'chunks_filtered': len(results),
                'total_chars': 0,
                'estimated_tokens': 0,
                'sources': [],
                'avg_confidence': 0.0
            }
        
        char_limit = max_tokens * 4
        context_parts = []
        total_chars = 0
        sources = set()
        chunks_used = 0
        
        for i, result in enumerate(filtered_results, 1):
            content = result['content']
            confidence = result.get('confidence', 1.0)
            source = result.get('metadata', {}).get('source', 'Unknown')
            chunking = result.get('chunking_strategy', 'unknown')
            search = result.get('search_strategy', 'unknown')
            
            if include_metadata:
                header = f"[Result {i} | Confidence: {confidence:.2f} | Chunking: {chunking} | Search: {search}]"
                if source != 'Unknown':
                    sources.add(source)
                    header += f"\n[Source: {source}]"
                chunk_text = f"{header}\n{content}\n"
            else:
                chunk_text = f"{content}\n"
            
            if total_chars + len(chunk_text) > char_limit:
                break
            
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
            chunks_used += 1
        
        context = "\n---\n".join(context_parts)
        
        return {
            'context': context,
            'chunks_used': chunks_used,
            'chunks_filtered': len(results) - len(filtered_results),
            'total_chars': total_chars,
            'estimated_tokens': total_chars // 4,
            'sources': list(sources),
            'avg_confidence': sum(r.get('confidence', 1.0) for r in filtered_results[:chunks_used]) / chunks_used if chunks_used > 0 else 0.0
        }
    
    def format_for_llm(self, query: str, context_dict: Dict[str, Any],
                      system_prompt: str = None) -> str:
        
        if system_prompt is None:
            system_prompt = "You are a helpful assistant. Answer based on the provided context."
        
        prompt = f"""{system_prompt}

Context:
{context_dict['context']}

Question: {query}

Answer:"""
        
        return prompt
    
    def clear_database(self):
        self.db_manager.clear_all()
        self.db_manager.initialize()
        self.search_strategies.invalidate_caches()
        if self.verbose:
            print("✅ All databases cleared\n")
    
    def get_stats(self) -> Dict[str, Any]:
        return self.db_manager.get_stats()
    
    def _print_results_summary(self, final_results: List[Dict], 
                              chunking_results: Dict[str, List[Dict]]):
        print(f"\n📊 Results Summary:")
        print(f"   • Final results: {len(final_results)}")
        
        chunking_dist = {}
        search_dist = {}
        
        for result in final_results:
            chunking = result.get('chunking_strategy', 'unknown')
            search = result.get('search_strategy', result.get('strategies_used', ['unknown']))
            
            chunking_dist[chunking] = chunking_dist.get(chunking, 0) + 1
            
            if isinstance(search, list):
                for s in search:
                    search_dist[s] = search_dist.get(s, 0) + 1
            else:
                search_dist[search] = search_dist.get(search, 0) + 1
        
        print(f"   • Chunking distribution: {dict(chunking_dist)}")
        print(f"   • Search strategy distribution: {dict(search_dist)}")
        
        if final_results:
            avg_confidence = sum(r.get('confidence', 0) for r in final_results) / len(final_results)
            print(f"   • Average confidence: {avg_confidence:.3f}")