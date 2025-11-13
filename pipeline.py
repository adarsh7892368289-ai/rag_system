from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.extraction import DocumentExtractor
from core.database import ChromaDBManager
from strategies.search_strategies import AdvancedSearchStrategies
from strategies.query_router import QueryRouter


class RAGPipeline:
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.extractor = DocumentExtractor()
        self.db_manager = ChromaDBManager()
        self.db_manager.initialize()
        self.search_strategies = AdvancedSearchStrategies(self.db_manager)
        self.query_router = QueryRouter()
    
    def ingest(self, sources: List[str], reset: bool = False, 
               save_extracted: bool = True, update_mode: str = 'skip',
               max_chunks_per_doc: int = 20):
        """
        Ingest documents into the system
        
        Args:
            sources: List of URLs or file paths
            reset: Clear database before ingestion
            save_extracted: Save extracted documents to JSON
            update_mode: 'skip', 'replace', or 'merge'
            max_chunks_per_doc: Max chunks per document (default: from config)
        """
        if self.verbose:
            print(f"\n📥 Ingesting {len(sources)} source(s)...")
        
        documents = self.extractor.extract_multiple(
            sources, 
            chunk=True, 
            parallel=True,
            max_chunks_per_doc=max_chunks_per_doc
        )
        
        if not documents:
            print("⚠️  No documents extracted")
            return
        
        if save_extracted:
            self.extractor.save_documents(documents)
        
        if reset or not self.db_manager.collection:
            self.db_manager.initialize(reset=reset)
        
        self.db_manager.add_documents(documents, update_mode=update_mode)
        self.search_strategies.invalidate_caches()
        
        if self.verbose:
            print(f"✅ Ingestion complete: {self.db_manager.collection.count()} chunks\n")
    
    def load_from_json(self, folder_path: str, reset: bool = False, 
                      update_mode: str = 'skip'):
        if self.verbose:
            print(f"\n📂 Loading from {folder_path}...")
        
        documents = self.extractor.load_from_folder(folder_path)
        
        if not documents:
            print("⚠️  No documents loaded")
            return
        
        if reset or not self.db_manager.collection:
            self.db_manager.initialize(reset=reset)
        
        self.db_manager.add_documents(documents, update_mode=update_mode)
        self.search_strategies.invalidate_caches()
        
        if self.verbose:
            print(f"✅ Loaded {self.db_manager.collection.count()} chunks\n")
    
    def query(self, query_text: str, top_k: int = 5, mode: str = None, 
             auto_route: bool = True, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        
        if mode is None and auto_route:
            mode = self.query_router.route(query_text)
            if self.verbose:
                print(f"🧭 Query type: '{self.query_router.detect_query_type(query_text)}' → Routing to: '{mode}'")
        elif mode is None:
            mode = 'parallel'
        
        if mode == 'semantic':
            results = self.search_strategies.semantic_search(query_text, n_results=top_k)
        elif mode == 'bm25':
            results = self.search_strategies.bm25_search(query_text, n_results=top_k)
        elif mode == 'hybrid':
            results = self.search_strategies.hybrid_search(query_text, n_results=top_k)
        elif mode == 'mmr':
            results = self.search_strategies.mmr_search(query_text, n_results=top_k)
        elif mode == 'rerank':
            results = self.search_strategies.rerank_search(query_text, n_results=top_k)
        elif mode == 'parallel':
            results = self.search_strategies.parallel_search(query_text, n_results=top_k)
        else:
            results = self.search_strategies.parallel_search(query_text, n_results=top_k)
        
        if min_confidence > 0.0:
            results = [r for r in results if r.get('confidence', 1.0) >= min_confidence]
        
        return results
    
    def query_batch(self, queries: List[str], top_k: int = 5, mode: str = None,
                    auto_route: bool = True, max_workers: int = 4) -> Dict[str, List[Dict[str, Any]]]:
        if self.verbose:
            print(f"\n🔄 Processing {len(queries)} queries...")
        
        results_dict = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {
                executor.submit(self.query, query, top_k, mode, auto_route): query
                for query in queries
            }
            
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results = future.result(timeout=60)
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
            
            if include_metadata:
                header = f"[Source {i} | Confidence: {confidence:.2f}]"
                if source != 'Unknown':
                    sources.add(source)
                    header += f"\n[From: {source}]"
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
        self.db_manager.clear()
        self.db_manager.initialize()
        self.search_strategies.invalidate_caches()
        if self.verbose:
            print("✅ Database cleared\n")
    
    def get_stats(self) -> Dict[str, Any]:
        return self.db_manager.get_stats()