"""
Production RAG Search Engine - Complete System

Features:
- Parallel multi-strategy execution for best results
- Confidence scoring with all strategy scores
- Result persistence for NLP pipelines
- Analytics and performance tracking
- Metadata reconstruction (Phase 1+2)
"""

import json
import time
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from config.settings import SEARCH, DATABASE, EMBEDDING, METADATA, DATA_DIR
from utils.logger import get_logger, PerformanceLogger
from utils.validators import TextValidator, ConfigValidator
from utils.metadata_manager import get_registry

logger = get_logger("search")


# ============================================================================
# CORE COMPONENTS
# ============================================================================

def flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested metadata for ChromaDB compatibility
    
    Args:
        metadata: Original metadata (may have nested dicts/lists)
    
    Returns:
        Flattened metadata compatible with ChromaDB
    """
    flat = {}
    
    for key, value in metadata.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                flat_key = f"{key}_{nested_key}"
                if isinstance(nested_value, (str, int, float, bool)) or nested_value is None:
                    flat[flat_key] = nested_value
                else:
                    flat[flat_key] = str(nested_value)
        elif isinstance(value, list):
            flat[key] = json.dumps(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[key] = value
        else:
            flat[key] = str(value)
    
    return flat


class EmbeddingGenerator:
    """Generate vector embeddings"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or EMBEDDING.model_name
        
        logger.info(f"🔄 Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, cache_folder=EMBEDDING.cache_dir)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"✅ Model loaded ({self.dimension}D)")
    
    def encode(self, text: str) -> List[float]:
        """Encode single text"""
        return self.model.encode(
            text,
            normalize_embeddings=EMBEDDING.normalize_embeddings,
            show_progress_bar=False
        ).tolist()
    
    def encode_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Encode multiple texts"""
        return self.model.encode(
            texts,
            batch_size=EMBEDDING.batch_size,
            normalize_embeddings=EMBEDDING.normalize_embeddings,
            show_progress_bar=show_progress
        )


class ChromaDBManager:
    """
    ChromaDB manager with metadata flattening and reconstruction
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory or DATABASE.persist_directory
        
        logger.info(f"\n🔵 Initializing ChromaDB: {self.persist_directory}")
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedding_model = EmbeddingGenerator()
        self.collection = None
        self.documents_cache = []
        self.registry = get_registry(METADATA.registry_dir) if METADATA.enable_registry else None
        
        logger.info("✅ ChromaDB initialized")
    
    def create_collection(self, collection_name: Optional[str] = None, reset: bool = False):
        """Create or get collection"""
        collection_name = collection_name or DATABASE.collection_name
        
        if reset:
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"🗑️  Deleted collection: {collection_name}")
            except:
                pass
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata=DATABASE.get_collection_metadata()
        )
        
        logger.info(f"✅ Collection '{collection_name}' ready")
        logger.info(f"   Documents: {self.collection.count()}")
        
        return self.collection
    
    def add_documents(self, documents: List[Dict], batch_size: Optional[int] = None):
        """Add documents with flattened metadata"""
        if not self.collection:
            raise ValueError("Create collection first using create_collection()")
        
        batch_size = batch_size or DATABASE.batch_size
        
        logger.info(f"\n📥 Adding {len(documents)} documents...")
        
        self.documents_cache = documents
        
        texts = [doc['content'] for doc in documents]
        ids = [doc['id'] for doc in documents]
        
        # Prepare metadata with proper types for ChromaDB
        metadatas = []
        for doc in documents:
            flat_meta = flatten_metadata(doc.get('metadata', {}))
            if 'document_id' in doc:
                flat_meta['document_id'] = doc['document_id']
            metadatas.append(flat_meta)
        
        logger.info("🔄 Generating embeddings...")
        with PerformanceLogger("Embedding generation"):
            embeddings = self.embedding_model.encode_batch(texts, show_progress=True).tolist()
        
        logger.info(f"💾 Storing in database (batch_size={batch_size})...")
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            
            self.collection.add(
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            
            if end_idx % (batch_size * 10) == 0 or end_idx == len(documents):
                logger.info(f"   ✓ {end_idx}/{len(documents)}")
        
        logger.info(f"✅ Added {self.collection.count()} documents")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Semantic search with metadata reconstruction"""
        if not self.collection:
            raise ValueError("Create collection first")
        
        query = TextValidator.validate_query(query)
        ConfigValidator.validate_search_params(n_results)
        
        query_embedding = self.embedding_model.encode(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        return self._format_results(results, query)
    
    def _format_results(self, results, query: str) -> List[Dict]:
        """Format results with metadata reconstruction"""
        if not results or not results.get('ids') or not results['ids'][0]:
            return []
        
        ids = results['ids'][0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        formatted_results = []
        for i in range(len(ids)):
            content = documents[i]
            distance = distances[i]
            chunk_metadata = metadatas[i]
            
            # Reconstruct full metadata from registry
            if METADATA.reconstruct_on_search and self.registry:
                document_id = chunk_metadata.get('document_id')
                if document_id:
                    full_metadata = self.registry.reconstruct_chunk_metadata(
                        document_id,
                        chunk_metadata
                    )
                else:
                    full_metadata = chunk_metadata
            else:
                full_metadata = chunk_metadata
            
            similarity_score = 1 / (1 + distance)
            
            formatted_results.append({
                'id': ids[i],
                'content': content,
                'metadata': full_metadata,
                'similarity_score': similarity_score,
                'distance': distance,
                'rank': i + 1,
                'query': query
            })
        
        return formatted_results


# ============================================================================
# ADVANCED SEARCH STRATEGIES
# ============================================================================

class AdvancedSearchStrategies:
    """All search strategies with parallel execution"""
    
    def __init__(self, db_manager: ChromaDBManager):
        self.db_manager = db_manager
        self.cross_encoder = None
        self.bm25 = None
        
        logger.info("🚀 Advanced search strategies initialized")
    
    def hybrid_search(self, query: str, n_results: int = 5, alpha: float = 0.7) -> List[Dict]:
        """Hybrid search: BM25 + Vector"""
        ConfigValidator.validate_search_params(n_results, alpha=alpha)
        
        if not self.bm25:
            self._initialize_bm25()
        
        vector_results = self.db_manager.search(query, n_results=n_results * 2)
        
        if not vector_results or not self.bm25:
            return vector_results[:n_results]
        
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        max_bm25 = max(bm25_scores)
        min_bm25 = min(bm25_scores)
        
        hybrid_results = []
        for result in vector_results:
            doc_idx = next(
                (i for i, doc in enumerate(self.db_manager.documents_cache)
                 if doc['id'] == result['id']),
                None
            )
            
            if doc_idx is not None:
                bm25_score = ((bm25_scores[doc_idx] - min_bm25) / (max_bm25 - min_bm25) 
                             if max_bm25 > min_bm25 else 0)
                vector_score = result['similarity_score']
                
                hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score
                
                result['bm25_score'] = float(bm25_score)
                result['vector_score'] = float(vector_score)
                result['hybrid_score'] = float(hybrid_score)
                result['final_score'] = float(hybrid_score)
                hybrid_results.append(result)
        
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        for i, result in enumerate(hybrid_results[:n_results], 1):
            result['rank'] = i
        
        return hybrid_results[:n_results]
    
    def mmr_search(self, query: str, n_results: int = 5, lambda_param: float = 0.7) -> List[Dict]:
        """MMR search for diverse results"""
        ConfigValidator.validate_search_params(n_results, lambda_param=lambda_param)
        
        candidates = self.db_manager.search(
            query,
            n_results=min(n_results * SEARCH.mmr_candidates_multiplier, SEARCH.max_top_k)
        )
        
        if not candidates:
            return []
        
        candidate_texts = [c['content'] for c in candidates]
        candidate_embeddings = self.db_manager.embedding_model.encode_batch(candidate_texts)
        
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
                relevance = candidates[idx]['similarity_score']
                
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
        for i, result in enumerate(mmr_results, 1):
            result['mmr_score'] = result['similarity_score']
            result['final_score'] = result['similarity_score']
            result['rank'] = i
        
        return mmr_results
    
    def rerank_search(self, query: str, n_results: int = 5,
                     initial_results: Optional[List[Dict]] = None) -> List[Dict]:
        """Re-rank using Cross-Encoder"""
        ConfigValidator.validate_search_params(n_results)
        
        if initial_results is None:
            initial_results = self.db_manager.search(
                query,
                n_results=min(SEARCH.rerank_top_k, SEARCH.max_top_k)
            )
        
        if not initial_results:
            return []
        
        if not self.cross_encoder:
            logger.info(f"Loading Cross-Encoder: {SEARCH.rerank_model}")
            self.cross_encoder = CrossEncoder(SEARCH.rerank_model)
        
        pairs = [[query, result['content']] for result in initial_results]
        
        with PerformanceLogger("Cross-encoder scoring"):
            scores = self.cross_encoder.predict(pairs)
        
        for i, result in enumerate(initial_results):
            result['rerank_score'] = float(scores[i])
            result['final_score'] = float(scores[i])
        
        reranked = sorted(initial_results, key=lambda x: x['rerank_score'], reverse=True)
        
        for i, result in enumerate(reranked[:n_results], 1):
            result['rank'] = i
        
        return reranked[:n_results]
    
    def _initialize_bm25(self):
        """Initialize BM25 index"""
        if not self.db_manager.documents_cache:
            logger.warning("⚠️  No documents for BM25")
            return
        
        logger.info("Building BM25 index...")
        corpus = [doc['content'] for doc in self.db_manager.documents_cache]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"✅ BM25 index built ({len(corpus)} docs)")


# ============================================================================
# RESULT TRACKING & ANALYTICS
# ============================================================================

class SearchResult:
    """Enhanced search result with confidence scoring"""
    
    def __init__(self, content: str, metadata: Dict, scores: Dict[str, float],
                 rank: int, query: str):
        self.content = content
        self.metadata = metadata
        self.scores = scores
        self.rank = rank
        self.query = query
        self.confidence = self._calculate_confidence()
    
    def _calculate_confidence(self) -> float:
        """Calculate unified confidence score from all strategies"""
        if not self.scores:
            return 0.0
        
        weights = {
            'semantic_score': 0.3,
            'hybrid_score': 0.3,
            'rerank_score': 0.25,
            'mmr_score': 0.15
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for score_type, weight in weights.items():
            if score_type in self.scores:
                weighted_sum += self.scores[score_type] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'metadata': self.metadata,
            'scores': self.scores,
            'confidence': self.confidence,
            'rank': self.rank,
            'query': self.query
        }
    
    def __repr__(self) -> str:
        return f"SearchResult(rank={self.rank}, confidence={self.confidence:.3f})"


class ResultTracker:
    """Track and persist search results"""
    
    def __init__(self, results_dir: Optional[str] = None):
        self.results_dir = Path(results_dir or DATA_DIR / "search_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.history_file = self.results_dir / "search_history.jsonl"
        
        logger.info(f"📊 Result tracker: {self.results_dir}")
    
    def save_results(self, query: str, results: List[SearchResult],
                    strategy_used: str, execution_time: float,
                    metadata: Optional[Dict] = None) -> str:
        """Save search results"""
        timestamp = datetime.now().isoformat()
        result_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result_data = {
            'result_id': result_id,
            'timestamp': timestamp,
            'query': query,
            'strategy': strategy_used,
            'execution_time': execution_time,
            'num_results': len(results),
            'avg_confidence': sum(r.confidence for r in results) / len(results) if results else 0,
            'results': [r.to_dict() for r in results],
            'metadata': metadata or {}
        }
        
        result_file = self.results_dir / f"result_{result_id}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        with open(self.history_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'result_id': result_id,
                'timestamp': timestamp,
                'query': query,
                'strategy': strategy_used,
                'num_results': len(results),
                'avg_confidence': result_data['avg_confidence'],
                'execution_time': execution_time
            }) + '\n')
        
        logger.info(f"💾 Results saved: {result_file}")
        return str(result_file)
    
    def get_analytics(self) -> Dict:
        """Get search analytics"""
        if not self.history_file.exists():
            return {'total_searches': 0, 'message': 'No search history yet'}
        
        history = []
        with open(self.history_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    history.append(json.loads(line))
                except:
                    pass
        
        if not history:
            return {'total_searches': 0, 'message': 'No valid search history'}
        
        strategies = Counter(h['strategy'] for h in history)
        avg_results = sum(h['num_results'] for h in history) / len(history)
        avg_confidence = sum(h['avg_confidence'] for h in history) / len(history)
        avg_time = sum(h['execution_time'] for h in history) / len(history)
        
        return {
            'total_searches': len(history),
            'strategies_used': dict(strategies),
            'avg_results_per_query': round(avg_results, 2),
            'avg_confidence_score': round(avg_confidence, 3),
            'avg_execution_time': round(avg_time, 3),
            'last_search': history[-1] if history else None
        }
    
    def export_for_nlp(self, result_id: str, format: str = 'context') -> Optional[str]:
        """Export results in NLP-friendly format"""
        result_file = self.results_dir / f"result_{result_id}.json"
        
        if not result_file.exists():
            logger.error(f"Result not found: {result_id}")
            return None
        
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = data['results']
        
        if format == 'context':
            context = "\n\n".join([
                f"[Source {i+1}] (Confidence: {r['confidence']:.2f})\n{r['content']}"
                for i, r in enumerate(results)
            ])
            return context
        
        elif format == 'qa':
            qa_format = {
                'question': data['query'],
                'contexts': [r['content'] for r in results],
                'confidence_scores': [r['confidence'] for r in results],
                'metadata': [r['metadata'] for r in results]
            }
            return json.dumps(qa_format, indent=2)
        
        elif format == 'chunks':
            return json.dumps([
                {
                    'text': r['content'],
                    'confidence': r['confidence'],
                    'metadata': r['metadata']
                }
                for r in results
            ], indent=2)
        
        return None


# ============================================================================
# UNIFIED SEARCH ENGINE
# ============================================================================

class UnifiedSearchEngine:
    """
    Production RAG search engine with parallel multi-strategy execution
    """
    
    def __init__(self, db_manager: ChromaDBManager, confidence_threshold: float = 0.3):
        self.db_manager = db_manager
        self.strategies = AdvancedSearchStrategies(db_manager)
        self.tracker = ResultTracker()
        self.confidence_threshold = confidence_threshold
        
        logger.info("🎯 Unified search engine ready")
        logger.info(f"   Confidence threshold: {confidence_threshold}")
    
    def search(self, query: str, top_n: int = 5, mode: str = 'parallel',
              save_results: bool = True) -> Tuple[List[SearchResult], Dict]:
        """
        Intelligent search with parallel strategy execution
        
        Args:
            query: Search query
            top_n: Number of results
            mode: 'parallel' (all strategies), 'fast', 'accurate', 'comprehensive'
            save_results: Save results to disk
        
        Returns:
            (results, metadata)
        """
        ConfigValidator.validate_search_params(top_n)
        
        start_time = time.time()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🔍 SEARCH: '{query}'")
        logger.info(f"   Mode: {mode} | Top N: {top_n}")
        logger.info('='*70)
        
        if mode == 'parallel':
            results = self._parallel_search(query, top_n)
            strategy_used = 'parallel_fusion'
        elif mode == 'fast':
            results = self._fast_search(query, top_n)
            strategy_used = 'semantic'
        elif mode == 'accurate':
            results = self._accurate_search(query, top_n)
            strategy_used = 'hybrid+rerank'
        else:
            results = self._parallel_search(query, top_n)
            strategy_used = 'parallel_fusion'
        
        execution_time = time.time() - start_time
        
        filtered_results = [r for r in results if r.confidence >= self.confidence_threshold]
        
        logger.info(f"\n✅ Search complete:")
        logger.info(f"   Strategy: {strategy_used}")
        logger.info(f"   Total candidates: {len(results)}")
        logger.info(f"   After filtering (≥{self.confidence_threshold}): {len(filtered_results)}")
        logger.info(f"   Execution time: {execution_time:.3f}s")
        
        if filtered_results:
            avg_conf = sum(r.confidence for r in filtered_results) / len(filtered_results)
            logger.info(f"   Avg confidence: {avg_conf:.3f}")
        
        metadata = {
            'strategy': strategy_used,
            'mode': mode,
            'total_candidates': len(results),
            'filtered_results': len(filtered_results),
            'confidence_threshold': self.confidence_threshold,
            'execution_time': execution_time
        }
        
        if save_results and filtered_results:
            result_file = self.tracker.save_results(
                query=query,
                results=filtered_results,
                strategy_used=strategy_used,
                execution_time=execution_time,
                metadata=metadata
            )
            metadata['result_file'] = result_file
        
        return filtered_results, metadata
    
    def _parallel_search(self, query: str, top_n: int) -> List[SearchResult]:
        """Run all strategies in parallel and fuse results"""
        with PerformanceLogger("Parallel multi-strategy search"):
            all_results = {}
            
            # Define strategies to run in parallel
            strategy_funcs = {
                'semantic': lambda: self.db_manager.search(query, n_results=top_n * 2),
                'hybrid': lambda: self.strategies.hybrid_search(query, n_results=top_n * 2),
                'mmr': lambda: self.strategies.mmr_search(query, n_results=top_n * 2),
                'rerank': lambda: self.strategies.rerank_search(query, n_results=top_n * 2)
            }
            
            # Execute in parallel
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
                                    'content': r['content'],
                                    'metadata': r['metadata'],
                                    'scores': {},
                                    'ranks': []
                                }
                            
                            score_key = f"{strategy}_score"
                            all_results[doc_id]['scores'][score_key] = r.get('final_score', 
                                                                             r.get('similarity_score', 0))
                            all_results[doc_id]['ranks'].append(r['rank'])
                            
                        logger.info(f"   ✓ {strategy}: {len(results)} results")
                    except Exception as e:
                        logger.warning(f"   ✗ {strategy} failed: {e}")
            
            # Fuse results using Reciprocal Rank Fusion
            fused_results = []
            for doc_id, data in all_results.items():
                rrf_score = sum(1 / (60 + rank) for rank in data['ranks'])
                data['scores']['fusion_score'] = rrf_score
                
                fused_results.append(
                    SearchResult(
                        content=data['content'],
                        metadata=data['metadata'],
                        scores=data['scores'],
                        rank=0,
                        query=query
                    )
                )
            
            fused_results.sort(key=lambda x: x.confidence, reverse=True)
            
            for i, result in enumerate(fused_results[:top_n * 2], 1):
                result.rank = i
            
            logger.info(f"   ✓ Fusion complete: {len(fused_results)} unique results")
            return fused_results[:top_n * 2]
    
    def _fast_search(self, query: str, top_n: int) -> List[SearchResult]:
        """Fast semantic search"""
        with PerformanceLogger("Fast search"):
            raw_results = self.db_manager.search(query, n_results=top_n)
        
        return [
            SearchResult(
                content=r['content'],
                metadata=r['metadata'],
                scores={'semantic_score': r['similarity_score']},
                rank=i+1,
                query=query
            )
            for i, r in enumerate(raw_results)
        ]
    
    def _accurate_search(self, query: str, top_n: int) -> List[SearchResult]:
        """Accurate hybrid + rerank search"""
        with PerformanceLogger("Accurate search"):
            raw_results = self.strategies.rerank_search(
                query,
                n_results=top_n,
                initial_results=self.strategies.hybrid_search(query, n_results=top_n * 2)
            )
        
        return [
            SearchResult(
                content=r['content'],
                metadata=r['metadata'],
                scores={
                    'hybrid_score': r.get('hybrid_score', 0),
                    'rerank_score': r.get('rerank_score', 0)
                },
                rank=i+1,
                query=query
            )
            for i, r in enumerate(raw_results)
        ]
    
    def get_analytics(self) -> Dict:
        """Get search analytics"""
        return self.tracker.get_analytics()
    
    def export_for_nlp(self, result_id: str, format: str = 'context') -> Optional[str]:
        """Export results for NLP"""
        return self.tracker.export_for_nlp(result_id, format)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_documents_from_json(folder_path: str = "data/extracted") -> List[Dict]:
    """Load documents from JSON files"""
    logger.info(f"\n📂 Loading from: {folder_path}")
    
    json_files = [f for f in Path(folder_path).glob("*.json") 
                  if f.name != "_index.json"]
    
    if not json_files:
        logger.warning(f"⚠️  No JSON files in {folder_path}")
        return []
    
    all_documents = []
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for chunk in data.get('chunks', []):
            doc = {
                'id': chunk['doc_id'],
                'content': chunk['content'],
                'metadata': chunk.get('metadata', {}),
                'document_id': chunk.get('document_id', '')
            }
            all_documents.append(doc)
    
    logger.info(f"✅ Loaded {len(all_documents)} chunks from {len(json_files)} files")
    return all_documents


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("RAG Search Engine - Demo")
    logger.info("="*70)
    
    # Initialize
    db = ChromaDBManager()
    collection = db.create_collection(reset=False)
    
    # Load documents if empty
    if collection.count() == 0:
        documents = load_documents_from_json()
        if documents:
            db.add_documents(documents)
    
    # Initialize search
    search_engine = UnifiedSearchEngine(db, confidence_threshold=0.3)
    
    # Test parallel search
    results, metadata = search_engine.search(
        "What is machine learning?",
        top_n=5,
        mode='parallel'
    )
    
    print(f"\n{'='*70}")
    print("SEARCH RESULTS")
    print('='*70)
    
    for result in results:
        print(f"\n[{result.rank}] Confidence: {result.confidence:.3f}")
        print(f"Content: {result.content[:200]}...")
        print(f"All Scores: {result.scores}")
    
    print(f"\n{'='*70}")
    print("ANALYTICS")
    print('='*70)
    analytics = search_engine.get_analytics()
    print(json.dumps(analytics, indent=2))