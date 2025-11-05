"""
Vector Search System - Phase 1+2 Complete

Key improvements:
1. Metadata flattening for ChromaDB compatibility (Phase 1 - CRITICAL FIX)
2. Metadata reconstruction from registry (Phase 2)
3. Clean, efficient search with all strategies
"""

import json
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from config.settings import SEARCH, DATABASE, EMBEDDING, METADATA
from utils.logger import get_logger, PerformanceLogger
from utils.validators import TextValidator, ConfigValidator
from utils.metadata_manager import get_registry

logger = get_logger("search")


def flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """    
    Args:
        metadata: Original metadata (may have nested dicts/lists)
    
    Returns:
        Flattened metadata compatible with ChromaDB
    
    Example:
        Input:  {'chunking': {'method': 'sentence_aware'}}
        Output: {'chunking_method': 'sentence_aware'}
    """
    flat = {}
    
    for key, value in metadata.items():
        if isinstance(value, dict):
            # Flatten nested dicts
            for nested_key, nested_value in value.items():
                flat_key = f"{key}_{nested_key}"
                if isinstance(nested_value, (str, int, float, bool)) or nested_value is None:
                    flat[flat_key] = nested_value
                else:
                    flat[flat_key] = str(nested_value)
        elif isinstance(value, list):
            # Convert lists to JSON string
            flat[key] = json.dumps(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[key] = value
        else:
            flat[key] = str(value)
    
    return flat


class EmbeddingGenerator:
    """Generate vector embeddings"""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize embedding model"""
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
    ChromaDB manager with Phase 1+2 improvements
    
    - Flattens metadata before storage (Phase 1)
    - Reconstructs full metadata on retrieval (Phase 2)
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize ChromaDB"""
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
        """
        Add documents with FLATTENED metadata (Phase 1 fix)
        
        Args:
            documents: List of dicts with 'id', 'content', 'metadata', 'document_id'
            batch_size: Batch size for processing
        """
        if not self.collection:
            raise ValueError("Create collection first using create_collection()")
        
        batch_size = batch_size or DATABASE.batch_size
        
        logger.info(f"\n📥 Adding {len(documents)} documents...")
        
        # Cache documents for BM25
        self.documents_cache = documents
        
        # Prepare data with FLATTENED metadata (PHASE 1 FIX)
        texts = [doc['content'] for doc in documents]
        ids = [doc['id'] for doc in documents]
        metadatas = [flatten_metadata(doc.get('metadata', {})) for doc in documents]
        
        # Add document_id to metadata for reconstruction
        for i, doc in enumerate(documents):
            if 'document_id' in doc:
                metadatas[i]['document_id'] = doc['document_id']
        
        # Generate embeddings
        logger.info("🔄 Generating embeddings...")
        with PerformanceLogger("Embedding generation"):
            embeddings = self.embedding_model.encode_batch(texts, show_progress=True).tolist()
        
        # Add in batches
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
    
    def search(self, query: str, n_results: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Semantic search with metadata reconstruction (Phase 2)
        
        Args:
            query: Search query
            n_results: Number of results
            filters: Metadata filters
        
        Returns:
            Results with reconstructed metadata
        """
        if not self.collection:
            raise ValueError("Create collection first")
        
        query = TextValidator.validate_query(query)
        ConfigValidator.validate_search_params(n_results)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filters,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format and reconstruct metadata (PHASE 2)
        return self._format_results(results, query)
    
    def _format_results(self, results, query: str) -> List[Dict]:
        """
        Format results with metadata reconstruction (Phase 2)
        
        Reconstructs full metadata by merging:
        - Document-level metadata from registry
        - Chunk-level metadata from ChromaDB
        """
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
            
            # PHASE 2: Reconstruct full metadata from registry
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
            
            # Convert distance to similarity (0-1, higher is better)
            similarity_score = 1 / (1 + distance)
            
            formatted_results.append({
                'id': ids[i],
                'content': content[:300] + "..." if len(content) > 300 else content,
                'full_content': content,
                'metadata': full_metadata,
                'similarity_score': similarity_score,
                'distance': distance,
                'rank': i + 1,
                'query': query
            })
        
        return formatted_results


class QueryRouter:
    """Intelligent query routing"""
    
    def __init__(self):
        """Initialize router"""
        self.routing_rules = {
            'factual': ['what is', 'what are', 'define', 'definition'],
            'explanation': ['how', 'why', 'explain', 'describe'],
            'comparison': ['compare', 'difference', 'versus', 'vs'],
            'listing': ['list', 'types of', 'examples of']
        }
        
        logger.info("🧭 Query router initialized")
    
    def route(self, query: str) -> Tuple[str, Dict]:
        """
        Route query to best search strategy
        
        Args:
            query: Search query
        
        Returns:
            (strategy_name, strategy_params)
        """
        query_lower = query.lower()
        query_type = self._classify_query(query_lower)
        query_length = len(query.split())
        has_specific_terms = self._has_specific_terms(query_lower)
        
        # Route based on analysis
        if query_type == "factual" and has_specific_terms:
            return "hybrid", {"alpha": 0.7}
        elif query_type == "explanation":
            return "mmr", {"lambda_param": 0.7}
        elif query_type == "comparison":
            return "semantic", {}
        elif has_specific_terms and query_length < 5:
            return "hybrid", {"alpha": 0.3}  # BM25-heavy
        else:
            return "semantic", {}
    
    def _classify_query(self, query: str) -> str:
        """Classify query type"""
        for qtype, keywords in self.routing_rules.items():
            if any(query.startswith(kw) for kw in keywords):
                return qtype
        return "general"
    
    def _has_specific_terms(self, query: str) -> bool:
        """Check for specific/technical terms"""
        words = query.split()
        has_acronyms = any(w.isupper() and len(w) >= 2 for w in words)
        has_proper_nouns = any(w[0].isupper() for w in words[1:])
        has_numbers = any(c.isdigit() for c in query)
        return has_acronyms or has_proper_nouns or has_numbers


class AdvancedSearchTechniques:
    """Advanced search methods"""
    
    def __init__(self, db_manager: ChromaDBManager):
        """Initialize advanced search"""
        self.db_manager = db_manager
        self.cross_encoder = None
        self.bm25 = None
        
        logger.info("🚀 Advanced search initialized")
    
    def hybrid_search(self, query: str, n_results: int = 5, alpha: float = 0.7) -> List[Dict]:
        """
        Hybrid search: BM25 + Vector
        
        Args:
            query: Search query
            n_results: Number of results
            alpha: Vector weight (0.0=pure BM25, 1.0=pure vector)
        
        Returns:
            Ranked results
        """
        ConfigValidator.validate_search_params(n_results, alpha=alpha)
        
        logger.debug(f"🔄 Hybrid search (α={alpha})")
        
        # Initialize BM25
        if not self.bm25:
            self._initialize_bm25()
        
        # Get vector results
        vector_results = self.db_manager.search(query, n_results=n_results * 2)
        
        if not vector_results:
            return []
        
        # Calculate BM25 scores
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens) if self.bm25 else []
        
        if not bm25_scores:
            return vector_results[:n_results]
        
        # Normalize BM25
        max_bm25 = max(bm25_scores)
        min_bm25 = min(bm25_scores)
        
        # Calculate hybrid scores
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
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(hybrid_results[:n_results], 1):
            result['rank'] = i
        
        logger.debug(f"✅ Hybrid search: {len(hybrid_results[:n_results])} results")
        return hybrid_results[:n_results]
    
    def mmr_search(self, query: str, n_results: int = 5, lambda_param: float = 0.7) -> List[Dict]:
        """
        MMR search for diverse results
        
        Args:
            query: Search query
            n_results: Number of results
            lambda_param: Relevance vs diversity (0.0=max diversity, 1.0=max relevance)
        
        Returns:
            Diverse results
        """
        ConfigValidator.validate_search_params(n_results, lambda_param=lambda_param)
        
        logger.debug(f"🔄 MMR search (λ={lambda_param})")
        
        # Get candidates
        candidates = self.db_manager.search(
            query,
            n_results=min(n_results * SEARCH.mmr_candidates_multiplier, SEARCH.max_top_k)
        )
        
        if not candidates:
            return []
        
        # Get embeddings
        candidate_texts = [c['full_content'] for c in candidates]
        candidate_embeddings = self.db_manager.embedding_model.encode_batch(candidate_texts)
        
        # MMR selection
        selected = []
        selected_embeddings = []
        remaining = list(range(len(candidates)))
        
        # Select first (most relevant)
        selected.append(0)
        selected_embeddings.append(candidate_embeddings[0])
        remaining.remove(0)
        
        # Iteratively select diverse documents
        while len(selected) < n_results and remaining:
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining:
                relevance = candidates[idx]['similarity_score']
                
                # Max similarity to selected
                candidate_emb = candidate_embeddings[idx].reshape(1, -1)
                selected_embs = np.array(selected_embeddings)
                similarities = cosine_similarity(candidate_emb, selected_embs)[0]
                max_sim = float(np.max(similarities))
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                selected_embeddings.append(candidate_embeddings[best_idx])
                remaining.remove(best_idx)
        
        # Prepare results
        mmr_results = [candidates[i] for i in selected]
        for i, result in enumerate(mmr_results, 1):
            result['mmr_score'] = result['similarity_score']
            result['final_score'] = result['similarity_score']
            result['rank'] = i
        
        logger.debug(f"✅ MMR search: {len(mmr_results)} results")
        return mmr_results
    
    def rerank_search(self, query: str, n_results: int = 5,
                     initial_results: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Re-rank using Cross-Encoder
        
        Args:
            query: Search query
            n_results: Final results
            initial_results: Pre-fetched results
        
        Returns:
            Re-ranked results
        """
        ConfigValidator.validate_search_params(n_results)
        
        logger.debug("🔄 Re-ranking with Cross-Encoder")
        
        if initial_results is None:
            initial_results = self.db_manager.search(
                query,
                n_results=min(SEARCH.rerank_top_k, SEARCH.max_top_k)
            )
        
        if not initial_results:
            return []
        
        # Initialize cross-encoder
        if not self.cross_encoder:
            logger.info(f"Loading Cross-Encoder: {SEARCH.rerank_model}")
            self.cross_encoder = CrossEncoder(SEARCH.rerank_model)
        
        # Prepare pairs
        pairs = [[query, result['full_content']] for result in initial_results]
        
        # Get scores
        with PerformanceLogger("Cross-encoder scoring"):
            scores = self.cross_encoder.predict(pairs)
        
        # Add scores
        for i, result in enumerate(initial_results):
            result['rerank_score'] = float(scores[i])
            result['final_score'] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(initial_results, key=lambda x: x['rerank_score'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked[:n_results], 1):
            result['rank'] = i
        
        logger.debug(f"✅ Re-ranking: {len(reranked[:n_results])} results")
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


class UnifiedSearchEngine:
    """
    Unified search with intelligent routing
    
    Phase 1+2 complete:
    - Metadata flattening (Phase 1)
    - Metadata reconstruction (Phase 2)
    - All search strategies
    """
    
    def __init__(self, db_manager: ChromaDBManager, enable_routing: bool = True):
        """Initialize search engine"""
        self.db_manager = db_manager
        self.advanced = AdvancedSearchTechniques(db_manager)
        self.router = QueryRouter() if enable_routing else None
        self.enable_routing = enable_routing
        
        logger.info("🎯 Unified search engine ready")
        if enable_routing:
            logger.info("   Intelligent routing: ENABLED")
    
    def search(self, query: str, n_results: int = 5,
              strategy: Optional[str] = None,
              strategy_params: Optional[Dict] = None,
              enable_reranking: Optional[bool] = None) -> List[Dict]:
        """
        Universal search with auto-routing
        
        Args:
            query: Search query
            n_results: Results to return
            strategy: Force strategy (None for auto)
            strategy_params: Strategy parameters
            enable_reranking: Enable re-ranking
        
        Returns:
            Search results with reconstructed metadata
        
        Strategies: semantic, hybrid, mmr, bm25_heavy
        """
        query = TextValidator.validate_query(query)
        ConfigValidator.validate_search_params(n_results)
        
        # Route query
        if strategy is None and self.enable_routing and self.router:
            strategy, auto_params = self.router.route(query)
            strategy_params = strategy_params or auto_params
        else:
            strategy = strategy or "semantic"
            strategy_params = strategy_params or {}
        
        logger.info(f"\n🔍 Search: '{query}'")
        logger.info(f"   Strategy: {strategy}")
        
        # Execute search
        with PerformanceLogger(f"Search ({strategy})"):
            if strategy == "semantic":
                results = self.db_manager.search(query, n_results)
            elif strategy == "hybrid":
                alpha = strategy_params.get("alpha", SEARCH.hybrid_alpha)
                results = self.advanced.hybrid_search(query, n_results, alpha)
            elif strategy == "mmr":
                lambda_param = strategy_params.get("lambda_param", SEARCH.mmr_lambda)
                results = self.advanced.mmr_search(query, n_results, lambda_param)
            elif strategy == "bm25_heavy":
                alpha = strategy_params.get("alpha", 0.3)
                results = self.advanced.hybrid_search(query, n_results, alpha)
            else:
                logger.warning(f"Unknown strategy '{strategy}', using semantic")
                results = self.db_manager.search(query, n_results)
        
        # Optional re-ranking
        if enable_reranking is None:
            enable_reranking = SEARCH.use_reranking
        
        if enable_reranking and results:
            results = self.advanced.rerank_search(query, n_results, results)
        
        logger.info(f"✅ Found {len(results)} results")
        
        return results
    
    def compare_strategies(self, query: str, n_results: int = 3) -> Dict[str, List[Dict]]:
        """Compare different search strategies"""
        logger.info(f"\n{'='*70}")
        logger.info(f"Comparing Strategies: '{query}'")
        logger.info(f"{'='*70}")
        
        strategies = {
            "semantic": {},
            "hybrid": {"alpha": 0.7},
            "mmr": {"lambda_param": 0.7},
            "bm25_heavy": {"alpha": 0.3}
        }
        
        comparison = {}
        for strategy, params in strategies.items():
            logger.info(f"\n{strategy.upper()}:")
            results = self.search(
                query,
                n_results=n_results,
                strategy=strategy,
                strategy_params=params,
                enable_reranking=False
            )
            comparison[strategy] = results
            
            if results:
                top = results[0]
                logger.info(f"   Score: {top.get('final_score', 0):.3f}")
                logger.info(f"   {top['content'][:100]}...")
        
        return comparison


def load_documents_from_json(folder_path: str = "data/extracted") -> List[Dict]:
    """
    Load documents from JSON files
    
    Args:
        folder_path: Path to extracted documents
    
    Returns:
        List of document dicts
    """
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
    logger.info("Vector Search - Phase 1+2 Demo")
    logger.info("="*70)
    
    # Initialize
    chroma = ChromaDBManager()
    collection = chroma.create_collection(reset=False)
    
    # Load documents if empty
    if collection.count() == 0:
        documents = load_documents_from_json()
        if documents:
            chroma.add_documents(documents)
    
    # Initialize search
    search_engine = UnifiedSearchEngine(chroma, enable_routing=True)
    
    # Test query
    results = search_engine.search("What is machine learning?", n_results=3)
    
    print(f"\n{'='*70}")
    print("Search Results:")
    print('='*70)
    
    for result in results:
        print(f"\n[{result['rank']}] Score: {result.get('final_score', 0):.3f}")
        print(f"    {result['content']}")
        print(f"    Metadata fields: {list(result['metadata'].keys())}")
    
    logger.info("\n✅ Demo complete")