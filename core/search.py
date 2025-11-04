"""
Vector Search System with Intelligent Routing

Features:
- Multiple search strategies (semantic, hybrid, MMR, re-ranking, filtered)
- Intelligent query routing (automatically selects best strategy)
- Performance optimizations (caching, batch processing)
- Rich result metadata and scoring
"""

import json
import re
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from collections import Counter
from functools import lru_cache
import numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from config.settings import (
    SEARCH, DATABASE, EMBEDDING, 
    QUERY_ROUTING_RULES, SEARCH_STRATEGY_MAP
)
from utils.logger import get_logger, PerformanceLogger
from utils.validators import TextValidator, ConfigValidator

logger = get_logger("search")


class EmbeddingGenerator:
    """
    Generate vector embeddings from text
    
    Supports multiple models and caching for performance.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding generator
        
        Args:
            model_name: Hugging Face model name (uses config default if None)
        """
        self.model_name = model_name or EMBEDDING.model_name
        
        logger.info(f"🔄 Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(
            self.model_name,
            cache_folder=EMBEDDING.cache_dir
        )
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"✅ Model loaded ({self.dimension} dimensions)")
    
    def encode(self, text: str) -> List[float]:
        """Convert single text to vector"""
        return self.model.encode(
            text,
            normalize_embeddings=EMBEDDING.normalize_embeddings,
            show_progress_bar=False
        ).tolist()
    
    def encode_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Convert multiple texts to vectors (batch processing)"""
        return self.model.encode(
            texts,
            batch_size=EMBEDDING.batch_size,
            normalize_embeddings=EMBEDDING.normalize_embeddings,
            show_progress_bar=show_progress
        )


class ChromaDBManager:
    """
    Manage ChromaDB vector database with advanced search capabilities
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB manager
        
        Args:
            persist_directory: Database directory (uses config default if None)
        """
        self.persist_directory = persist_directory or DATABASE.persist_directory
        
        logger.info(f"\n🔵 Initializing ChromaDB: {self.persist_directory}")
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedding_model = EmbeddingGenerator()
        self.collection = None
        self.documents_cache = []  # For BM25 and analysis
        
        logger.info("✅ ChromaDB initialized")
    
    def create_collection(self, collection_name: Optional[str] = None, 
                         reset: bool = False):
        """
        Create or get collection with optimized settings
        
        Args:
            collection_name: Collection name (uses config default if None)
            reset: Delete existing collection if True
        """
        collection_name = collection_name or DATABASE.collection_name
        
        if reset:
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"🗑️  Deleted existing collection: {collection_name}")
            except:
                pass
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata=DATABASE.get_collection_metadata()
        )
        
        logger.info(f"✅ Collection '{collection_name}' ready")
        logger.info(f"   Documents in collection: {self.collection.count()}")
        
        return self.collection
    
    def add_documents(self, documents: List[Dict], batch_size: Optional[int] = None):
        """
        Add documents with embeddings to ChromaDB
        
        Args:
            documents: List of document dicts with 'id', 'content', 'metadata'
            batch_size: Batch size for processing (uses config default if None)
        """
        if not self.collection:
            raise ValueError("Create collection first using create_collection()")
        
        batch_size = batch_size or DATABASE.batch_size
        
        logger.info(f"\n📥 Adding {len(documents)} documents...")
        
        # Cache documents for BM25
        self.documents_cache = documents
        
        # Prepare data
        texts = [doc['content'] for doc in documents]
        ids = [doc['id'] for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        
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
                logger.info(f"   ✓ {end_idx}/{len(documents)} documents")
        
        logger.info(f"✅ Added {self.collection.count()} documents to collection")
    
    def search(self, query: str, n_results: int = 5, 
              filters: Optional[Dict] = None) -> List[Dict]:
        """
        Basic semantic search using cosine similarity
        
        Args:
            query: Search query
            n_results: Number of results to return
            filters: Metadata filters (optional)
        
        Returns:
            List of search results with scores and metadata
        """
        if not self.collection:
            raise ValueError("Create collection first")
        
        # Validate query
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
        
        # Format results
        return self._format_results(results, query)
    
    def _format_results(self, results, query: str) -> List[Dict]:
        """Format ChromaDB results into standardized structure"""
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
            
            # Convert distance to similarity score (0-1, higher is better)
            similarity_score = 1 / (1 + distance)
            
            formatted_results.append({
                'id': ids[i],
                'content': content[:300] + "..." if len(content) > 300 else content,
                'full_content': content,
                'metadata': metadatas[i],
                'similarity_score': similarity_score,
                'distance': distance,
                'rank': i + 1,
                'query': query
            })
        
        return formatted_results

class QueryRouter:
    """
    Intelligent query routing - automatically selects best search strategy
    
    Analyzes query characteristics and routes to optimal search method.
    """
    
    def __init__(self):
        """Initialize query router"""
        self.routing_rules = QUERY_ROUTING_RULES
        self.strategy_map = SEARCH_STRATEGY_MAP
        
        logger.info("🧭 Query router initialized")
    
    def route(self, query: str) -> Tuple[str, Dict]:
        """
        Route query to best search strategy
        
        Args:
            query: Search query
        
        Returns:
            Tuple of (strategy_name, strategy_params)
        """
        query_lower = query.lower()
        
        # Analyze query characteristics
        query_type = self._classify_query_type(query_lower)
        query_length = len(query.split())
        has_specific_terms = self._has_specific_terms(query_lower)
        
        # Route based on analysis
        if query_type == "factual" and has_specific_terms:
            strategy = "hybrid"
            params = {"alpha": 0.7}  # Balanced
            
        elif query_type == "explanation":
            strategy = "mmr"
            params = {"lambda_param": 0.7}  # Diverse explanations
            
        elif query_type == "comparison":
            strategy = "semantic"
            params = {}
            
        elif query_type == "listing":
            strategy = "hybrid"
            params = {"alpha": 0.6}  # More keyword-focused
            
        elif has_specific_terms and query_length < 5:
            strategy = "bm25_heavy"
            params = {"alpha": 0.3}  # Keyword-dominant
            
        else:
            strategy = "semantic"
            params = {}
        
        logger.debug(f"🧭 Routed query to '{strategy}' strategy (type={query_type})")
        
        return strategy, params
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query into type"""
        # Check for question keywords
        for keyword_type, keywords in self.routing_rules.items():
            if any(query.startswith(kw) for kw in keywords):
                return keyword_type.replace('_keywords', '')
        
        return "general"
    
    def _has_specific_terms(self, query: str) -> bool:
        """Check if query contains specific/technical terms"""
        # Look for capitalized words, acronyms, or technical patterns
        words = query.split()
        
        # Check for acronyms (all caps, 2+ letters)
        has_acronyms = any(w.isupper() and len(w) >= 2 for w in words)
        
        # Check for capitalized terms (not at start)
        has_proper_nouns = any(w[0].isupper() for w in words[1:])
        
        # Check for numbers
        has_numbers = any(char.isdigit() for char in query)
        
        # Check for technical symbols
        has_symbols = any(char in query for char in ['_', '-', '/', ':'])
        
        return has_acronyms or has_proper_nouns or has_numbers or has_symbols


class AdvancedSearchTechniques:
    """
    Advanced search methods for optimal retrieval
    
    Implements:
    1. Hybrid Search (BM25 + Vector)
    2. MMR (Maximal Marginal Relevance)
    3. Re-ranking (Cross-Encoder)
    4. Filtered Search
    """
    
    def __init__(self, db_manager: ChromaDBManager):
        """
        Initialize advanced search
        
        Args:
            db_manager: ChromaDB manager instance
        """
        self.db_manager = db_manager
        self.cross_encoder = None
        self.bm25 = None
        self._embedding_cache = {}
        
        logger.info("🚀 Advanced search techniques initialized")
    
    def hybrid_search(self, query: str, n_results: int = 5, 
                     alpha: float = 0.7) -> List[Dict]:
        """
        Hybrid search combining BM25 (keyword) + Vector (semantic)
        
        Args:
            query: Search query
            n_results: Results to return
            alpha: Weight for vector search (0.0=pure BM25, 1.0=pure vector)
        
        Returns:
            Ranked results with hybrid scores
        """
        ConfigValidator.validate_search_params(n_results, alpha=alpha)
        
        logger.debug(f"🔄 Hybrid search (α={alpha})")
        
        # Initialize BM25 if needed
        if not self.bm25:
            self._initialize_bm25()
        
        # Get vector search results (retrieve more candidates)
        vector_results = self.db_manager.search(query, n_results=n_results * 2)
        
        if not vector_results:
            return []
        
        # Calculate BM25 scores
        query_tokens = query.lower().split()
        if self.bm25 is None:
            return []
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize BM25 scores
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        min_bm25 = min(bm25_scores)
        
        # Calculate hybrid scores
        hybrid_results = []
        for result in vector_results:
            # Find document index
            doc_idx = next(
                (i for i, doc in enumerate(self.db_manager.documents_cache)
                 if doc['id'] == result['id']),
                None
            )
            
            if doc_idx is not None:
                # Normalize BM25 score to [0, 1]
                bm25_score = (bm25_scores[doc_idx] - min_bm25) / (max_bm25 - min_bm25) if max_bm25 > min_bm25 else 0
                vector_score = result['similarity_score']
                
                # Combine scores
                hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score
                
                result['bm25_score'] = float(bm25_score)
                result['vector_score'] = float(vector_score)
                result['hybrid_score'] = float(hybrid_score)
                result['final_score'] = float(hybrid_score)
                hybrid_results.append(result)
        
        # Sort by hybrid score
        hybrid_results = sorted(hybrid_results, 
                               key=lambda x: x['hybrid_score'], 
                               reverse=True)
        
        # Update ranks
        for i, result in enumerate(hybrid_results[:n_results], 1):
            result['rank'] = i
        
        logger.debug(f"✅ Hybrid search complete: {len(hybrid_results[:n_results])} results")
        return hybrid_results[:n_results]
    
    def mmr_search(self, query: str, n_results: int = 5,
                  lambda_param: float = 0.7) -> List[Dict]:
        """
        MMR search for diverse results
        
        Args:
            query: Search query
            n_results: Results to return
            lambda_param: Relevance vs diversity (0.0=max diversity, 1.0=max relevance)
        
        Returns:
            Diverse results ranked by MMR
        """
        ConfigValidator.validate_search_params(n_results, lambda_param=lambda_param)
        
        logger.debug(f"🔄 MMR search (λ={lambda_param})")
        
        # Get more candidates than needed
        candidates = self.db_manager.search(
            query,
            n_results=min(n_results * SEARCH.mmr_candidates_multiplier, SEARCH.max_top_k)
        )
        
        if not candidates:
            return []
        
        # Get embeddings for all candidates
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
                # Relevance to query
                relevance = candidates[idx]['similarity_score']
                
                # Max similarity to already selected documents
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
        
        logger.debug(f"✅ MMR search complete: {len(mmr_results)} diverse results")
        return mmr_results
    
    def rerank_search(self, query: str, n_results: int = 5,
                     initial_results: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Re-rank results using Cross-Encoder for higher accuracy
        
        Args:
            query: Search query
            n_results: Final results to return
            initial_results: Pre-fetched results (fetches if None)
        
        Returns:
            Re-ranked results with cross-encoder scores
        """
        ConfigValidator.validate_search_params(n_results)
        
        logger.debug("🔄 Re-ranking with Cross-Encoder")
        
        # Get initial results if not provided
        if initial_results is None:
            initial_results = self.db_manager.search(
                query,
                n_results=min(SEARCH.rerank_top_k, SEARCH.max_top_k)
            )
        
        if not initial_results:
            return []
        
        # Initialize cross-encoder if needed
        if not self.cross_encoder:
            logger.info(f"Loading Cross-Encoder: {SEARCH.rerank_model}")
            self.cross_encoder = CrossEncoder(SEARCH.rerank_model)
        
        # Prepare pairs
        pairs = [[query, result['full_content']] for result in initial_results]
        
        # Get scores
        with PerformanceLogger("Cross-encoder scoring"):
            scores = self.cross_encoder.predict(pairs)
        
        # Add scores to results
        for i, result in enumerate(initial_results):
            result['rerank_score'] = float(scores[i])
            result['final_score'] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(initial_results, 
                         key=lambda x: x['rerank_score'],
                         reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked[:n_results], 1):
            result['rank'] = i
        
        logger.debug(f"✅ Re-ranking complete: {len(reranked[:n_results])} results")
        return reranked[:n_results]
    
    def filtered_search(self, query: str, filters: Dict,
                       n_results: int = 5) -> List[Dict]:
        """
        Search with metadata filters
        
        Args:
            query: Search query
            filters: ChromaDB metadata filters
            n_results: Results to return
        
        Returns:
            Filtered search results
        
        Example filters:
            {"source_type": "pdf"}
            {"source_type": {"$in": ["pdf", "web"]}}
            {"chunk_word_count": {"$gt": 100}}
        """
        ConfigValidator.validate_search_params(n_results)
        
        logger.debug(f"🔄 Filtered search: {filters}")
        
        results = self.db_manager.search(query, n_results=n_results, filters=filters)
        
        logger.debug(f"✅ Filtered search complete: {len(results)} results")
        return results
    
    def _initialize_bm25(self):
        """Initialize BM25 index from cached documents"""
        if not self.db_manager.documents_cache:
            logger.warning("⚠️  No documents cached for BM25")
            return
        
        logger.info("Building BM25 index...")
        corpus = [doc['content'] for doc in self.db_manager.documents_cache]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"✅ BM25 index built ({len(corpus)} documents)")


class UnifiedSearchEngine:
    """
    Unified search engine with intelligent routing
    
    One-stop interface for all search capabilities with automatic strategy selection.
    """
    
    def __init__(self, db_manager: ChromaDBManager, enable_routing: bool = True):
        """
        Initialize unified search engine
        
        Args:
            db_manager: ChromaDB manager instance
            enable_routing: Enable intelligent query routing
        """
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
        Universal search with automatic strategy selection
        
        Args:
            query: Search query
            n_results: Results to return
            strategy: Force specific strategy (None for auto-routing)
            strategy_params: Strategy-specific parameters
            enable_reranking: Enable re-ranking (uses config default if None)
        
        Returns:
            Search results with rich metadata
        
        Strategies:
            - "semantic": Pure vector similarity
            - "hybrid": BM25 + Vector (best for most queries)
            - "mmr": Diverse results
            - "bm25_heavy": Keyword-focused hybrid
            - "filtered": With metadata filters
        """
        # Validate
        query = TextValidator.validate_query(query)
        ConfigValidator.validate_search_params(n_results)
        
        # Route query if no strategy specified
        if strategy is None and self.enable_routing and self.router:
            strategy, auto_params = self.router.route(query)
            strategy_params = strategy_params or auto_params
        else:
            strategy = strategy or "semantic"
            strategy_params = strategy_params or {}
        
        logger.info(f"\n🔍 Search: '{query}'")
        logger.info(f"   Strategy: {strategy}")
        
        # Execute search based on strategy
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
                alpha = strategy_params.get("alpha", 0.3)  # More BM25 weight
                results = self.advanced.hybrid_search(query, n_results, alpha)
                
            elif strategy == "filtered":
                filters = strategy_params.get("filters", {})
                results = self.advanced.filtered_search(query, filters, n_results)
                
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
        """
        Compare different search strategies side-by-side
        
        Args:
            query: Search query
            n_results: Results per strategy
        
        Returns:
            Dict mapping strategy names to results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Comparing Search Strategies: '{query}'")
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
            
            # Print top result
            if results:
                top = results[0]
                logger.info(f"   Top result (score={top.get('final_score', 0):.3f}):")
                logger.info(f"   {top['content'][:100]}...")
        
        return comparison


# Utility functions
def load_documents_from_json(folder_path: str = "data/extracted") -> List[Dict]:
    """
    Load documents from extracted JSON files
    
    Args:
        folder_path: Path to extracted documents folder
    
    Returns:
        List of document dicts
    """
    logger.info(f"\n📂 Loading documents from: {folder_path}")
    
    json_files = [f for f in Path(folder_path).glob("*.json") 
                  if f.name != "_index.json"]
    
    if not json_files:
        logger.warning(f"⚠️  No JSON files found in {folder_path}")
        return []
    
    all_documents = []
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for chunk in data.get('chunks', []):
            doc = {
                'id': chunk['doc_id'],
                'content': chunk['content'],
                'metadata': chunk.get('metadata', {})
            }
            all_documents.append(doc)
    
    logger.info(f"✅ Loaded {len(all_documents)} document chunks from {len(json_files)} files")
    return all_documents


if __name__ == "__main__":
    # Demo
    logger.info("="*70)
    logger.info("Vector Search System - Demo")
    logger.info("="*70)
    
    # Initialize
    chroma = ChromaDBManager()
    collection = chroma.create_collection(reset=False)
    
    # Load documents if collection is empty
    if collection.count() == 0:
        documents = load_documents_from_json()
        if documents:
            chroma.add_documents(documents)
    
    # Initialize search engine
    search_engine = UnifiedSearchEngine(chroma, enable_routing=True)
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How does BERT work?",
        "Compare supervised and unsupervised learning"
    ]
    
    for query in test_queries:
        results = search_engine.search(query, n_results=3)
        
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)
        
        for result in results:
            print(f"\n[{result['rank']}] Score: {result.get('final_score', 0):.3f}")
            print(f"    {result['content']}")
    
    logger.info("\n✅ Demo complete")