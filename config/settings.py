"""
Centralized Configuration for RAG System

All hyperparameters and settings in one place for easy tuning.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any


# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "cache"

# Create directories
for directory in [DATA_DIR / "raw", DATA_DIR / "extracted", DATA_DIR / "chroma_db", MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ChunkingConfig:
    """Chunking configuration"""
    method: str = "sentence_aware"  # sentence_aware, semantic, fixed_size, paragraph
    target_words: int = 150  # Optimal for most LLMs (100-200 words)
    overlap_words: int = 30  # 20% overlap recommended
    min_chunk_words: int = 50  # Minimum viable chunk
    max_chunk_words: int = 300  # Maximum before forced split
    
    # Semantic chunking specific
    similarity_threshold: float = 0.6  # Topic change detection
    
    # Paragraph chunking specific
    max_paragraphs: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'target_words': self.target_words,
            'overlap_words': self.overlap_words,
            'min_chunk_words': self.min_chunk_words
        }


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    # Model options (in order of quality/speed trade-off):
    # - all-MiniLM-L6-v2: 384 dim, fastest, good quality
    # - all-mpnet-base-v2: 768 dim, best quality/speed balance 
    # - all-MiniLM-L12-v2: 384 dim, good balance
    # - paraphrase-multilingual-MiniLM-L12-v2: 384 dim, multilingual
    
    model_name: str = "all-mpnet-base-v2"
    dimension: int = 768  # Auto-detected from model
    batch_size: int = 32  # For batch encoding
    normalize_embeddings: bool = True  # Better for cosine similarity
    show_progress: bool = True
    cache_dir: str = str(MODELS_DIR)


@dataclass
class SearchConfig:
    """Search configuration"""
    default_top_k: int = 5  # Default results to return
    max_top_k: int = 20  # Maximum allowed
    
    # Hybrid search (BM25 + Vector)
    hybrid_alpha: float = 0.7  # 0.7 = 70% vector, 30% BM25
    hybrid_top_k: int = 10  # Retrieve more candidates for hybrid
    
    # MMR (Maximal Marginal Relevance)
    mmr_lambda: float = 0.7  # 0.7 = 70% relevance, 30% diversity
    mmr_candidates_multiplier: int = 3  # Retrieve 3x candidates
    
    # Re-ranking
    use_reranking: bool = True
    rerank_top_k: int = 10  # Only rerank top-10 (expensive operation)
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Search routing (intelligent strategy selection)
    enable_smart_routing: bool = True
    
    # Filtered search
    enable_metadata_filters: bool = True


@dataclass
class ExtractionConfig:
    """Document extraction configuration"""
    output_dir: str = str(DATA_DIR / "extracted")
    
    # Web scraping
    use_scrapy: bool = True
    follow_links: bool = False
    max_pages_per_seed: int = 10
    download_timeout: int = 30
    concurrent_requests: int = 2
    download_delay: float = 1.0  # Polite crawling
    retry_times: int = 2
    
    # File processing
    batch_size: int = 100  # For bulk operations
    
    # PDF extraction
    pdf_extract_images: bool = False  # Image text extraction (slower)
    
    # Excel/CSV
    excel_max_rows: int = 10000  # Limit for large files


@dataclass
class DatabaseConfig:
    """ChromaDB configuration"""
    persist_directory: str = str(DATA_DIR / "chroma_db")
    collection_name: str = "documents"
    distance_metric: str = "cosine"  # cosine, l2, ip
    
    # HNSW index parameters (for faster search)
    hnsw_space: str = "cosine"
    hnsw_construction_ef: int = 100  # Higher = better recall, slower build
    hnsw_search_ef: int = 100  # Higher = better recall, slower search
    hnsw_m: int = 16  # Number of bi-directional links
    
    # Batch operations
    batch_size: int = 100
    
    def get_collection_metadata(self) -> Dict[str, Any]:
        """Get metadata for collection creation"""
        return {
            "hnsw:space": self.hnsw_space,
            "hnsw:construction_ef": self.hnsw_construction_ef,
            "hnsw:search_ef": self.hnsw_search_ef,
            "hnsw:M": self.hnsw_m
        }


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    log_date_format: str = "%Y-%m-%d %H:%M:%S"
    log_to_file: bool = True
    log_file: str = str(BASE_DIR / "logs" / "rag_system.log")
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


# Global configuration instances
CHUNKING = ChunkingConfig()
EMBEDDING = EmbeddingConfig()
SEARCH = SearchConfig()
EXTRACTION = ExtractionConfig()
DATABASE = DatabaseConfig()
LOGGING = LoggingConfig()


# Query routing rules
QUERY_ROUTING_RULES = {
    "factual_keywords": ["what", "define", "definition", "meaning"],
    "explanation_keywords": ["how", "why", "explain", "describe"],
    "comparison_keywords": ["difference", "compare", "versus", "vs"],
    "listing_keywords": ["list", "types", "kinds", "examples"],
}


# Search strategy mapping by query type
SEARCH_STRATEGY_MAP = {
    "factual": "hybrid",  # Need exact terms + semantic
    "explanation": "mmr",  # Want diverse perspectives
    "comparison": "semantic",  # Semantic understanding needed
    "listing": "hybrid",  # Mix of keywords + meaning
    "specific": "bm25_heavy",  # Exact match important
    "general": "semantic",  # Pure semantic understanding
}


def get_config_summary() -> str:
    """Get human-readable configuration summary"""
    return f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                  RAG SYSTEM CONFIGURATION                    ║
    ╠══════════════════════════════════════════════════════════════╣
    ║ 📊 Chunking:                                                 ║
    ║    Method: {CHUNKING.method:<45} ║
    ║    Target: {CHUNKING.target_words} words (overlap: {CHUNKING.overlap_words})                        ║
    ║                                                              ║
    ║ 🤖 Embeddings:                                               ║
    ║    Model: {EMBEDDING.model_name:<46} ║
    ║    Dimensions: {EMBEDDING.dimension}                                        ║
    ║                                                              ║
    ║ 🔍 Search:                                                   ║
    ║    Default Top-K: {SEARCH.default_top_k}                                           ║
    ║    Hybrid Alpha: {SEARCH.hybrid_alpha} (70% vector, 30% BM25)            ║
    ║    MMR Lambda: {SEARCH.mmr_lambda} (70% relevance, 30% diversity)      ║
    ║    Re-ranking: {'Enabled' if SEARCH.use_reranking else 'Disabled':<43} ║
    ║                                                              ║
    ║ 💾 Database:                                                 ║
    ║    Location: {DATABASE.persist_directory[-35:]:<42} ║
    ║    Collection: {DATABASE.collection_name:<44} ║
    ╚══════════════════════════════════════════════════════════════╝
    """


if __name__ == "__main__":
    print(get_config_summary())