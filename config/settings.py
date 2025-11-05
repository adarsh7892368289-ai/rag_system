"""
Centralized Configuration 

All settings for RAG system with hierarchical metadata support.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


@dataclass
class MetadataConfig:
    """
    Metadata management configuration
    
    Separates document-level vs chunk-level metadata to eliminate redundancy.
    """
    # Registry settings
    registry_dir: str = str(DATA_DIR / "registry")
    enable_registry: bool = True  # Use document registry system
    
    # Document-level fields (stored once per document)
    document_fields: List[str] = field(default_factory=lambda: [
        'title',
        'url',
        'domain',
        'source_type',
        'filename',
        'extracted_at',
        'extraction_method',
        'total_pages',
        'file_size',
        'meta_description',
        'status_code'
    ])
    
    # Chunk-level fields (stored per chunk)
    chunk_fields: List[str] = field(default_factory=lambda: [
        'chunk_index',
        'chunk_char_count',
        'chunk_word_count',
        'chunk_sentence_count',
        'chunk_keywords',
        'avg_sentence_length'
    ])
    
    # Chunking summary fields (stored once per document)
    chunking_fields: List[str] = field(default_factory=lambda: [
        'chunking_method',
        'chunking_target_words',
        'chunking_overlap_words',
        'total_chunks'
    ])
    
    # Reconstruction settings
    reconstruct_on_search: bool = True  # Rebuild full metadata on retrieval


@dataclass
class ChunkingConfig:
    """Chunking strategy configuration"""
    method: str = "sentence_aware"  # sentence_aware, semantic, fixed_size, paragraph, recursive
    target_words: int = 150
    overlap_words: int = 30
    min_chunk_words: int = 50
    max_chunk_words: int = 500
    
    # Semantic chunking
    similarity_threshold: float = 0.5


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = "all-mpnet-base-v2"  # Best quality/speed
    dimension: int = 768
    batch_size: int = 32
    normalize_embeddings: bool = True
    cache_dir: str = str(DATA_DIR / "models")


@dataclass
class DatabaseConfig:
    """ChromaDB configuration"""
    persist_directory: str = str(DATA_DIR / "chroma_db")
    collection_name: str = "rag_documents"
    batch_size: int = 100
    distance_function: str = "l2"  # l2, cosine, ip
    
    def get_collection_metadata(self) -> Dict:
        """Get metadata for collection creation"""
        return {
            "hnsw:space": self.distance_function,
            "description": "RAG documents with hierarchical metadata"
        }


@dataclass
class SearchConfig:
    """Search configuration"""
    default_top_k: int = 5
    max_top_k: int = 50
    
    # Hybrid search
    hybrid_alpha: float = 0.7  # 0.0=pure BM25, 1.0=pure vector
    
    # MMR search
    mmr_lambda: float = 0.7  # 0.0=max diversity, 1.0=max relevance
    mmr_candidates_multiplier: int = 3
    
    # Re-ranking
    use_reranking: bool = False  # Disabled by default (slow)
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 20


@dataclass
class ExtractionConfig:
    """Document extraction configuration"""
    output_dir: str = str(DATA_DIR / "extracted")
    
    # Web scraping
    download_timeout: int = 30
    download_delay: float = 0.5
    concurrent_requests: int = 4
    retry_times: int = 2
    
    # File processing
    max_file_size_mb: int = 100


# Instantiate configs
METADATA = MetadataConfig()
CHUNKING = ChunkingConfig()
EMBEDDING = EmbeddingConfig()
DATABASE = DatabaseConfig()
SEARCH = SearchConfig()
EXTRACTION = ExtractionConfig()


# Query routing rules
QUERY_ROUTING_RULES = {
    'factual_keywords': ['what is', 'what are', 'define', 'definition'],
    'explanation_keywords': ['how', 'why', 'explain', 'describe'],
    'comparison_keywords': ['compare', 'difference', 'versus', 'vs'],
    'listing_keywords': ['list', 'types of', 'examples of', 'kinds of']
}

SEARCH_STRATEGY_MAP = {
    'factual': 'hybrid',
    'explanation': 'mmr',
    'comparison': 'semantic',
    'listing': 'hybrid',
    'general': 'semantic'
}


def print_config_summary():
    """Print configuration summary"""
    print("\n" + "="*70)
    print("RAG SYSTEM CONFIGURATION")
    print("="*70)
    
    print("\n📊 METADATA (Phase 1+2):")
    print(f"   Registry enabled: {METADATA.enable_registry}")
    print(f"   Registry dir: {METADATA.registry_dir}")
    print(f"   Document fields: {len(METADATA.document_fields)}")
    print(f"   Chunk fields: {len(METADATA.chunk_fields)}")
    
    print("\n✂️  CHUNKING:")
    print(f"   Method: {CHUNKING.method}")
    print(f"   Target size: {CHUNKING.target_words} words")
    print(f"   Overlap: {CHUNKING.overlap_words} words")
    
    print("\n🧠 EMBEDDING:")
    print(f"   Model: {EMBEDDING.model_name}")
    print(f"   Dimensions: {EMBEDDING.dimension}")
    print(f"   Batch size: {EMBEDDING.batch_size}")
    
    print("\n💾 DATABASE:")
    print(f"   Directory: {DATABASE.persist_directory}")
    print(f"   Collection: {DATABASE.collection_name}")
    print(f"   Distance: {DATABASE.distance_function}")
    
    print("\n🔍 SEARCH:")
    print(f"   Default results: {SEARCH.default_top_k}")
    print(f"   Hybrid alpha: {SEARCH.hybrid_alpha}")
    print(f"   MMR lambda: {SEARCH.mmr_lambda}")
    print(f"   Re-ranking: {SEARCH.use_reranking}")
    
    print("\n📥 EXTRACTION:")
    print(f"   Output: {EXTRACTION.output_dir}")
    print(f"   Timeout: {EXTRACTION.download_timeout}s")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    print_config_summary()