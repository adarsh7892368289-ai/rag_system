import os
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class ExtractionConfig:
    supported_formats = ['.pdf', '.docx', '.pptx', '.xlsx', '.csv', '.txt', '.json', '.html']
    timeout = 30
    max_workers = 4
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    output_dir = os.path.join(BASE_DIR, 'data', 'extracted')

@dataclass
class ChunkingConfig:
    method = 'sentence_aware'
    target_words = 300
    overlap_words = 50
    min_chunk_words = 30
    max_chunk_words = 500
    similarity_threshold = 0.3

@dataclass
class EmbeddingConfig:
    model_name = 'all-MiniLM-L6-v2'
    batch_size = 32
    normalize_embeddings = True
    device = 'cpu'
    cache_dir = os.path.join(BASE_DIR, 'models')

@dataclass
class DatabaseConfig:
    persist_directory = os.path.join(BASE_DIR, 'chroma_db')
    collection_name = 'rag_documents'
    batch_size = 100

@dataclass
class SearchConfig:
    default_mode = 'parallel'
    confidence_threshold = 0.3
    top_k = 5
    max_top_k = 50
    hybrid_alpha = 0.7
    mmr_lambda = 0.7
    mmr_candidates_multiplier = 3
    rerank_model = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    rerank_top_k = 20
    bm25_k1 = 1.5
    bm25_b = 0.75
    enable_reranking = True 

EXTRACTION = ExtractionConfig()
CHUNKING = ChunkingConfig()
EMBEDDING = EmbeddingConfig()
DATABASE = DatabaseConfig()
SEARCH = SearchConfig()