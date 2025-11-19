# Advanced RAG System

A state-of-the-art Retrieval-Augmented Generation (RAG) system featuring ensemble chunking, parallel search strategies, adaptive routing, and result fusion using Reciprocal Rank Fusion (RRF).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Data Flow](#data-flow)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Search Strategies](#search-strategies)
- [Ensemble Chunking](#ensemble-chunking)
- [Result Fusion](#result-fusion)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

This RAG system implements cutting-edge techniques for document processing and retrieval, combining multiple chunking strategies with parallel search execution and intelligent result fusion. The system automatically adapts to query types, processes documents in parallel, and provides comprehensive result provenance tracking.

## Key Features

- **Multi-Format Support**: Web pages, PDF, Word, Excel, PowerPoint, CSV, JSON, TXT, HTML
- **Ensemble Chunking**: 4 parallel strategies (sentence-aware, semantic, paragraph, fixed-size) with quality scoring
- **Parallel Search Strategies**: 6 strategies (semantic, BM25, hybrid, MMR, rerank, parallel fusion) executed concurrently
- **Intelligent Routing**: Automatic strategy selection based on query type detection
- **Result Fusion**: Reciprocal Rank Fusion (RRF) with confidence scoring and diversity filtering
- **Adaptive Scoring**: Percentile-based normalization with metadata boosting
- **Parallel Processing**: Concurrent document extraction, chunking, and search execution
- **Result Tracking**: Automatic saving of intermediate and final results with full provenance
- **LLM Integration**: Built-in context formatting for language model consumption
- **Batch Processing**: Efficient handling of multiple queries simultaneously

## Data Flow

### Document Ingestion Pipeline

```
Sources → DocumentExtractor → EnsembleChunker → MultiCollectionManager → ChromaDB Collections
    ↓              ↓              ↓              ↓              ↓
Multi-format  4 parallel     Quality scoring  Separate        Vector
parsing       chunking       & deduplication collections    embeddings
strategies                     by strategy
```

### Query Processing Pipeline

```
Query → QueryRouter → AdvancedSearchStrategies → ResultFusion → UnifiedScorer → Final Results
   ↓         ↓              ↓                        ↓              ↓              ↓
User input  Type detection  Parallel execution      RRF fusion   Adaptive      Ranked with
           (how-to,        (4-16 searches)         Confidence    boosting     provenance
           definition,                           scoring
           comparison,
           etc.)
```

### Result Processing Flow

```
Raw Results → ResultTracker → LLM Context Builder → Formatted Prompt
     ↓              ↓              ↓              ↓
Fusion scores   JSON storage   Token counting   System + context
Confidence     Provenance     Metadata         messages
Provenance     tracking       filtering
```

## Architecture

```
rag_system/
├── pipeline.py                   # Main RAGPipeline orchestrator
│                                 # - Ingestion coordination
│                                 # - Query processing
│                                 # - Batch operations
│                                 # - LLM context building
├── core/                         # Core processing components
│   ├── extraction.py             # DocumentExtractor - Multi-format parsing
│   │                             # - Web scraping, file parsing
│   │                             # - Parallel processing
│   ├── chunking.py               # EnsembleChunker - 4 chunking strategies
│   │                             # - Quality scoring, deduplication
│   ├── embedding.py              # EmbeddingGenerator - SentenceTransformer
│   └── database.py               # MultiCollectionManager - ChromaDB
├── strategies/                   # Search and fusion strategies
│   ├── search_strategies.py      # AdvancedSearchStrategies - 6 search methods
│   ├── query_router.py           # QueryRouter - Intelligent routing
│   ├── fusion.py                 # ResultFusion - RRF implementation
│   └── result_tracker.py         # ResultTracker - JSON result storage
├── scoring/                      # Adaptive scoring system
│   ├── scorer.py                 # UnifiedScorer - Percentile-based scoring
│   └── __init__.py
├── config/                       # Configuration management
│   ├── settings.py               # Comprehensive settings
│   └── __init__.py
├── utils/                        # Utilities
│   └── helpers.py                # Text processing helpers
├── data/                         # Data storage (auto-created)
│   ├── extracted/                # Saved extracted documents
│   ├── chroma_db/                # Vector database
│   ├── search_results/           # Query results (intermediate/final)
│   └── models/                   # Cached models
└── example_usage.py              # Usage examples
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
git clone <repository-url>
cd rag-system
pip install -r requirements.txt
```

### Required Packages

- `sentence-transformers>=2.2.2`: For embedding generation
- `chromadb>=0.4.0`: Vector database
- `numpy>=1.24.0`: Numerical computations
- `pandas>=2.0.0`: Data processing
- `scikit-learn>=1.3.0`: Machine learning utilities
- `rank-bm25>=0.2.2`: Keyword-based search
- `requests>=2.31.0`: HTTP requests
- `beautifulsoup4>=4.12.0`: HTML parsing
- `lxml>=4.9.0`: XML processing
- `PyPDF2>=3.0.0`: PDF text extraction
- `python-docx>=0.8.11`: Word document processing
- `openpyxl>=3.1.0`: Excel file handling
- `python-pptx>=0.6.21`: PowerPoint processing

## Quick Start

### Basic Usage

```python
from pipeline import RAGPipeline

# Initialize the RAG system
pipeline = RAGPipeline()

# Ingest documents with ensemble chunking
sources = [
    "https://en.wikipedia.org/wiki/Machine_learning",
    "path/to/your/document.pdf"
]
pipeline.ingest(sources, reset=True)

# Search with auto-routing and parallel strategies
results = pipeline.query("What is machine learning?", top_k=5)

# Results include fusion scores and strategy provenance
for result in results:
    print(f"Score: {result['final_score']:.3f}")
    print(f"RRF Score: {result['fusion_score']:.6f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Strategies: {result['strategies_used']}")
    print(f"Content: {result['content'][:200]}...")
    print("---")
```

### Run Validation Tests

```bash
# Test ensemble chunking and parallel search
python example_usage.py
```

## Usage Guide

### Document Ingestion

#### Ingest with Ensemble Chunking

```python
pipeline = RAGPipeline()

# Ensemble chunking automatically applies 4 strategies
sources = [
    "https://example.com/article1",
    "documents/research_paper.pdf",
    "data/dataset.xlsx"
]

pipeline.ingest(
    sources=sources,
    reset=True,                    # Clear database
    save_extracted=True,           # Save to JSON
    update_mode='skip'             # 'skip', 'replace', 'merge'
)
```

#### Load from Saved Documents

```python
# Load previously extracted documents
pipeline.load_from_json('data/extracted', reset=False)
```

### Querying

#### Auto-Routing Search

```python
# Automatically selects optimal strategy based on query type
results = pipeline.query("How do neural networks work?")  # Routes to 'rerank'
results = pipeline.query("machine learning algorithms")   # Routes to 'hybrid'
results = pipeline.query("compare CNN and RNN")           # Routes to 'parallel'
```

#### Manual Strategy Selection

```python
# Force specific strategy
results = pipeline.query("query", mode='parallel', auto_route=False)
results = pipeline.query("query", mode='semantic', auto_route=False)
results = pipeline.query("query", mode='rerank', auto_route=False)
```

#### Advanced Query Options

```python
results = pipeline.query(
    query="complex technical question",
    top_k=10,
    mode='parallel',       # Use all strategies with RRF fusion
    auto_route=False,
    min_confidence=0.2,    # Filter low-confidence results
    save_results=True      # Save to data/search_results/
)

# Results include comprehensive metadata
for result in results:
    fusion_score = result['fusion_score']      # RRF score
    confidence = result['confidence']          # Strategy agreement
    strategies = result['strategies_used']     # Which strategies found it
    ranks = result['strategy_ranks']          # Rank in each strategy
    chunking = result['chunking_strategy']    # Which chunking method
```

#### Batch Processing

```python
# Process multiple queries in parallel
queries = [
    "What is machine learning?",
    "How do neural networks work?",
    "Compare supervised and unsupervised learning"
]

results_dict = pipeline.query_batch(
    queries=queries,
    top_k=5,
    mode='parallel',
    max_workers=4
)

for query, results in results_dict.items():
    print(f"Query: {query}")
    print(f"Results: {len(results)}")
```

### LLM Integration

#### Build Context for Language Models

```python
# Get formatted context for LLM consumption
context_dict = pipeline.build_llm_context(
    results=results,
    max_tokens=4000,
    include_metadata=True,
    min_confidence=0.2
)

print(f"Context tokens: ~{context_dict['estimated_tokens']}")
print(f"Chunks used: {context_dict['chunks_used']}")
print(f"Average confidence: {context_dict['avg_confidence']:.3f}")
```

#### Format Complete Prompt

```python
# Create full prompt with system message
prompt = pipeline.format_for_llm(
    query="Explain machine learning",
    context_dict=context_dict,
    system_prompt="You are a helpful AI assistant. Answer based on the provided context."
)

print(prompt)  # Ready for LLM API call
```

### Database Management

```python
# Get statistics
stats = pipeline.get_stats()
print(f"Documents: {stats['total']}")

# Clear database
pipeline.clear_database()
```

## Search Strategies

The system implements 6 search strategies, executed in parallel for optimal results:

### 1. Semantic Search (`semantic`)

Pure vector similarity using SentenceTransformer embeddings. Fast and effective for conceptual queries.

### 2. BM25 Search (`bm25`)

Keyword-based search using BM25 algorithm. Excellent for exact term matching and sparse queries.

### 3. Hybrid Search (`hybrid`)

Combines semantic and BM25 with configurable weights (default: 70% semantic, 30% keyword).

### 4. MMR Search (`mmr`)

Maximal Marginal Relevance for diverse results. Balances relevance vs. diversity using lambda parameter.

### 5. Rerank Search (`rerank`)

Cross-encoder reranking for high accuracy. Uses pre-trained cross-encoder model for query-document pairs.

### 6. Parallel Search (`parallel`) - **DEFAULT**

Executes all 4 strategies (semantic, hybrid, mmr, rerank) in parallel and fuses results using Reciprocal Rank Fusion (RRF).

```python
# Parallel search with RRF fusion (recommended)
results = pipeline.query("complex query", mode='parallel')
```

## Ensemble Chunking

The system uses **EnsembleChunker** to apply multiple chunking strategies and select the best chunks:

### 4 Chunking Strategies

1. **Sentence-Aware**: Respects sentence boundaries with configurable overlap
2. **Semantic**: Chunks based on semantic similarity between sentences
3. **Paragraph**: Uses paragraph boundaries as natural breaks
4. **Fixed-Size**: Fixed word count with configurable overlap

### Ensemble Process

1. **Parallel Execution**: All 4 strategies run simultaneously
2. **Quality Scoring**: Each chunk scored on length, keywords, coherence, information density
3. **Deduplication**: Removes near-duplicate chunks using Jaccard similarity
4. **Best Selection**: Top N chunks selected ensuring document coverage

### Benefits

- **Better Quality**: Combines strengths of multiple approaches
- **Document Coverage**: Ensures important sections are captured
- **Adaptive**: Works across different document types and domains

```python
# Ensemble chunking is used automatically in extraction
extractor = DocumentExtractor()  # Always uses ensemble chunking
```

## Result Fusion

The system uses **Reciprocal Rank Fusion (RRF)** to combine results from multiple search strategies:

### RRF Algorithm

```
RRF Score = Σ(1 / (k + rank_in_strategy_i))
```

- **k = 60**: Standard RRF constant
- **True Top-N**: Global ranking across all strategies
- **Robust**: Less sensitive to individual strategy biases

### Fusion Features

- **Provenance Tracking**: Which strategies found each result
- **Confidence Scoring**: Based on strategy agreement and coverage
- **Diversity**: MMR applied to prevent duplicate results
- **Parallel Execution**: All strategies run concurrently

### Confidence Calculation

```
Confidence = 0.40 × Coverage + 0.30 × Quality + 0.30 × Agreement
```

- **Coverage**: How many strategies found the result
- **Quality**: Average rank position across strategies
- **Agreement**: Consistency of ranking across strategies

## Configuration

Comprehensive configuration in `config/settings.py`:

### Core Settings

```python
# Chunking configuration
CHUNKING = ChunkingConfig(
    target_words=300,              # Target chunk size
    overlap_words=50,              # Overlap between chunks
    min_chunk_words=30,            # Minimum chunk size
    max_chunk_words=500,           # Maximum chunk size
    similarity_threshold=0.3       # Semantic similarity threshold
)

# Search configuration
SEARCH = SearchConfig(
    default_mode='parallel',       # Default search mode
    top_k=5,                       # Default results per query
    hybrid_alpha=0.7,              # Semantic:keyword weight ratio
    mmr_lambda=0.7,                # Relevance:diversity balance
    rerank_model='cross-encoder/ms-marco-MiniLM-L-12-v2'
)

# Embedding configuration
EMBEDDING = EmbeddingConfig(
    model_name='all-MiniLM-L6-v2',
    batch_size=32,
    normalize_embeddings=True
)
```

### Advanced Options

```python
# Extraction settings
EXTRACTION = ExtractionConfig(
    supported_formats=['.pdf', '.docx', '.pptx', '.xlsx', '.csv', '.txt', '.json', '.html'],
    timeout=30,                    # Request timeout
    max_workers=4,                 # Parallel processing workers
    user_agent='Mozilla/5.0...',   # User agent for web scraping
    output_dir='data/extracted'    # Extraction output directory
)

# Database settings
DATABASE = DatabaseConfig(
    persist_directory='data/chroma_db',
    collection_name='rag_documents',
    batch_size=100
)
```

## API Reference

### RAGPipeline

Main pipeline orchestrator:

```python
class RAGPipeline:
    def __init__(self, verbose: bool = True)

    # Ingestion methods
    def ingest(self, sources: List[str], reset: bool = False,
               save_extracted: bool = True, update_mode: str = 'skip')
    def load_from_json(self, folder_path: str, reset: bool = False,
                      update_mode: str = 'skip')

    # Query methods
    def query(self, query_text: str, top_k: int = 5, mode: str = None,
             auto_route: bool = True, min_confidence: float = 0.0,
             save_results: bool = True) -> List[Dict[str, Any]]
    def query_batch(self, queries: List[str], top_k: int = 5, mode: str = None,
                    auto_route: bool = True, max_workers: int = 4,
                    save_results: bool = True) -> Dict[str, List[Dict[str, Any]]]

    # LLM integration
    def build_llm_context(self, results: List[Dict[str, Any]],
                         max_tokens: int = 4000,
                         include_metadata: bool = True,
                         min_confidence: float = 0.0) -> Dict[str, Any]
    def format_for_llm(self, query: str, context_dict: Dict[str, Any],
                      system_prompt: str = None) -> str

    # Database management
    def clear_database(self)
    def get_stats(self) -> Dict[str, Any]
```

### DocumentExtractor

Multi-format document extraction:

```python
class DocumentExtractor:
    def __init__(self)

    # Main methods
    def extract_multiple(self, sources: List[str]) -> Dict[str, List[Dict[str, Any]]]
    def save_documents(self, strategy_documents: Dict[str, List[Dict[str, Any]]])
    def load_from_folder(self, folder_path: str) -> Dict[str, List[Dict[str, Any]]]

    # Format-specific extractors
    def _extract_web(self, url: str) -> Dict[str, List[Dict[str, Any]]]
    def _extract_file(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]
    def _extract_pdf(self, path: Path) -> str
    def _extract_word(self, path: Path) -> str
    def _extract_excel(self, path: Path) -> str
    def _extract_csv(self, path: Path) -> str
    def _extract_pptx(self, path: Path) -> str
    def _extract_txt(self, path: Path) -> str
    def _extract_json(self, path: Path) -> str
```

### EnsembleChunker

Advanced chunking with multiple strategies:

```python
class EnsembleChunker:
    def __init__(self)

    # Main methods
    def chunk_all_strategies(self, text: str) -> Dict[str, List[Chunk]]

    # Individual strategies
    def _sentence_aware_chunk(self, text: str) -> List[Chunk]
    def _semantic_chunk(self, text: str) -> List[Chunk]
    def _paragraph_chunk(self, text: str) -> List[Chunk]
    def _fixed_size_chunk(self, text: str) -> List[Chunk]

    # Quality and processing
    def _create_chunk(self, text: str, index: int) -> Chunk
    def _calculate_document_stats(self, chunks: List[Chunk]) -> Dict
```

### MultiCollectionManager

ChromaDB database manager with separate collections:

```python
class MultiCollectionManager:
    def __init__(self)

    # Initialization
    def initialize(self, reset: bool = False)

    # Document management
    def add_documents_by_strategy(self, strategy_documents: Dict[str, List[Dict]],
                                  update_mode: str = 'skip')

    # Search
    def search(self, query: str, strategy: str, n_results: int = 5) -> List[Dict[str, Any]]

    # Utilities
    def clear_all(self)
    def get_stats(self) -> Dict[str, Any]
```

### AdvancedSearchStrategies

Search strategy implementations:

```python
class AdvancedSearchStrategies:
    def __init__(self, db_manager)

    # Individual strategies
    def semantic_search(self, query: str, strategy: str, n_results: int = 5) -> List[Dict]
    def bm25_search(self, query: str, strategy: str, n_results: int = 5) -> List[Dict]
    def hybrid_search(self, query: str, strategy: str, n_results: int = 5, alpha: float = None) -> List[Dict]
    def mmr_search(self, query: str, strategy: str, n_results: int = 5, lambda_param: float = None) -> List[Dict]
    def rerank_search(self, query: str, strategy: str, n_results: int = 5) -> List[Dict]

    # Parallel execution
    def parallel_search_single_strategy(self, query: str, chunking_strategy: str, n_results: int = 5) -> List[Dict]
    def parallel_search_all(self, query: str, n_results: int = 5) -> Dict[str, List[Dict]]
    def single_strategy_all_chunking(self, query: str, search_strategy: str, n_results: int = 5) -> Dict[str, List[Dict]]
```

### ResultFusion

Reciprocal Rank Fusion implementation:

```python
class ResultFusion:
    def __init__(self, embedding_generator=None)

    # Main fusion method
    def reciprocal_rank_fusion(self, strategy_results: Dict[str, List[Dict]],
                               top_n: int = 5) -> List[Dict]

    # Helper methods
    def _calculate_confidence(self, doc_data: Dict) -> float
    def _apply_diversity(self, results: List[Dict], top_n: int) -> List[Dict]
    def deduplicate_results(self, results: List[Dict], threshold: float = 0.95) -> List[Dict]
```

### UnifiedScorer

Adaptive scoring system:

```python
class UnifiedScorer:
    # Normalization methods
    def normalize_bm25(self, scores: np.ndarray) -> np.ndarray
    def normalize_cross_encoder_scores(self, scores: np.ndarray) -> np.ndarray

    # Scoring methods
    def calculate_metadata_boost(self, metadata: Dict[str, Any],
                                 query: str = None,
                                 result_pool: List[Dict] = None) -> float
    def compute_final_score(self, base_score: float,
                           metadata: Dict[str, Any],
                           query: str = None,
                           result_pool: List[Dict] = None) -> float
```

### QueryRouter

Intelligent query routing:

```python
class QueryRouter:
    @staticmethod
    def detect_query_type(query: str) -> str  # 'how-to', 'definition', 'comparison', etc.

    @staticmethod
    def route(query: str) -> str  # Returns optimal search mode
```

### ResultTracker

Result persistence and tracking:

```python
class ResultTracker:
    def __init__(self, base_dir: str = 'data/search_results')

    # Saving methods
    def save_intermediate_results(self, query: str, chunking_results: Dict[str, List[Dict]],
                                 mode: str, timestamp: str = None)
    def save_final_results(self, query: str, final_results: List[Dict],
                          mode: str, timestamp: str = None)

    # Utilities
    def get_recent_results(self, result_type: str = 'final', limit: int = 10) -> List[str]
```

## Testing

The system includes comprehensive validation tests:

### Run Tests

```bash
# Test ensemble chunking and parallel search
python example_usage.py
```

### Test Results

Expected output includes:

- ✅ All scores ≤ 1.0 and ≥ 0.0
- ✅ Quality scores present and reasonable
- ✅ Multiple strategies executed in parallel mode
- ✅ Keywords extracted and meaningful (≥4 characters)
- ✅ RRF scores decrease monotonically
- ✅ Confidence scores calculated and in [0,1] range
- ✅ No duplicate results in final output
- ✅ Result provenance tracking (strategies_used, strategy_ranks)

### Test Coverage

- **Ensemble Chunking**: Validates all 4 strategies, quality scoring, deduplication
- **Parallel Search**: Tests concurrent strategy execution and result fusion
- **Result Fusion**: Validates RRF implementation, confidence scoring, diversity
- **Query Routing**: Tests automatic strategy selection based on query type
- **LLM Integration**: Validates context building and prompt formatting

## Contributing

### Development Setup

```bash
git clone <repository-url>
cd rag-system
pip install -r requirements.txt
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Add comprehensive docstrings for classes and methods
- Write descriptive commit messages

### Testing

```bash
# Run existing tests
python example_usage.py

# Add new tests for new features
# Ensure all scores are properly normalized
# Validate result provenance and confidence scores
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation
6. Submit pull request

### Performance Notes

- **Parallel Processing**: The system maximizes concurrency at every stage

  - Document extraction: Parallel across sources
  - Chunking: 4 strategies run simultaneously
  - Search: Multiple strategies execute in parallel
  - Batch queries: Concurrent processing

- **Memory Management**: Large document collections are handled efficiently

  - Streaming document processing
  - Batch embedding generation
  - Incremental database updates

- **Scalability**: Designed for production use
  - Configurable batch sizes and worker counts
  - Persistent vector storage with ChromaDB
  - Result caching and deduplication

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This system represents a production-ready RAG implementation with state-of-the-art techniques for document processing and retrieval. The ensemble chunking and parallel search features provide significant improvements over single-strategy approaches, while the intelligent routing and fusion ensure optimal results for diverse query types.
