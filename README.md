# Advanced RAG System

A state-of-the-art Retrieval-Augmented Generation (RAG) system featuring ensemble chunking, parallel search strategies, and result fusion using Reciprocal Rank Fusion (RRF).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
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

This RAG system implements cutting-edge techniques for document processing and retrieval, combining multiple chunking strategies and search methods with robust result fusion.

## Key Features

- **Multi-Format Support**: Web pages, PDF, Word, Excel, PowerPoint, CSV, JSON, TXT
- **Ensemble Chunking**: 4 strategies (sentence-aware, semantic, paragraph, fixed-size) with quality selection
- **Parallel Search Strategies**: Semantic, Hybrid, BM25, MMR, Rerank, and Parallel fusion
- **Result Fusion**: Reciprocal Rank Fusion (RRF) for combining multiple strategies
- **Adaptive Scoring**: Quality-based chunk filtering and metadata boosting
- **Intelligent Routing**: Automatic strategy selection based on query type
- **Parallel Processing**: Concurrent document extraction and strategy execution

## Architecture

```
rag_system/
├── core/                          # Core processing components
│   ├── extraction.py             # Multi-format document extraction with ensemble chunking
│   ├── chunking.py               # EnsembleChunker (4 strategies) + legacy DocumentChunker
│   ├── embedding.py              # SentenceTransformer embeddings
│   ├── database.py               # ChromaDB vector database
│   └── document_processor.py     # Text processing utilities
├── strategies/                   # Search and fusion strategies
│   ├── search_strategies.py      # 6 advanced search strategies + parallel execution
│   ├── query_router.py           # Intelligent query routing
│   ├── fusion.py                 # Reciprocal Rank Fusion (RRF)
│   └── __init__.py
├── scoring/                      # Adaptive scoring system
│   ├── scorer.py                 # UnifiedScorer with metadata boosting
│   └── __init__.py
├── config/                       # Configuration management
│   ├── settings.py               # Comprehensive configuration
│   └── __init__.py
├── utils/                        # Utilities
│   └── helpers.py                # Helper functions
├── data/                         # Data storage (auto-created)
│   ├── extracted/                # Saved extracted documents
│   ├── chroma_db/                # Vector database
│   └── models/                   # Cached models
├── pipeline.py                   # Main RAG pipeline orchestrator
├── example_usage.py              # Ensemble chunking test
├── test_rrf.py                   # Result fusion validation
├── requirements.txt              # Dependencies
└── README.md                     # This file
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
# Test ensemble chunking
python example_usage.py

# Test result fusion (RRF)
python test_rrf.py
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
    auto_route=False
)

# Results include fusion metadata
for result in results:
    fusion_score = result['fusion_score']      # RRF score
    confidence = result['confidence']          # Strategy agreement
    strategies = result['strategies_used']     # Which strategies found it
    ranks = result['strategy_ranks']          # Rank in each strategy
```

### Database Management

```python
# Get statistics
stats = pipeline.get_stats()
print(f"Documents: {stats['count']}")

# Clear database
pipeline.clear_database()
```

## Search Strategies

The system implements 6 search strategies, executed in parallel for optimal results:

### 1. Semantic Search (`semantic`)

Pure vector similarity using embeddings. Fast and effective for conceptual queries.

### 2. BM25 Search (`bm25`)

Keyword-based search using BM25 algorithm. Excellent for exact term matching.

### 3. Hybrid Search (`hybrid`)

Combines semantic and BM25 with configurable weights (default: 70% semantic, 30% keyword).

### 4. MMR Search (`mmr`)

Maximal Marginal Relevance for diverse results. Balances relevance vs. diversity.

### 5. Rerank Search (`rerank`)

Cross-encoder reranking for high accuracy. Best for complex understanding.

### 6. Parallel Search (`parallel`) - **DEFAULT**

Executes all 4 strategies (semantic, hybrid, mmr, rerank) in parallel and fuses results using Reciprocal Rank Fusion (RRF).

```python
# Parallel search with RRF fusion
results = pipeline.query("complex query", mode='parallel')
```

## Ensemble Chunking

The system uses **EnsembleChunker** to apply multiple chunking strategies and select the best chunks:

### 4 Chunking Strategies

1. **Sentence-Aware**: Respects sentence boundaries with overlap
2. **Semantic**: Chunks based on semantic similarity between sentences
3. **Paragraph**: Uses paragraph boundaries as natural breaks
4. **Fixed-Size**: Fixed word count with configurable overlap

### Ensemble Process

1. **Parallel Execution**: All 4 strategies run simultaneously
2. **Deduplication**: Removes near-duplicate chunks using Jaccard similarity
3. **Quality Scoring**: Each chunk scored on length, keywords, coherence, information density
4. **Best Selection**: Top N chunks selected ensuring document coverage

### Benefits

- **Better Quality**: Combines strengths of multiple approaches
- **Document Coverage**: Ensures important sections are captured
- **Adaptive**: Works across different document types and domains

```python
# Ensemble chunking is used automatically in extraction
extractor = DocumentExtractor(use_ensemble=True)  # Default: True
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
    method='sentence_aware',       # Primary method for legacy chunker
    target_words=300,              # Target chunk size
    overlap_words=50,              # Overlap between chunks
    similarity_threshold=0.3       # Semantic similarity threshold
)

# Search configuration
SEARCH = SearchConfig(
    default_mode='parallel',       # Default search mode
    hybrid_alpha=0.7,              # Semantic:keyword weight ratio
    mmr_lambda=0.7,                # Relevance:diversity balance
    rerank_model='cross-encoder/ms-marco-MiniLM-L-12-v2'
)
```

### Advanced Options

```python
# Extraction settings
EXTRACTION = ExtractionConfig(
    supported_formats=['.pdf', '.docx', '.pptx', '.xlsx', '.csv', '.txt', '.json', '.html'],
    timeout=30,                    # Request timeout
    max_workers=4,                 # Parallel processing workers
    user_agent='Mozilla/5.0...'    # User agent for web scraping
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
    def __init__(self, verbose=True)
    def ingest(self, sources, reset, save_extracted, update_mode)
    def load_from_json(self, folder_path, reset, update_mode)
    def query(self, query_text, top_k, mode, auto_route)
    def query_batch(self, queries, top_k, mode, auto_route, max_workers)
    def build_llm_context(self, results, max_tokens, include_metadata, min_confidence)
    def format_for_llm(self, query, context_dict, system_prompt)
    def clear_database(self)
    def get_stats(self)
```

### DocumentExtractor

Multi-format document extraction:

```python
class DocumentExtractor:
    def __init__(self, use_ensemble=True)  # Default: ensemble chunking
    def extract_multiple(self, sources, chunk, parallel)
    def save_documents(self, documents)
    def load_from_folder(self, folder_path)
```

### EnsembleChunker

Advanced chunking with multiple strategies:

```python
class EnsembleChunker:
    def chunk_with_ensemble(self, text, max_chunks=20)
    # Internal methods for each strategy
```

### AdvancedSearchStrategies

Search strategy implementations:

```python
class AdvancedSearchStrategies:
    def semantic_search(self, query, n_results)
    def bm25_search(self, query, n_results)
    def hybrid_search(self, query, n_results, alpha)
    def mmr_search(self, query, n_results, lambda_param)
    def rerank_search(self, query, n_results)
    def parallel_search(self, query, n_results)  # RRF fusion
```

### ResultFusion

Reciprocal Rank Fusion implementation:

```python
class ResultFusion:
    def reciprocal_rank_fusion(self, strategy_results, top_n)
    def deduplicate_results(self, results, threshold)
```

### UnifiedScorer

Adaptive scoring system:

```python
class UnifiedScorer:
    def normalize_bm25(self, scores)
    def normalize_cross_encoder_scores(self, scores)
    def compute_final_score(self, base_score, metadata, query, result_pool)
    def calculate_metadata_boost(self, metadata, query, result_pool)
```

## Testing

The system includes comprehensive validation tests:

### Run Tests

```bash
# Test ensemble chunking
python example_usage.py

# Test result fusion (RRF)
python test_rrf.py
```

### Test Coverage

- **Ensemble Chunking**: Validates all 4 strategies, deduplication, quality scoring
- **Result Fusion**: Tests RRF implementation, provenance tracking, confidence scoring
- **Parallel Execution**: Validates concurrent strategy execution
- **Quality Assurance**: Score validation, keyword extraction, metadata integrity

### Test Results

Expected output includes:

- ✅ All scores ≤ 1.0
- ✅ Quality scores present
- ✅ Multiple strategies executed
- ✅ Keywords are real words (≥4 characters)
- ✅ RRF scores decrease monotonically
- ✅ Confidence scores calculated
- ✅ No duplicate results

## Contributing

### Development Setup

```bash
git clone <repository-url>
cd rag-system
pip install -r requirements.txt
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add comprehensive docstrings
- Write descriptive commit messages

### Testing

```bash
# Run specific tests
python example_usage.py
python test_rrf.py

# Add new tests for new features
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation
6. Submit pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This system represents a production-ready RAG implementation with state-of-the-art techniques for document processing and retrieval. The ensemble chunking and result fusion features provide significant improvements over single-strategy approaches.
