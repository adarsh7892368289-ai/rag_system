# RAG System

A modular Retrieval-Augmented Generation (RAG) system with advanced search strategies, multi-format document processing, and intelligent query routing.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Search Strategies](#search-strategies)
- [Contributing](#contributing)
- [License](#license)

## Overview

This RAG system provides document ingestion, processing, and intelligent retrieval with support for multiple formats, 8 search strategies, and automatic query routing.

## Project Structure

```
rag_system/
├── core/                          # Core components
│   ├── extraction.py             # Document extraction
│   ├── chunking.py               # Text chunking strategies
│   ├── embedding.py              # Embedding generation
│   ├── database.py               # ChromaDB management
│   └── document_processor.py     # Text processing
├── strategies/                   # Search strategies
│   ├── query_processor.py        # Query processing
│   ├── search_engine.py          # Unified search engine
│   └── search_strategies.py      # Individual strategies
├── config/                       # Configuration
│   ├── settings.py               # System settings
│   └── __init__.py
├── utils/                        # Utilities
│   └── helpers.py                # Helper functions
├── data/                         # Data storage (auto-created)
├── pipeline.py                   # Main pipeline
├── example_usage.py              # Usage examples
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## Key Features

- **Multi-Format Support**: PDF, Word, Excel, PowerPoint, CSV, JSON, TXT, HTML, Web pages
- **8 Search Strategies**: Semantic, Hybrid, BM25, MMR, Rerank, Parallel, Fast, Accurate
- **Auto-Routing**: Query type detection and optimal strategy selection
- **Duplicate Handling**: Skip, replace, or merge document updates
- **Parallel Processing**: Concurrent document extraction and processing
- **Configurable Chunking**: Sentence-aware, semantic, paragraph, fixed-size methods
- **Score Normalization**: Consistent 0-1 scoring across all strategies
- **Search Technique Tracking**: Results include the search technique used for transparency

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-system
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Required Packages

- `sentence-transformers>=2.2.2`: For embedding generation
- `chromadb>=0.4.0`: Vector database for document storage
- `numpy>=1.24.0`: Numerical computations
- `pandas>=2.0.0`: Data processing
- `scikit-learn>=1.3.0`: Machine learning utilities
- `rank-bm25>=0.2.2`: Keyword-based search
- `requests>=2.31.0`: HTTP requests for web scraping
- `beautifulsoup4>=4.12.0`: HTML parsing
- `lxml>=4.9.0`: XML processing
- `pdfplumber>=0.10.0`: PDF text extraction
- `python-docx>=0.8.11`: Word document processing
- `python-pptx>=0.6.21`: PowerPoint processing
- `openpyxl>=3.1.0`: Excel file handling

## Quick Start

### Basic Usage

```python
from pipeline import RAGPipeline

# Initialize the RAG system
pipeline = RAGPipeline()

# Define sources (URLs or file paths)
sources = [
    "https://en.wikipedia.org/wiki/Machine_learning",
    "path/to/your/document.pdf"
]

# Ingest documents
pipeline.ingest(sources, reset=True)

# Search with auto-routing
results = pipeline.query("What is machine learning?", top_k=5)

# Print results
for result in results:
    print(f"Score: {result['final_score']:.3f}")
    print(f"Technique: {result['search_technique']}")
    print(f"Content: {result['content'][:200]}...")
    print("---")
```

### Command Line Usage

```bash
# Run example usage
python example_usage.py
```

## Usage Guide

### Document Ingestion

#### Ingest from URLs and Files

```python
# Initialize pipeline
pipeline = RAGPipeline()

# Define sources
sources = [
    "https://example.com/article1",
    "https://example.com/article2",
    "documents/research_paper.pdf",
    "data/dataset.xlsx"
]

# Ingest with options
pipeline.ingest(
    sources=sources,
    reset=True,                    # Clear database before ingestion
    save_extracted=True,           # Save extracted documents to JSON
    update_mode='skip'             # 'skip', 'replace', or 'merge'
)
```

#### Load from Saved JSON Files

```python
# Load previously extracted documents
pipeline.load_from_json(
    folder_path='data/extracted',
    reset=False,
    update_mode='merge'
)
```

### Querying

#### Basic Search

```python
# Simple search with auto-routing
results = pipeline.query("How do neural networks work?")
```

#### Advanced Search Options

```python
# Manual mode selection
results = pipeline.query(
    query="machine learning algorithms",
    top_k=10,
    mode='hybrid',        # 'semantic', 'hybrid', 'bm25', 'mmr', 'rerank', etc.
    auto_route=False      # Disable auto-routing
)

# Search with specific mode
results = pipeline.query(
    query="compare deep learning frameworks",
    mode='parallel'       # Multi-strategy fusion
)
```

#### Different Query Types

```python
# How-to questions (routes to 'rerank')
results = pipeline.query("How to implement word embeddings?")

# Definition questions (routes to 'semantic')
results = pipeline.query("What is supervised learning?")

# Comparison questions (routes to 'parallel')
results = pipeline.query("Compare CNN and RNN architectures")

# Keyword queries (routes to 'hybrid')
results = pipeline.query("neural network backpropagation")
```

### Database Management

```python
# Get system statistics
stats = pipeline.get_stats()
print(f"Documents: {stats['count']}")

# Clear database
pipeline.clear_database()

# Get all document IDs
ids = pipeline.db_manager.get_all_ids()
```

## Configuration

The system uses a modular configuration system located in `config/settings.py`.

### Extraction Configuration

```python
EXTRACTION = ExtractionConfig(
    supported_formats=['.pdf', '.docx', '.pptx', '.xlsx', '.csv', '.txt', '.json', '.html'],
    timeout=30,                    # Request timeout in seconds
    max_workers=4,                 # Parallel processing workers
    user_agent='Mozilla/5.0...',   # User agent for web requests
    output_dir='data/extracted'    # Directory for saved documents
)
```

### Chunking Configuration

```python
CHUNKING = ChunkingConfig(
    method='sentence_aware',       # 'sentence_aware', 'semantic', 'paragraph', 'fixed_size'
    target_words=300,              # Target chunk size
    overlap_words=50,              # Overlap between chunks
    min_chunk_words=30,            # Minimum chunk size
    max_chunk_words=500,           # Maximum chunk size
    similarity_threshold=0.3       # Semantic similarity threshold
)
```

### Embedding Configuration

```python
EMBEDDING = EmbeddingConfig(
    model_name='all-MiniLM-L6-v2',  # SentenceTransformer model
    batch_size=32,                  # Batch size for encoding
    normalize_embeddings=True,      # Normalize embeddings
    device='cpu',                   # 'cpu' or 'cuda'
    cache_dir='data/models'         # Model cache directory
)
```

### Search Configuration

```python
SEARCH = SearchConfig(
    default_mode='parallel',        # Default search mode
    confidence_thresholds={         # Score thresholds by mode
        'parallel': 0.15,
        'fast': 0.25,
        'accurate': 0.30,
        'semantic': 0.25,
        'hybrid': 0.20,
        'bm25': 0.10,
        'mmr': 0.25,
        'rerank': 0.40
    },
    hybrid_alpha=0.7,               # Hybrid search weight (semantic:keyword)
    mmr_lambda=0.7,                 # MMR diversity weight
    rerank_model='cross-encoder/ms-marco-MiniLM-L-6-v-2',
    enable_reranking=True
)
```

## API Reference

### RAGPipeline

Main pipeline class for document processing and search.

#### Methods

- `__init__()`: Initialize the RAG pipeline
- `ingest(sources, reset, save_extracted, update_mode)`: Ingest documents from sources
- `load_from_json(folder_path, reset, update_mode)`: Load documents from JSON files
- `query(query_text, top_k, mode, auto_route)`: Search for documents
- `clear_database()`: Clear all documents from database
- `get_stats()`: Get system statistics

### DocumentExtractor

Handles document extraction from various sources.

#### Methods

- `extract_multiple(sources, chunk, parallel)`: Extract from multiple sources
- `save_documents(documents)`: Save extracted documents to JSON

### DocumentChunker

Intelligent text chunking with multiple strategies.

#### Methods

- `chunk(text, method)`: Chunk text using specified method

### EmbeddingGenerator

Embedding generation using SentenceTransformers.

#### Methods

- `encode(text)`: Generate embedding for single text
- `encode_batch(texts)`: Generate embeddings for multiple texts

### ChromaDBManager

Vector database management with ChromaDB.

#### Methods

- `initialize(reset)`: Initialize database connection
- `add_documents(documents, update_mode)`: Add documents with duplicate handling
- `search(query, n_results)`: Search for similar documents

### AdvancedSearchStrategies

Implements various search strategies.

#### Methods

- `semantic_search(query, n_results)`: Pure vector similarity search
- `bm25_search(query, n_results)`: Keyword-based search
- `hybrid_search(query, n_results, alpha)`: Combined semantic + keyword search
- `mmr_search(query, n_results, lambda_param)`: Maximal Marginal Relevance
- `rerank_search(query, n_results)`: Cross-encoder reranking

## Search Strategies

### 1. Semantic Search (`semantic`)
Pure vector similarity search using embeddings. Fast and good for conceptual queries.

```python
results = pipeline.query("machine learning basics", mode='semantic')
```

### 2. BM25 Search (`bm25`)
Keyword-based search using BM25 algorithm. Good for exact term matching.

```python
results = pipeline.query("neural network architecture", mode='bm25')
```

### 3. Hybrid Search (`hybrid`)
Combines semantic and BM25 search with configurable weights.

```python
results = pipeline.query("deep learning applications", mode='hybrid')
```

### 4. MMR Search (`mmr`)
Maximal Marginal Relevance for diverse results. Balances relevance and diversity.

```python
results = pipeline.query("AI techniques", mode='mmr')
```

### 5. Rerank Search (`rerank`)
Uses cross-encoder for high-accuracy reranking of initial candidates.

```python
results = pipeline.query("how transformers work", mode='rerank')
```

### 6. Parallel Search (`parallel`)
Executes multiple strategies in parallel and fuses results using RRF.

```python
results = pipeline.query("complex query", mode='parallel')
```

### 7. Fast Search (`fast`)
Optimized semantic search for speed.

```python
results = pipeline.query("quick lookup", mode='fast')
```

### 8. Accurate Search (`accurate`)
High-accuracy search using hybrid + reranking.

```python
results = pipeline.query("detailed explanation needed", mode='accurate')
```

### Auto-Routing

The system automatically routes queries based on type:

- **How-to questions** → `rerank` (complex understanding)
- **Definition questions** → `semantic` (conceptual similarity)
- **Comparison questions** → `parallel` (multiple perspectives)
- **List questions** → `mmr` (diversity important)
- **Keyword queries** → `hybrid` (balance keyword + semantic)
- **General questions** → `parallel` (best overall performance)

## Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

1. Fork the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements.txt
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Add docstrings to all classes and methods
- Write descriptive commit messages

### Testing

```bash
# Run tests
python -m pytest

# Run with coverage
python -m pytest --cov=rag_system
```

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure all tests pass
4. Update documentation if needed
5. Submit a pull request with a clear description

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This system is designed for research and development purposes. For production use, consider additional security measures, error handling, and performance optimizations.
