# Advanced RAG System

A retrieval-augmented generation pipeline for mixed-format documents. The system ingests URLs and local files, chunks each document with four strategies in parallel, indexes each chunking variant in its own ChromaDB collection, and fuses the results into a ranked list with provenance and confidence scores.

## Overview

This repository contains a Python-based RAG pipeline built around `RAGPipeline` in [pipeline.py](pipeline.py). It is designed for local experimentation and internal tooling, not a hosted web service.

### What it does

- Ingests web pages, PDFs, DOCX, PPTX, XLSX, CSV, TXT, HTML, and JSON content.
- Applies four chunking strategies in parallel: sentence-aware, semantic, paragraph, and fixed-size.
- Stores each chunking strategy in a separate ChromaDB collection.
- Runs semantic, BM25, hybrid, MMR, and rerank retrieval strategies.
- Fuses results with Reciprocal Rank Fusion (RRF) and diversity filtering.
- Saves intermediate and final query results to `data/search_results/`.
- Builds an LLM-ready context payload from retrieved chunks.

## Repository layout

- [pipeline.py](pipeline.py) — `RAGPipeline` orchestration for ingest, query, batch query, and LLM context formatting.
- [example_usage.py](example_usage.py) — end-to-end demo script.
- [core/extraction.py](core/extraction.py) — multi-format source extraction.
- [core/chunking.py](core/chunking.py) — ensemble chunking strategies and chunk scoring.
- [core/database.py](core/database.py) — ChromaDB integration and collection management.
- [core/embedding.py](core/embedding.py) — embedding wrapper around `sentence-transformers`.
- [strategies/search_strategies.py](strategies/search_strategies.py) — retrieval primitives and orchestration.
- [strategies/fusion.py](strategies/fusion.py) — RRF and diversity-based fusion.
- [strategies/query_router.py](strategies/query_router.py) — query classification and auto-routing.
- [strategies/result_tracker.py](strategies/result_tracker.py) — JSON persistence for intermediate and final results.
- [scoring/scorer.py](scoring/scorer.py) — score normalization and confidence math.
- [config/settings.py](config/settings.py) — application configuration.

## Installation

### Prerequisites

- Python 3.8+
- `pip`

### Install dependencies

```bash
pip install -r requirements.txt
```

### Runtime notes

- The embedding model `all-MiniLM-L6-v2` is downloaded automatically on first use and cached under `data/models/`.
- The rerank model `cross-encoder/ms-marco-MiniLM-L-12-v2` is loaded lazily the first time rerank is used.
- `pdfplumber` is the PDF parser used by the codebase. The README no longer references `PyPDF2`.

## Quick start

```python
from pipeline import RAGPipeline

pipeline = RAGPipeline(verbose=True)

pipeline.ingest(
    sources=[
        "https://en.wikipedia.org/wiki/Machine_learning",
        "./docs/example.pdf",
    ],
    reset=True,
    save_extracted=True,
    update_mode="skip",
)

results = pipeline.query(
    query_text="What is machine learning?",
    top_k=5,
    mode="parallel",
    auto_route=False,
    save_results=True,
)

for result in results:
    print(f"Final score: {result['final_score']:.3f}")
    print(f"Fusion score: {result['fusion_score']:.6f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Chunking strategy: {result['chunking_strategy']}")
    print(f"Strategies used: {result['strategies_used']}")
    print(result["content"][:200])
    print("---")
```

## Example script

Run the built-in demo script to test ingestion and both parallel and auto-routed queries:

```bash
python example_usage.py
```

## How the pipeline works

### Ingestion

`pipeline.ingest(...)` calls the extractor, chunks the text with all four strategies, and inserts the chunks into ChromaDB. The data is saved in separate collections per chunking strategy, so retrieval can compare different chunk views of the same source.

### Querying

`pipeline.query(...)` accepts:

- `query_text`: the user question.
- `top_k`: number of final results to return.
- `mode`: one of `None`, `parallel`, `semantic`, `bm25`, `hybrid`, `mmr`, or `rerank`.
- `auto_route`: when `True`, `QueryRouter` selects a mode automatically.
- `min_confidence`: optional minimum confidence filter.
- `save_results`: optionally persist intermediate and final JSON outputs.

### Auto routing

When `mode=None` and `auto_route=True`, the router chooses a search mode from the query text:

- `how to`, `how do`, `can`, `does` -> `rerank`
- `what is`, `define`, `explain` -> `semantic`
- `compare`, `vs`, `difference` -> `parallel`
- `list`, `enumerate` -> `mmr`
- short queries (`<= 3` tokens) -> `hybrid`
- otherwise -> `parallel`

### Search strategies

The system currently supports:

- `semantic`: pure embedding similarity
- `bm25`: lexical BM25 search
- `hybrid`: weighted combination of semantic and BM25
- `mmr`: diversity-aware retrieval
- `rerank`: cross-encoder re-ranking
- `parallel`: runs four searches per chunking strategy and fuses the results

In `parallel` mode, the pipeline evaluates 16 search combinations total: 4 chunking strategies x 4 search strategies.

## Result metadata

Each returned result includes:

- `final_score` — normalized fused score used for the final ranking
- `fusion_score` — raw RRF score before normalization
- `confidence` — agreement/coverage/quality estimate
- `strategies_used` — which retrieval strategies found the chunk
- `strategy_ranks` — per-strategy rank information
- `strategy_scores` — per-strategy score information
- `chunking_strategy` — the chunking variant that produced the result
- `content` — chunk text
- `metadata` — source, title, and related metadata

## LLM integration

The pipeline can format retrieved results into a prompt for downstream models:

```python
context = pipeline.build_llm_context(results, max_tokens=4000)
prompt = pipeline.format_for_llm(
    query="Explain machine learning",
    context_dict=context,
    system_prompt="You are a helpful assistant. Answer using the provided context.",
)
```

`build_llm_context(...)` greedily includes chunks until the token budget is reached, using an approximate 4 characters per token estimate.

## Configuration

Key settings are defined in [config/settings.py](config/settings.py):

- `target_words = 300`
- `overlap_words = 50`
- `min_chunk_words = 30`
- `max_chunk_words = 500`
- `similarity_threshold = 0.3`
- `hybrid_alpha = 0.7`
- `mmr_lambda = 0.7`
- `rerank_model = cross-encoder/ms-marco-MiniLM-L-12-v2`
- `persist_directory = data/chroma_db`

## Data and persistence

The pipeline creates and uses the following directories at runtime:

- `data/chroma_db/` — persistent ChromaDB collections
- `data/extracted/` — saved extracted source documents
- `data/search_results/intermediate/` — per-strategy intermediate outputs
- `data/search_results/final/` — fused final results
- `data/models/` — cached model artifacts

## Batch querying

Use `query_batch(...)` to run several queries in parallel:

```python
results_by_query = pipeline.query_batch(
    queries=[
        "What is machine learning?",
        "How do neural networks work?",
        "Compare supervised and unsupervised learning",
    ],
    top_k=5,
    mode="parallel",
    max_workers=4,
    save_results=True,
)
```

## Troubleshooting

- If ingestion returns no documents, confirm that the path or URL is accessible and supported.
- If `parallel` or `rerank` is slow on first use, the cross-encoder is being downloaded and cached.
- If you want to rebuild the index from scratch, call `pipeline.ingest(..., reset=True)` or `pipeline.clear_database()`.
- To reload previously extracted chunks from disk, use `pipeline.load_from_json("data/extracted", reset=False)`.

## Development notes

- This repository is a Python library and CLI-style workflow; it does not expose an HTTP API.
- ChromaDB is used as a local persistent vector store.
- The implementation intentionally keeps `torch` and `transformers` in `requirements.txt` because the embedding and rerank stacks depend on them.

## References

- [README.md](README.md)
- [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)
- [requirements.txt](requirements.txt)
- [example_usage.py](example_usage.py)
