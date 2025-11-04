# 🚀 Production-Ready RAG System

A complete, production-ready Retrieval-Augmented Generation (RAG) system with intelligent search routing, multiple chunking strategies, and advanced vector search techniques.

## ✨ Features

### 🔧 Core Capabilities
- **Multi-Source Extraction**: Web (Scrapy/BeautifulSoup), PDF, Excel, Word, PowerPoint, CSV, JSON, TXT
- **Smart Chunking**: 5 strategies (sentence-aware, semantic, fixed-size, paragraph, recursive)
- **Advanced Search**: Semantic, Hybrid (BM25+Vector), MMR, Re-ranking, Filtered
- **Intelligent Routing**: Automatically selects best search strategy based on query type
- **Rich Metadata**: Comprehensive tracking and filtering capabilities

### 🎯 Search Strategies

| Strategy | Best For | Description |
|----------|----------|-------------|
| **Semantic** | General queries | Pure vector similarity |
| **Hybrid** | Most queries ⭐ | Combines BM25 keywords + vector semantics |
| **MMR** | Exploratory search | Diverse results, reduces redundancy |
| **Re-ranking** | High accuracy | Cross-encoder second-stage ranking |
| **BM25-Heavy** | Exact matches | Keyword-focused for specific terms |

### 🧠 Intelligent Query Routing

The system automatically analyzes queries and routes them:
- **"What is X?"** → Hybrid (factual + semantic)
- **"How/Why/Explain"** → MMR (diverse perspectives)
- **"Compare X and Y"** → Semantic (understanding needed)
- **Short + specific terms** → BM25-Heavy (exact matches)

## 📦 Installation

```bash
# Clone repository
git clone <your-repo>
cd rag_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import chromadb; import sentence_transformers; print('✅ Installation successful')"
```

## 🚀 Quick Start

### 1. Full Pipeline (Extract → Index → Search)

```bash
# Process URLs
python main.py --mode full --sources \
    "https://en.wikipedia.org/wiki/Machine_learning" \
    "https://en.wikipedia.org/wiki/Deep_learning"

# Process local files
python main.py --mode full --sources \
    "documents/paper.pdf" \
    "documents/report.docx"

# Mix URLs and files
python main.py --mode full --sources \
    "https://example.com" \
    "document.pdf" \
    "data.xlsx"
```

### 2. Extract Documents Only

```bash
python main.py --mode extract \
    --sources "https://example.com" "document.pdf" \
    --output "my_extracted_docs"
```

### 3. Search Existing Database

```bash
# Simple search
python main.py --mode search \
    --query "What is machine learning?"

# Force specific strategy
python main.py --mode search \
    --query "BERT architecture" \
    --strategy hybrid

# Compare all strategies
python main.py --mode search \
    --query "neural networks" \
    --compare
```

### 4. Interactive Mode

```bash
python main.py --mode interactive

# Then enter queries interactively:
# 🔍 Query: What is deep learning?
# 🔍 Query: compare: supervised vs unsupervised learning
# 🔍 Query: quit
```

## 📚 Python API Usage

### Complete Pipeline

```python
from main import RAGPipeline

# Initialize
pipeline = RAGPipeline()

# Run full pipeline
search_engine = pipeline.run_full_pipeline(
    sources=[
        "https://example.com",
        "document.pdf"
    ],
    chunk=True,
    use_scrapy=True
)

# Search
results = search_engine.search(
    "What is machine learning?",
    n_results=5
)

# Display results
for result in results:
    print(f"Score: {result['final_score']:.3f}")
    print(f"Content: {result['content']}")
    print(f"Source: {result['metadata']['source']}")
```

### Extract Only

```python
from core.extraction import DocumentExtractor

extractor = DocumentExtractor()

documents = extractor.extract_multiple(
    sources=["https://example.com", "file.pdf"],
    chunk=True,
    parallel=True
)

extractor.save_documents(documents)
```

### Search Only

```python
from core.search import ChromaDBManager, UnifiedSearchEngine, load_documents_from_json

# Load existing data
db_manager = ChromaDBManager()
db_manager.create_collection("my_docs")

documents = load_documents_from_json("data/extracted")
db_manager.add_documents(documents)

# Initialize search
search_engine = UnifiedSearchEngine(db_manager, enable_routing=True)

# Search with auto-routing
results = search_engine.search("machine learning", n_results=5)

# Force specific strategy
results = search_engine.search(
    "machine learning",
    strategy="hybrid",
    strategy_params={"alpha": 0.7}
)

# Compare strategies
comparison = search_engine.compare_strategies("machine learning", n_results=3)
```

### Custom Chunking

```python
from core.chunking import Chunker, ChunkingConfig

config = ChunkingConfig(
    method="sentence_aware",
    target_words=200,
    overlap_words=40
)

chunker = Chunker(config)
chunks = chunker.chunk(your_text)

# Get statistics
stats = chunker.get_chunk_statistics(chunks)
print(f"Average chunk size: {stats['avg_word_count']} words")
```

## ⚙️ Configuration

Edit `config/settings.py` to customize:

### Chunking
```python
CHUNKING = ChunkingConfig(
    method="sentence_aware",  # sentence_aware, semantic, fixed_size, paragraph
    target_words=150,         # Target chunk size
    overlap_words=30,         # Overlap for context
    min_chunk_words=50        # Minimum viable chunk
)
```

### Embeddings
```python
EMBEDDING = EmbeddingConfig(
    model_name="all-mpnet-base-v2",  # Best quality/speed balance
    # Alternatives:
    # "all-MiniLM-L6-v2" - Faster, smaller (384 dim)
    # "all-MiniLM-L12-v2" - Good balance
    dimension=768,
    batch_size=32
)
```

### Search
```python
SEARCH = SearchConfig(
    default_top_k=5,
    hybrid_alpha=0.7,         # 70% vector, 30% BM25
    mmr_lambda=0.7,           # 70% relevance, 30% diversity
    use_reranking=True,       # Enable cross-encoder re-ranking
    enable_smart_routing=True # Auto-select strategy
)
```

## 🎨 Project Structure

```
rag_system/
├── config/
│   ├── __init__.py
│   └── settings.py          # Centralized configuration
├── core/
│   ├── __init__.py
│   ├── chunking.py          # 5 chunking strategies
│   ├── extraction.py        # Multi-source document extraction
│   └── search.py            # Advanced vector search
├── utils/
│   ├── __init__.py
│   ├── logger.py            # Structured logging
│   └── validators.py        # Input validation
├── data/
│   ├── raw/                 # Original documents
│   ├── extracted/           # Processed documents
│   └── chroma_db/           # Vector database
├── logs/                    # System logs
├── main.py                  # Entry point
├── requirements.txt
└── README.md
```

## 🔍 Search Strategy Details

### Semantic Search (Pure Vector)
- **When to use**: General queries, semantic understanding needed
- **Pros**: Good for similar meaning, works across paraphrases
- **Cons**: May miss exact keyword matches

### Hybrid Search (BM25 + Vector)
- **When to use**: Most queries (default) ⭐
- **Pros**: Combines keyword precision + semantic understanding
- **Cons**: Slightly slower than pure methods
- **Parameters**: `alpha` (0.0=pure BM25, 1.0=pure vector)

### MMR (Maximal Marginal Relevance)
- **When to use**: Exploratory search, want diverse results
- **Pros**: Reduces redundancy, diverse perspectives
- **Cons**: May sacrifice some relevance for diversity
- **Parameters**: `lambda_param` (0.0=max diversity, 1.0=max relevance)

### Re-ranking (Cross-Encoder)
- **When to use**: High-accuracy requirements
- **Pros**: Most accurate, joint query-document encoding
- **Cons**: Slower, should be used as second stage
- **Usage**: Automatically applied if `use_reranking=True`

## 📊 Performance Tips

### For Speed
1. Use `fixed_size` chunking (fastest)
2. Use `all-MiniLM-L6-v2` embedding model (384 dim)
3. Disable re-ranking: `use_reranking=False`
4. Reduce `batch_size` if memory limited

### For Quality
1. Use `sentence_aware` chunking (best context preservation)
2. Use `all-mpnet-base-v2` embedding model (768 dim)
3. Enable re-ranking: `use_reranking=True`
4. Use hybrid search with `alpha=0.7`

### For Large Documents
1. Use `paragraph` or `recursive` chunking
2. Increase `target_words` to 200-300
3. Use MMR to get diverse sections

## 🧪 Examples

### Example 1: Research Paper Analysis
```bash
python main.py --mode full \
    --sources "papers/*.pdf" \
    --collection research_papers

python main.py --mode search \
    --query "What are the main findings?" \
    --collection research_papers \
    --strategy mmr
```

### Example 2: Documentation Search
```bash
python main.py --mode full \
    --sources "https://docs.example.com" \
    --collection docs

python main.py --mode interactive \
    --collection docs
```

### Example 3: Multi-Source Knowledge Base
```bash
python main.py --mode full \
    --sources \
        "https://wiki.example.com" \
        "internal_docs/*.pdf" \
        "reports/*.xlsx" \
    --collection knowledge_base
```

## 🐛 Troubleshooting

### ChromaDB SQLite Error
```bash
# If you see SQLite version errors:
pip install pysqlite3-binary
```

### Out of Memory
```python
# Reduce batch size in config/settings.py
EMBEDDING = EmbeddingConfig(
    batch_size=16  # Reduce from 32
)
```

### Slow Embedding Generation
```bash
# Install PyTorch with CUDA for GPU acceleration
# Visit: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Scrapy Errors
```python
# Disable Scrapy, use BeautifulSoup instead
python main.py --mode extract --no-scrapy --sources "https://example.com"
```

## 📈 Evaluation & Metrics

### Compare Search Strategies
```python
from core.search import UnifiedSearchEngine

comparison = search_engine.compare_strategies(
    "What is machine learning?",
    n_results=5
)

# Analyze which strategy performs best for your data
```

### Custom Metrics
```python
# Implement your own evaluation
def evaluate_results(query, results, ground_truth):
    # Calculate MRR, NDCG, Precision@K, etc.
    pass
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Additional chunking strategies
- [ ] More embedding models
- [ ] Query expansion
- [ ] Result caching
- [ ] API endpoint wrapper
- [ ] Streamlit/Gradio UI

## 📄 License

MIT License - feel free to use in your projects!

## 🙏 Acknowledgments

Built with:
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [Scrapy](https://scrapy.org/) - Web scraping
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) - BM25 implementation

---

**Made with ❤️ for the RAG community**

For questions or issues, please open a GitHub issue.