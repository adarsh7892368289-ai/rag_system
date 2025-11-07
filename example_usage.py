from pipeline import RAGPipeline

pipeline = RAGPipeline()

sources = [
    "https://en.wikipedia.org/wiki/Embedding_(machine_learning)",
    "https://www.ibm.com/think/topics/embedding",
    "https://en.wikipedia.org/wiki/BERT_(language_model)",
    "https://en.wikipedia.org/wiki/Word2vec",
    "https://sbert.net/",
    "https://huggingface.co/sentence-transformers",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://www.ibm.com/think/topics/machine-learning",
    "https://aws.amazon.com/what-is/embeddings-in-machine-learning/",
    "https://developers.google.com/machine-learning/crash-course/embeddings/embedding-space",
    "https://www.cloudflare.com/learning/ai/what-are-embeddings/"
]

# pipeline.ingest(sources, reset=True, save_extracted=True)
pipeline.load_from_json('data/extracted', reset=False)

results = pipeline.query(
    query_text="How to change engine oil?",
    top_k=5,
    mode='parallel'
)

for i, result in enumerate(results, 1):
    print(f"\n{i}. Score: {result['final_score']:.4f}")
    print(f"Content: {result['content'][:200]}...")
    print(f"Source: {result['metadata'].get('source', 'N/A')}")

pipeline.get_stats()

results_hybrid = pipeline.query(
    "What is machine learning?",
    top_k=3,
    mode='hybrid'
)

results_accurate = pipeline.query(
    "Compare deep learning and neural networks",
    top_k=3,
    mode='rerank'
)

results_fast = pipeline.query(
    "Define embeddings in machine learning",
    top_k=3,
    mode='mmr'
)