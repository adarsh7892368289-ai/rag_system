from pipeline import RAGPipeline
import sys
from io import StringIO


def suppress_output(func):
    """Decorator to suppress print statements during execution"""
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            result = func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
        return result
    return wrapper


def main():
    print("\n" + "="*70)
    print("Advanced RAG System - Professional Demo")
    print("="*70)
    
    pipeline = RAGPipeline(verbose=False)
    
    # 1. Multi-Format Document Ingestion
    print("\n[1] Multi-Format Ingestion")
    print("-" * 70)
    
    sources = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        # "documents/report.pdf",
        # "data/spreadsheet.xlsx",
        # "slides/presentation.pptx",
        # "data/dataset.csv",
    ]
    
    print("Ingesting documents...")
    pipeline.ingest(sources, reset=True, max_chunks_per_doc=50)
    stats = pipeline.get_stats()
    print(f"✓ Ingested {stats['count']} chunks from {len(sources)} sources\n")
    
    # 2. Basic Search
    print("[2] Basic Search")
    print("-" * 70)
    
    results = pipeline.query("What is machine learning?", top_k=3)
    
    for i, r in enumerate(results, 1):
        print(f"Result {i}: Score {r['final_score']:.3f}")
        print(f"   {r['content'][:120]}...\n")
    
    # 3. RRF Fusion with Confidence
    print("[3] Parallel Search with RRF Fusion")
    print("-" * 70)
    
    results = pipeline.query("neural networks deep learning", mode='parallel', top_k=3)
    
    for i, r in enumerate(results, 1):
        conf = r.get('confidence', 0)
        strats = r.get('strategies_used', [])
        print(f"Result {i}: Score {r['final_score']:.3f} | Confidence {conf:.3f} | {len(strats)} strategies")
    print()
    
    # 4. LLM Context Building
    print("[4] LLM Integration")
    print("-" * 70)
    
    query = "Explain supervised learning"
    results = pipeline.query(query, top_k=5, min_confidence=0.6)
    context = pipeline.build_llm_context(results, max_tokens=3000)
    
    print(f"Query: {query}")
    print(f"✓ Context: {context['chunks_used']} chunks, {context['estimated_tokens']} tokens")
    print(f"✓ Avg confidence: {context['avg_confidence']:.3f}")
    print(f"✓ Sources: {len(context['sources'])}\n")
    
    prompt = pipeline.format_for_llm(query, context)
    print(f"LLM Prompt ready ({len(prompt)} chars)\n")
    
    # 5. Batch Processing
    print("[5] Batch Query Processing")
    print("-" * 70)
    
    queries = [
        "What is supervised learning?",
        "Explain neural networks",
        "Types of machine learning algorithms"
    ]
    
    results_dict = pipeline.query_batch(queries, top_k=3, max_workers=3)
    
    for q, res in results_dict.items():
        avg = sum(r['final_score'] for r in res) / len(res) if res else 0
        print(f"✓ '{q[:40]}...' → {len(res)} results (avg {avg:.3f})")
    print()
    
    # 6. Strategy Comparison
    print("[6] Search Strategy Comparison")
    print("-" * 70)
    
    query = "artificial intelligence applications"
    modes = ['semantic', 'hybrid', 'parallel']
    
    for mode in modes:
        results = pipeline.query(query, mode=mode, top_k=5, auto_route=False)
        avg = sum(r['final_score'] for r in results) / len(results) if results else 0
        print(f"{mode.capitalize():12} → {len(results)} results (avg {avg:.3f})")
    
    print("\n" + "="*70)
    print("✅ Demo Complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()