import json
import os
from datetime import datetime
from pipeline import RAGPipeline

def build_database():
    """Build the RAG database from sources"""
    
    pipeline = RAGPipeline()
    
    sources = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://www.ibm.com/think/topics/machine-learning",
        "https://en.wikipedia.org/wiki/Embedding_(machine_learning)",
        "https://www.ibm.com/think/topics/embedding",
    ]
    
    print("\n📥 Building database from sources...")
    pipeline.ingest(sources, reset=True, save_extracted=True)
    
    stats = pipeline.get_stats()
    print(f"\n✅ Database ready: {stats['count']} documents\n")

def search_queries():
    """Search using existing database"""
    
    # Initialize (uses existing database)
    pipeline = RAGPipeline()
    
    # Define test queries
    queries = [
        ("What is machine learning?", None, True),
        ("neural network architectures", None, False),
        ("Compare supervised and unsupervised learning", None, True),
        ("How do neural networks learn?", None, False),
        ("machine learning applications", None, False),
    ]
    
    print("\n🔍 Running search queries...")
    print("="*70)
    
    for query, mode, auto_route in queries:
        results = pipeline.query(query, top_k=3, mode=mode, auto_route=auto_route)
        
        actual_mode = mode if mode else 'auto'
        save_results(results, query, actual_mode)
        
        # Print compact results
        print(f"\n'{query}'")
        print(f"Mode: {actual_mode} | Found: {len(results)}")
        
        for i, r in enumerate(results[:2], 1):  # Show top 2
            score = r.get('final_score', r.get('confidence', 0))
            preview = r['content'][:100].replace('\n', ' ')
            print(f"  {i}. [{score:.4f}] {preview}...")
    
    print("\n" + "="*70)
    print(f"✅ Results saved to data/results/\n")


def save_results(results, query, mode, output_dir='data/results'):
    """Save results to JSON"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    query_slug = query.replace(' ', '_').replace('?', '')[:40]
    filename = f"{timestamp}_{mode}_{query_slug}.json"
    
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
        json.dump({
            'query': query,
            'mode': mode,
            'timestamp': timestamp,
            'results': results
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # build_database()
    search_queries()
