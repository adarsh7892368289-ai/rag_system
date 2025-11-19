from pipeline import RAGPipeline


def main():
    print("="*80)
    print("Advanced RAG System - Multi-Collection Testing")
    print("="*80)
    
    pipeline = RAGPipeline(verbose=True)
    
    print("\n[1] INGESTION PHASE")
    sources = [
        "https://en.wikipedia.org/wiki/Machine_learning",
    ]
    
    pipeline.ingest(
        sources=sources,
        reset=True,
        save_extracted=True,
        update_mode='skip'
    )
    
    stats = pipeline.get_stats()
    print(f"\n📊 Database Statistics:")
    for strategy, counts in stats.items():
        if strategy != 'total':
            print(f"   • {strategy}: {counts['count']} chunks")
    print(f"   • TOTAL: {stats['total']} chunks")
    
    print("\n" + "="*80)
    print("[2] PARALLEL MODE QUERY (16 searches)")
    print("="*80)
    
    query1 = "What is machine learning?"
    results1 = pipeline.query(
        query_text=query1,
        top_k=5,
        mode='parallel',
        auto_route=False,
        save_results=True
    )
    
    print(f"\n🔍 Query: {query1}")
    print(f"📋 Top {len(results1)} Results:\n")
    
    for i, result in enumerate(results1, 1):
        print(f"[{i}] Score: {result['final_score']:.3f} | Confidence: {result.get('confidence', 0):.3f}")
        print(f"    Chunking: {result.get('chunking_strategy')} | Strategies: {result.get('strategies_used', [])}")
        print(f"    Content: {result['content'][:150]}...")
        print()
    
    print("\n" + "="*80)
    print("[3] AUTO-ROUTE MODE QUERY (4 searches)")
    print("="*80)
    
    query2 = "How do neural networks work?"
    results2 = pipeline.query(
        query_text=query2,
        top_k=5,
        mode=None,
        auto_route=True,
        save_results=True
    )
    
    print(f"\n🔍 Query: {query2}")
    print(f"📋 Top {len(results2)} Results:\n")
    
    for i, result in enumerate(results2, 1):
        print(f"[{i}] Score: {result['final_score']:.3f} | Confidence: {result.get('confidence', 0):.3f}")
        print(f"    Chunking: {result.get('chunking_strategy')} | Strategies: {result.get('strategies_used', [])}")
        print(f"    Content: {result['content'][:150]}...")
        print()
    
    print("\n" + "="*80)
    print("[4] RESULT ANALYSIS")
    print("="*80)
    
    print(f"\n📊 Parallel Mode Results Analysis:")
    print(f"   • Total results: {len(results1)}")
    print(f"   • Score range: {results1[-1]['final_score']:.3f} - {results1[0]['final_score']:.3f}")
    print(f"   • Avg confidence: {sum(r.get('confidence', 0) for r in results1) / len(results1):.3f}")
    
    print(f"\n📊 Auto-Route Mode Results Analysis:")
    print(f"   • Total results: {len(results2)}")
    print(f"   • Score range: {results2[-1]['final_score']:.3f} - {results2[0]['final_score']:.3f}")
    print(f"   • Avg confidence: {sum(r.get('confidence', 0) for r in results2) / len(results2):.3f}")
    
    print("\n✅ Results saved to data/search_results/")
    print("   • Intermediate: 4 chunking group results per query")
    print("   • Final: Fused results with full metadata")
    
    print("="*80)
    print("✅ Testing Complete!")
    print("="*80)


if __name__ == "__main__":
    main()