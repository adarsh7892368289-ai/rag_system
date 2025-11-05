"""
Complete Usage Examples for Production RAG System

This file demonstrates all features of the RAG system:
1. Building collections from sources
2. Searching with different modes
3. Using results for NLP pipelines
4. Analytics and result tracking
5. Python API usage

Run examples:
    python example_usage.py --example 1  # Build collection
    python example_usage.py --example 2  # Basic search
    python example_usage.py --example 3  # Parallel search 
    python example_usage.py --example 4  # NLP integration
    python example_usage.py --example 5  # Advanced usage
    python example_usage.py --all         # Run all examples
"""

import json
import argparse
from pathlib import Path

from core.extraction import DocumentExtractor
from core.search import ChromaDBManager, UnifiedSearchEngine, load_documents_from_json
from utils.logger import get_logger

logger = get_logger("examples")


# ============================================================================
# EXAMPLE 1: Building a Collection from Scratch
# ============================================================================

def example_1_build_collection():
    """
    Example 1: Extract documents and build searchable collection
    
    This example shows how to:
    - Extract from web sources and files
    - Chunk documents automatically
    - Index into ChromaDB
    - Save for later use
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Building Collection from Sources")
    print("="*70)
    
    # Initialize extractor
    extractor = DocumentExtractor()
    
    # Define sources (mix of web and files)
    sources = [
        # Web sources
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        
        # File sources (uncomment if you have these files)
        # "documents/research_paper.pdf",
        # "documents/report.docx",
        # "documents/data.csv",
    ]
    
    # Extract documents with automatic chunking
    print("\n📥 Step 1: Extracting documents...")
    documents = extractor.extract_multiple(
        sources=sources,
        chunk=True,           # Enable smart chunking
        use_scrapy=False,     # Use BeautifulSoup (more reliable)
        parallel=True         # Process files in parallel
    )
    
    print(f"✅ Extracted {len(documents)} chunks")
    
    # Save extracted documents
    print("\n💾 Step 2: Saving extracted documents...")
    extractor.save_documents(documents)
    print("✅ Saved to data/extracted/")
    
    # Initialize database
    print("\n🔵 Step 3: Indexing into ChromaDB...")
    db = ChromaDBManager()
    collection = db.create_collection(
        collection_name="ml_knowledge_base",
        reset=True  # Reset if exists
    )
    
    # Prepare documents for indexing
    docs_for_indexing = [
        {
            'id': doc.doc_id,
            'content': doc.content,
            'metadata': doc.metadata,
            'document_id': doc.document_id
        }
        for doc in documents
    ]
    
    # Add to database
    db.add_documents(docs_for_indexing)
    
    print(f"✅ Indexed {collection.count()} documents")
    print("\n" + "="*70)
    print("✅ COLLECTION BUILT SUCCESSFULLY!")
    print("="*70)
    print(f"Collection name: {collection.name}")
    print(f"Total documents: {collection.count()}")
    print(f"Ready for searching!")
    print("="*70 + "\n")


# ============================================================================
# EXAMPLE 2: Basic Search
# ============================================================================

def example_2_basic_search():
    """
    Example 2: Simple semantic search
    
    Shows how to:
    - Initialize search engine
    - Perform basic searches
    - View results with confidence scores
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Basic Search")
    print("="*70)
    
    # Initialize
    db = ChromaDBManager()
    db.create_collection(collection_name="ml_knowledge_base")
    
    # Check if collection has data
    if db.collection.count() == 0:
        print("\n⚠️  Collection is empty! Run Example 1 first to build collection.")
        return
    
    # Initialize search engine
    search_engine = UnifiedSearchEngine(
        db,
        confidence_threshold=0.3  # Minimum confidence score
    )
    
    # Perform search
    query = "What is machine learning?"
    print(f"\n🔍 Searching: '{query}'")
    
    results, metadata = search_engine.search(
        query=query,
        top_n=5,
        mode='fast',  # Fast semantic search
        save_results=True
    )
    
    # Display results
    print(f"\n📊 Found {len(results)} results:")
    print("="*70)
    
    for result in results:
        print(f"\n[Rank {result.rank}] Confidence: {result.confidence:.3f}")
        print(f"{'─'*70}")
        print(f"Content: {result.content[:200]}...")
        print(f"Scores: {result.scores}")
        
        # Show metadata
        if 'title' in result.metadata:
            print(f"Title: {result.metadata['title']}")
        if 'source' in result.metadata:
            print(f"Source: {result.metadata['source'][:60]}...")
    
    print("\n" + "="*70)
    print(f"✅ Results saved to: {metadata.get('result_file')}")
    print("="*70 + "\n")


# ============================================================================
# EXAMPLE 3: Parallel Multi-Strategy Search (BEST RESULTS)
# ============================================================================

def example_3_parallel_search():
    """
    Example 3: Parallel multi-strategy search for best results
    
    This is the RECOMMENDED approach for production use!
    
    Runs 4 strategies simultaneously:
    1. Semantic search (vector similarity)
    2. Hybrid search (BM25 + vector)
    3. MMR search (diverse results)
    4. Rerank search (cross-encoder)
    
    Then combines all results using Reciprocal Rank Fusion
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Parallel Multi-Strategy Search (BEST RESULTS)")
    print("="*70)
    
    # Initialize
    db = ChromaDBManager()
    db.create_collection(collection_name="ml_knowledge_base")
    
    if db.collection.count() == 0:
        print("\n⚠️  Collection is empty! Run Example 1 first.")
        return
    
    # Initialize search engine
    search_engine = UnifiedSearchEngine(
        db,
        confidence_threshold=0.4  # Higher threshold for quality
    )
    
    # Test queries
    queries = [
        "What is deep learning?",
        "How does neural network training work?",
        "Compare supervised and unsupervised learning"
    ]
    
    for query in queries:
        print(f"\n{'='*70}")
        print(f"🔍 Query: '{query}'")
        print('='*70)
        
        # Parallel search (runs all 4 strategies)
        results, metadata = search_engine.search(
            query=query,
            top_n=5,
            mode='parallel',  # ⭐ KEY: Runs all strategies in parallel
            save_results=True
        )
        
        print(f"\n📊 Strategy: {metadata['strategy']}")
        print(f"⏱️  Execution time: {metadata['execution_time']:.3f}s")
        print(f"📈 Results: {len(results)}")
        
        # Show top 3 results with ALL scores
        for i, result in enumerate(results[:3], 1):
            print(f"\n[Rank {i}] Confidence: {result.confidence:.3f}")
            print(f"{'─'*50}")
            print(f"Content: {result.content[:150]}...")
            
            # Show all strategy scores
            print(f"\n📊 All Strategy Scores:")
            for strategy, score in result.scores.items():
                print(f"   • {strategy:20s}: {score:.3f}")
        
        print("\n" + "="*70)


# ============================================================================
# EXAMPLE 4: Using Results for NLP Pipeline
# ============================================================================

def example_4_nlp_integration():
    """
    Example 4: Export and use results for NLP pipelines
    
    Shows how to:
    - Search and get high-confidence results
    - Export in different formats (context, QA, chunks)
    - Use with LLM/NLP models
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: NLP Pipeline Integration")
    print("="*70)
    
    # Initialize
    db = ChromaDBManager()
    db.create_collection(collection_name="ml_knowledge_base")
    
    if db.collection.count() == 0:
        print("\n⚠️  Collection is empty! Run Example 1 first.")
        return
    
    search_engine = UnifiedSearchEngine(db, confidence_threshold=0.5)
    
    # Search query
    query = "Explain neural networks and how they work"
    print(f"\n🔍 Query: '{query}'")
    
    # Get results
    results, metadata = search_engine.search(
        query=query,
        top_n=10,
        mode='parallel',
        save_results=True
    )
    
    print(f"\n✅ Found {len(results)} high-confidence results")
    
    # Extract result ID for export
    result_file = metadata.get('result_file')
    if result_file:
        result_id = Path(result_file).stem.replace('result_', '')
        
        # Export in different formats
        print("\n📤 Exporting results...")
        
        # 1. Context format (for RAG)
        context_export = search_engine.export_for_nlp(result_id, 'context')
        if context_export:
            print("\n1️⃣  CONTEXT FORMAT (for RAG):")
            print("─"*70)
            print(context_export[:300] + "...")
            
            # Save to file
            context_file = f"data/search_results/result_{result_id}_context.txt"
            with open(context_file, 'w', encoding='utf-8') as f:
                f.write(context_export)
            print(f"✅ Saved to: {context_file}")
        
        # 2. QA format (for question answering)
        qa_export = search_engine.export_for_nlp(result_id, 'qa')
        if qa_export:
            print("\n2️⃣  QA FORMAT (for Question Answering):")
            print("─"*70)
            qa_data = json.loads(qa_export)
            print(f"Question: {qa_data['question']}")
            print(f"Contexts: {len(qa_data['contexts'])} passages")
            print(f"Confidence scores: {qa_data['confidence_scores'][:3]}...")
            
            # Save to file
            qa_file = f"data/search_results/result_{result_id}_qa.json"
            with open(qa_file, 'w', encoding='utf-8') as f:
                f.write(qa_export)
            print(f"✅ Saved to: {qa_file}")
        
        # 3. Chunks format (for processing)
        chunks_export = search_engine.export_for_nlp(result_id, 'chunks')
        if chunks_export:
            chunks_data = json.loads(chunks_export)
            print(f"\n3️⃣  CHUNKS FORMAT (for Processing):")
            print("─"*70)
            print(f"Total chunks: {len(chunks_data)}")
            print(f"First chunk: {chunks_data[0]['text'][:100]}...")
            
            # Save to file
            chunks_file = f"data/search_results/result_{result_id}_chunks.json"
            with open(chunks_file, 'w', encoding='utf-8') as f:
                f.write(chunks_export)
            print(f"✅ Saved to: {chunks_file}")
    
    # Show how to use with LLM
    print("\n" + "="*70)
    print("💡 Using with LLM/NLP Model:")
    print("="*70)
    
    # Get high-confidence contexts
    contexts = [r.content for r in results if r.confidence > 0.6]
    
    print(f"""
# Example LLM prompt:
prompt = f\"\"\"
Based on the following context, answer the question.

Context:
{' '.join(contexts[:3])}

Question: {query}

Answer:
\"\"\"

# Then pass to your LLM (OpenAI, Anthropic, etc.)
# response = llm.generate(prompt)
""")
    
    print("\n" + "="*70)


# ============================================================================
# EXAMPLE 5: Advanced Usage - Analytics and Result Tracking
# ============================================================================

def example_5_advanced_usage():
    """
    Example 5: Advanced features
    
    Shows:
    - Search analytics
    - Result comparison
    - Performance tracking
    - Custom confidence thresholds
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Advanced Usage")
    print("="*70)
    
    # Initialize
    db = ChromaDBManager()
    db.create_collection(collection_name="ml_knowledge_base")
    
    if db.collection.count() == 0:
        print("\n⚠️  Collection is empty! Run Example 1 first.")
        return
    
    # Different confidence thresholds
    thresholds = [0.3, 0.5, 0.7]
    query = "What are convolutional neural networks?"
    
    print(f"\n🔍 Query: '{query}'")
    print("\n📊 Testing different confidence thresholds:")
    
    for threshold in thresholds:
        search_engine = UnifiedSearchEngine(db, confidence_threshold=threshold)
        
        results, metadata = search_engine.search(
            query=query,
            top_n=10,
            mode='parallel',
            save_results=True
        )
        
        avg_conf = sum(r.confidence for r in results) / len(results) if results else 0
        
        print(f"\n  Threshold {threshold}: {len(results)} results, avg confidence: {avg_conf:.3f}")
    
    # Show analytics
    print("\n" + "="*70)
    print("📊 SEARCH ANALYTICS")
    print("="*70)
    
    search_engine = UnifiedSearchEngine(db, confidence_threshold=0.3)
    analytics = search_engine.get_analytics()
    
    print(json.dumps(analytics, indent=2))
    
    # Compare search modes
    print("\n" + "="*70)
    print("⚡ COMPARING SEARCH MODES")
    print("="*70)
    
    modes = ['fast', 'accurate', 'parallel']
    
    for mode in modes:
        results, metadata = search_engine.search(
            query="machine learning algorithms",
            top_n=5,
            mode=mode,
            save_results=False
        )
        
        print(f"\n{mode.upper()} mode:")
        print(f"  Time: {metadata['execution_time']:.3f}s")
        print(f"  Results: {len(results)}")
        if results:
            print(f"  Top confidence: {results[0].confidence:.3f}")
            print(f"  Strategies used: {list(results[0].scores.keys())}")
    
    print("\n" + "="*70)


# ============================================================================
# EXAMPLE 6: Complete Workflow (Build → Search → Use)
# ============================================================================

def example_6_complete_workflow():
    """
    Example 6: Complete end-to-end workflow
    
    This is what you'd typically do in production:
    1. Build collection once
    2. Search multiple times
    3. Use results for your NLP task
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Complete End-to-End Workflow")
    print("="*70)
    
    # Step 1: Build (do this once)
    print("\n" + "="*70)
    print("STEP 1: Build Collection (one-time setup)")
    print("="*70)
    
    extractor = DocumentExtractor()
    sources = [
        "https://en.wikipedia.org/wiki/Machine_learning",
    ]
    
    print("\n📥 Extracting...")
    documents = extractor.extract_multiple(sources, chunk=True, use_scrapy=False)
    extractor.save_documents(documents)
    
    print("\n💾 Indexing...")
    db = ChromaDBManager()
    db.create_collection("production_db", reset=True)
    
    docs_for_indexing = [
        {'id': doc.doc_id, 'content': doc.content, 
         'metadata': doc.metadata, 'document_id': doc.document_id}
        for doc in documents
    ]
    db.add_documents(docs_for_indexing)
    
    print(f"✅ Collection ready with {db.collection.count()} documents")
    
    # Step 2: Search (do this many times)
    print("\n" + "="*70)
    print("STEP 2: Search (use for queries)")
    print("="*70)
    
    search_engine = UnifiedSearchEngine(db, confidence_threshold=0.4)
    
    user_queries = [
        "What is supervised learning?",
        "How do neural networks learn?",
        "What are the types of machine learning?"
    ]
    
    for query in user_queries:
        print(f"\n🔍 '{query}'")
        
        results, metadata = search_engine.search(
            query=query,
            top_n=3,
            mode='parallel',
            save_results=True
        )
        
        print(f"   ✅ {len(results)} results in {metadata['execution_time']:.2f}s")
        if results:
            print(f"   📊 Top confidence: {results[0].confidence:.3f}")
    
    # Step 3: Use results
    print("\n" + "="*70)
    print("STEP 3: Use Results for NLP Task")
    print("="*70)
    
    # Get last search results
    query = user_queries[-1]
    results, metadata = search_engine.search(query, top_n=5, mode='parallel')
    
    # Extract high-confidence contexts
    contexts = [r.content for r in results if r.confidence > 0.5]
    
    print(f"\n✅ Extracted {len(contexts)} high-confidence contexts")
    print("\n💡 Ready to use with your LLM/NLP model:")
    print(f"   - Query: {query}")
    print(f"   - Contexts: {len(contexts)} passages")
    print(f"   - Total words: {sum(len(c.split()) for c in contexts)}")
    
    print("\n" + "="*70)
    print("✅ WORKFLOW COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Pass contexts to your LLM")
    print("2. Generate answer based on contexts")
    print("3. Track and improve results over time")
    print("="*70 + "\n")


# ============================================================================
# MAIN - Run Examples
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Production RAG System - Usage Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help='Run specific example (1-6)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all examples'
    )
    
    args = parser.parse_args()
    
    examples = {
        1: ("Build Collection", example_1_build_collection),
        2: ("Basic Search", example_2_basic_search),
        3: ("Parallel Search (RECOMMENDED)", example_3_parallel_search),
        4: ("NLP Integration", example_4_nlp_integration),
        5: ("Advanced Usage", example_5_advanced_usage),
        6: ("Complete Workflow", example_6_complete_workflow),
    }
    
    if args.all:
        print("\n" + "🚀"*35)
        print("RUNNING ALL EXAMPLES")
        print("🚀"*35)
        
        for num, (name, func) in examples.items():
            try:
                func()
                input(f"\n✅ Example {num} complete. Press Enter to continue...")
            except Exception as e:
                print(f"\n❌ Example {num} failed: {e}")
                continue
    
    elif args.example:
        num = args.example
        name, func = examples[num]
        print(f"\n🚀 Running Example {num}: {name}")
        func()
    
    else:
        print("\n" + "="*70)
        print("Production RAG System - Usage Examples")
        print("="*70)
        print("\nAvailable examples:")
        for num, (name, _) in examples.items():
            print(f"  {num}. {name}")
        print("\nUsage:")
        print("  python example_usage.py --example 1    # Run specific example")
        print("  python example_usage.py --all           # Run all examples")
        print("\nRecommended order:")
        print("  1. Run Example 1 first (build collection)")
        print("  2. Then run Examples 2-6 (search and use)")
        print("  3. Example 3 (Parallel Search) is RECOMMENDED for production!")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()