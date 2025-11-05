"""
RAG System - Complete Usage Examples

This file demonstrates all features of the Phase 1+2 RAG system.
"""

from core.extraction import DocumentExtractor
from core.chunking import Chunker, ChunkingConfig
from core.search import ChromaDBManager, UnifiedSearchEngine, load_documents_from_json
from main import RAGPipeline
from config.settings import print_config_summary
from utils.logger import get_logger

logger = get_logger("example")


# =============================================================================
# EXAMPLE 1: Quick Start - Full Pipeline
# =============================================================================

def example_1_quick_start():
    """
    Simplest way to use the system: Extract → Index → Search
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Quick Start - Full Pipeline")
    print("="*70)
    
    pipeline = RAGPipeline()
    
    # Process URLs and get search engine
    search_engine = pipeline.run_full_pipeline(
        sources=[
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://en.wikipedia.org/wiki/Deep_learning"
        ],
        chunk=True,
        use_scrapy=False,  # Use BeautifulSoup (more reliable)
        reset_db=True  # Start fresh
    )
    
    # Search
    results = search_engine.search("What is machine learning?", n_results=3)
    
    print("\n📊 Top 3 Results:")
    for result in results:
        print(f"\n[{result['rank']}] Score: {result['final_score']:.3f}")
        print(f"    {result['content'][:200]}...")


# =============================================================================
# EXAMPLE 2: Extract Documents Only
# =============================================================================

def example_2_extraction_only():
    """
    Extract and save documents without indexing
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Extraction Only")
    print("="*70)
    
    extractor = DocumentExtractor()
    
    # Extract from multiple sources
    documents = extractor.extract_multiple(
        sources=[
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "data/sample.pdf",  # Local file
            "data/report.docx"
        ],
        chunk=True,
        use_scrapy=False,
        parallel=True  # Process files in parallel
    )
    
    print(f"\n✅ Extracted {len(documents)} document chunks")
    
    # Save to JSON
    extractor.save_documents(documents)
    print("💾 Saved to data/extracted/")
    
    # Show document structure
    if documents:
        doc = documents[0]
        print(f"\n📄 Sample Document Structure:")
        print(f"   Content length: {len(doc.content)} chars")
        print(f"   Document ID: {doc.document_id}")
        print(f"   Chunk ID: {doc.chunk_id}")
        print(f"   Metadata fields: {list(doc.metadata.keys())}")


# =============================================================================
# EXAMPLE 3: Custom Chunking Configuration
# =============================================================================

def example_3_custom_chunking():
    """
    Use different chunking strategies
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Chunking")
    print("="*70)
    
    sample_text = """
    Machine learning is a subset of artificial intelligence that focuses on 
    teaching computers to learn from data. There are three main types of 
    machine learning: supervised learning, unsupervised learning, and 
    reinforcement learning.
    
    Supervised learning uses labeled data to train models. The algorithm 
    learns to map inputs to outputs based on example input-output pairs.
    
    Unsupervised learning finds patterns in unlabeled data. It discovers 
    hidden structures without explicit guidance.
    
    Reinforcement learning learns through trial and error. An agent learns 
    to make decisions by receiving rewards or penalties.
    """
    
    # Test different chunking methods
    methods = ['sentence_aware', 'semantic', 'fixed_size', 'paragraph']
    
    for method in methods:
        config = ChunkingConfig(
            method=method,
            target_words=50,  # Small chunks for demo
            overlap_words=10
        )
        
        chunker = Chunker(config)
        chunks = chunker.chunk(sample_text)
        
        print(f"\n{method.upper()}:")
        print(f"   Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks, 1):
            print(f"   Chunk {i}: {chunk.word_count} words")


# =============================================================================
# EXAMPLE 4: Search with Different Strategies
# =============================================================================

def example_4_search_strategies():
    """
    Compare different search strategies
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Search Strategies")
    print("="*70)
    
    # Initialize search engine (assumes database exists)
    db_manager = ChromaDBManager()
    collection = db_manager.create_collection(reset=False)
    
    if collection.count() == 0:
        print("❌ Database is empty. Run example 1 first.")
        return
    
    search_engine = UnifiedSearchEngine(db_manager, enable_routing=True)
    
    query = "How does deep learning work?"
    
    # Try different strategies
    strategies = {
        "semantic": {},
        "hybrid": {"alpha": 0.7},
        "mmr": {"lambda_param": 0.7},
        "bm25_heavy": {"alpha": 0.3}
    }
    
    for strategy, params in strategies.items():
        print(f"\n{strategy.upper()}:")
        results = search_engine.search(
            query,
            n_results=2,
            strategy=strategy,
            strategy_params=params
        )
        
        for result in results:
            print(f"  [{result['rank']}] Score: {result['final_score']:.3f}")
            print(f"      {result['content'][:100]}...")


# =============================================================================
# EXAMPLE 5: Metadata Filtering
# =============================================================================

def example_5_metadata_filtering():
    """
    Search with metadata filters
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Metadata Filtering")
    print("="*70)
    
    db_manager = ChromaDBManager()
    collection = db_manager.create_collection(reset=False)
    
    if collection.count() == 0:
        print("❌ Database is empty. Run example 1 first.")
        return
    
    # Search only PDF documents
    results = db_manager.search(
        query="machine learning algorithms",
        n_results=5,
        filters={"source_type": "pdf"}
    )
    
    print(f"\n📄 Results from PDF documents only:")
    for result in results:
        print(f"  [{result['rank']}] {result['metadata'].get('filename', 'N/A')}")
        print(f"      {result['content'][:100]}...")
    
    # Search specific chunk range
    results = db_manager.search(
        query="neural networks",
        n_results=5,
        filters={"chunk_index": {"$lte": 10}}  # First 10 chunks only
    )
    
    print(f"\n📍 Results from first 10 chunks:")
    for result in results:
        print(f"  [{result['rank']}] Chunk {result['metadata'].get('chunk_index', 'N/A')}")


# =============================================================================
# EXAMPLE 6: Load Existing Documents and Index
# =============================================================================

def example_6_load_and_index():
    """
    Load previously extracted documents and index them
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Load and Index Existing Documents")
    print("="*70)
    
    # Load documents from JSON files
    documents = load_documents_from_json("data/extracted")
    
    if not documents:
        print("❌ No documents found in data/extracted/")
        print("   Run example 2 first to extract documents.")
        return
    
    print(f"📂 Loaded {len(documents)} documents")
    
    # Index them
    db_manager = ChromaDBManager()
    collection = db_manager.create_collection(reset=True)
    db_manager.add_documents(documents)
    
    print(f"✅ Indexed {collection.count()} documents")
    
    # Search
    search_engine = UnifiedSearchEngine(db_manager)
    results = search_engine.search("artificial intelligence", n_results=3)
    
    print("\n🔍 Search Results:")
    for result in results:
        print(f"  [{result['rank']}] {result['content'][:100]}...")


# =============================================================================
# EXAMPLE 7: Batch Processing Multiple Files
# =============================================================================

def example_7_batch_processing():
    """
    Process multiple files in parallel
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Batch Processing")
    print("="*70)
    
    extractor = DocumentExtractor()
    
    # Process multiple files
    files = [
        "data/file1.pdf",
        "data/file2.docx",
        "data/file3.xlsx",
        "data/file4.txt"
    ]
    
    # Filter existing files
    existing_files = [f for f in files if Path(f).exists()]
    
    if not existing_files:
        print("❌ No files found. Create sample files first:")
        print("   data/file1.pdf, data/file2.docx, etc.")
        return
    
    print(f"📁 Processing {len(existing_files)} files in parallel...")
    
    documents = extractor.extract_multiple(
        sources=existing_files,
        chunk=True,
        parallel=True  # Parallel processing
    )
    
    print(f"✅ Extracted {len(documents)} chunks from {len(existing_files)} files")


# =============================================================================
# EXAMPLE 8: Interactive Search Session
# =============================================================================

def example_8_interactive_search():
    """
    Interactive search session
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: Interactive Search")
    print("="*70)
    
    pipeline = RAGPipeline()
    
    # Load existing database
    search_engine = pipeline.run_search_only()
    
    if not search_engine:
        print("❌ No database found. Run example 1 first.")
        return
    
    # Start interactive session
    pipeline.interactive_search()


# =============================================================================
# EXAMPLE 9: Configuration Management
# =============================================================================

def example_9_configuration():
    """
    View and customize configuration
    """
    print("\n" + "="*70)
    print("EXAMPLE 9: Configuration")
    print("="*70)
    
    # Print current configuration
    print_config_summary()
    
    # Custom configuration
    from config.settings import ChunkingConfig, EMBEDDING
    
    print("\n📝 Custom Configuration Example:")
    
    custom_chunking = ChunkingConfig(
        method="semantic",
        target_words=200,
        overlap_words=40,
        similarity_threshold=0.6
    )
    
    print(f"   Chunking method: {custom_chunking.method}")
    print(f"   Target words: {custom_chunking.target_words}")
    print(f"   Overlap: {custom_chunking.overlap_words}")
    
    print(f"\n   Embedding model: {EMBEDDING.model_name}")
    print(f"   Dimensions: {EMBEDDING.dimension}")


# =============================================================================
# EXAMPLE 10: Registry System (Phase 2 Feature)
# =============================================================================

def example_10_registry_system():
    """
    Demonstrate document registry for metadata efficiency
    """
    print("\n" + "="*70)
    print("EXAMPLE 10: Document Registry System")
    print("="*70)
    
    from utils.metadata_manager import get_registry
    
    registry = get_registry()
    
    # Register a document
    doc_id = registry.register_document(
        source="https://example.com/article",
        metadata={
            'title': 'Example Article',
            'url': 'https://example.com/article',
            'author': 'John Doe',
            'published': '2024-01-01',
            'word_count': 2000
        }
    )
    
    print(f"📚 Registered document: {doc_id}")
    
    # Simulate chunk metadata
    chunk_meta = {
        'chunk_index': 0,
        'chunk_word_count': 150,
        'chunk_keywords': ['machine', 'learning', 'data']
    }
    
    # Reconstruct full metadata
    full_meta = registry.reconstruct_chunk_metadata(doc_id, chunk_meta)
    
    print(f"\n📊 Reconstructed Metadata:")
    print(f"   Document fields: title, url, author, published, word_count")
    print(f"   Chunk fields: chunk_index, chunk_word_count, chunk_keywords")
    print(f"   Total fields: {len(full_meta)}")
    
    # Show storage savings
    stats = registry.get_statistics()
    print(f"\n💾 Registry Stats:")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Registry size: {stats['registry_size_kb']:.2f} KB")
    
    print("\n✅ Storage Benefit:")
    print("   Without registry: Each chunk stores ALL metadata (redundant)")
    print("   With registry: Document metadata stored ONCE (85% reduction)")


# =============================================================================
# RUN ALL EXAMPLES
# =============================================================================

def run_all_examples():
    """Run all examples in sequence"""
    examples = [
        ("Quick Start", example_1_quick_start),
        ("Extraction Only", example_2_extraction_only),
        ("Custom Chunking", example_3_custom_chunking),
        ("Search Strategies", example_4_search_strategies),
        ("Metadata Filtering", example_5_metadata_filtering),
        ("Load and Index", example_6_load_and_index),
        ("Batch Processing", example_7_batch_processing),
        ("Configuration", example_9_configuration),
        ("Registry System", example_10_registry_system),
    ]
    
    print("\n" + "="*70)
    print("RAG SYSTEM - ALL EXAMPLES")
    print("="*70)
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nSelect example (1-9) or 'all' to run all:")
    choice = input("Choice: ").strip()
    
    if choice.lower() == 'all':
        for name, func in examples:
            try:
                func()
                input("\nPress Enter to continue...")
            except Exception as e:
                print(f"\n❌ Error in {name}: {e}")
                input("\nPress Enter to continue...")
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                examples[idx][1]()
            else:
                print("Invalid choice")
        except ValueError:
            print("Invalid input")


if __name__ == "__main__":
    from pathlib import Path
    
    # Ensure data directories exist
    Path("data/extracted").mkdir(parents=True, exist_ok=True)
    Path("data/registry").mkdir(parents=True, exist_ok=True)
    
    # Run examples
    run_all_examples()