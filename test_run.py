"""
Sample Test File - RAG Pipeline Usage Examples

This file demonstrates how to use the RAG pipeline with separate methods
for extraction, indexing, and searching.

Run different tests:
    python test_pipeline.py --test basic
    python test_pipeline.py --test extraction
    python test_pipeline.py --test search
    python test_pipeline.py --test comparison
    python test_pipeline.py --test all
"""

import argparse
from pathlib import Path
from typing import Dict, List, Any

from main import RAGPipeline
from core.search import ChromaDBManager, UnifiedSearchEngine
from config.settings import EXTRACTION, DATABASE
from utils.logger import get_logger

logger = get_logger("test")


class RAGTester:
    """Test suite for RAG pipeline"""
    
    def __init__(self):
        """Initialize tester"""
        self.pipeline = RAGPipeline()
        logger.info("🧪 RAG Tester initialized")
    
    def test_basic_flow(self):
        """
        Test 1: Basic end-to-end flow
        Extract → Index → Search
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 1: Basic End-to-End Flow")
        logger.info("="*70)
        
        # Sample sources (replace with your own)
        sources = [
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://en.wikipedia.org/wiki/Artificial_intelligence"
        ]
        
        # Run full pipeline
        self.pipeline.run_full_pipeline(
            sources=sources,
            collection_name="test_collection",
            reset_db=True,  # Fresh start
            chunk=True,
            use_scrapy=False  # Use BeautifulSoup for reliability
        )
        
        # Test some queries
        test_queries = [
            "What is machine learning?",
            "Types of AI",
            "supervised learning"
        ]
        
        for query in test_queries:
            logger.info(f"\n🔍 Testing query: '{query}'")
            
            # Ensure search_engine is available
            if self.pipeline.search_engine:
                results = self.pipeline.search_engine.search(query, n_results=3)
                
                print(f"\n{'='*70}")
                print(f"Query: {query}")
                print(f"Found {len(results)} results")
                if results:
                    print(f"Top result score: {results[0].get('final_score', 0):.3f}")
                    print(f"Content preview: {results[0]['content'][:150]}...")
        
        logger.info("\n✅ Basic flow test complete")
    
    def test_extraction_only(self):
        """
        Test 2: Extraction only (no database)
        Useful for preprocessing and saving documents
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 2: Extraction Only")
        logger.info("="*70)
        
        # Sample sources
        sources = [
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
        ]
        
        # Extract without creating database
        documents = self.pipeline.run_extraction_only(
            sources=sources,
            output_dir="data/test_extracted",
            chunk=True,
            use_scrapy=False
        )
        
        # Analyze extracted documents
        if documents:
            logger.info(f"\n📊 Extraction Statistics:")
            logger.info(f"   Total documents: {len(documents)}")
            logger.info(f"   Total characters: {sum(len(d.content) for d in documents)}")
            logger.info(f"   Avg chunk size: {sum(len(d.content) for d in documents) / len(documents):.0f} chars")
            
            # Show sample document
            sample = documents[0]
            logger.info(f"\n📄 Sample Document:")
            logger.info(f"   ID: {sample.doc_id}")
            logger.info(f"   Content: {sample.content[:200]}...")
            logger.info(f"   Metadata: {sample.metadata}")
        
        logger.info("\n✅ Extraction test complete")
    
    def test_search_with_existing_db(self):
        """
        Test 3: Search with existing database
        Assumes you already have documents indexed
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 3: Search with Existing Database")
        logger.info("="*70)
        
        # Try to search existing collection
        try:
            test_queries = [
                "machine learning algorithms",
                "neural networks",
                "data preprocessing"
            ]
            
            for query in test_queries:
                logger.info(f"\n🔍 Query: '{query}'")
                
                results = self.pipeline.run_search_only(
                    query=query,
                    collection_name="test_collection",
                    n_results=5,
                    strategy=None,  # Auto-routing
                    compare_strategies=False
                )
                
                if results:
                    logger.info(f"   ✓ Found {len(results)} results")
                    if results and isinstance(results, list) and len(results) > 0:
                        top_result = results[0]
                        if isinstance(top_result, dict):
                            logger.info(f"   Top score: {top_result.get('final_score', 0):.3f}")
                else:
                    logger.warning(f"   ⚠️  No results found")
            
            logger.info("\n✅ Search test complete")
            
        except SystemExit:
            logger.warning("⚠️  No existing database found. Run extraction first.")
    
    def test_strategy_comparison(self):
        """
        Test 4: Compare different search strategies
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 4: Strategy Comparison")
        logger.info("="*70)
        
        # Ensure we have data
        try:
            query = "machine learning applications"
            
            logger.info(f"\n🔍 Comparing strategies for: '{query}'")
            
            comparison = self.pipeline.run_search_only(
                query=query,
                collection_name="test_collection",
                n_results=3,
                compare_strategies=True
            )
            
            # Check if comparison is a dict (it should be when compare_strategies=True)
            if isinstance(comparison, dict):
                # Analyze comparison
                print(f"\n{'='*70}")
                print(f"STRATEGY COMPARISON RESULTS")
                print('='*70)
                
                for strategy, results in comparison.items():
                    print(f"\n{strategy.upper()}:")
                    if results and isinstance(results, list):
                        avg_score = sum(r.get('final_score', 0) for r in results) / len(results)
                        print(f"  Results: {len(results)}")
                        print(f"  Avg Score: {avg_score:.3f}")
                        if len(results) > 0:
                            print(f"  Top Result: {results[0]['content'][:100]}...")
                    else:
                        print(f"  No results")
            
            logger.info("\n✅ Strategy comparison test complete")
            
        except SystemExit:
            logger.warning("⚠️  No existing database found. Run extraction first.")
    
    def test_filtered_search(self):
        """
        Test 5: Filtered search with metadata
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 5: Filtered Search")
        logger.info("="*70)
        
        try:
            # Initialize search if not already done
            if not self.pipeline.search_engine:
                self.pipeline.db_manager = ChromaDBManager()
                collection = self.pipeline.db_manager.create_collection("test_collection", reset=False)
                
                if collection.count() == 0:
                    logger.warning("⚠️  No documents in database")
                    return
                
                self.pipeline.search_engine = UnifiedSearchEngine(
                    self.pipeline.db_manager,
                    enable_routing=True
                )
            
            # Test different filters
            test_cases = [
                {
                    "query": "machine learning",
                    "filters": {"source_type": "web"},
                    "description": "Web sources only"
                },
                {
                    "query": "artificial intelligence",
                    "filters": {"chunk_word_count": {"$gt": 100}},
                    "description": "Large chunks (>100 words)"
                }
            ]
            
            for test in test_cases:
                logger.info(f"\n🔍 Testing: {test['description']}")
                logger.info(f"   Query: {test['query']}")
                logger.info(f"   Filters: {test['filters']}")
                
                results = self.pipeline.search_engine.search(
                    query=test['query'],
                    n_results=3,
                    strategy="filtered",
                    strategy_params={"filters": test['filters']}
                )
                
                print(f"\n   Results: {len(results)}")
                if results:
                    print(f"   Top score: {results[0].get('final_score', 0):.3f}")
            
            logger.info("\n✅ Filtered search test complete")
            
        except Exception as e:
            logger.error(f"❌ Filtered search test failed: {e}")
    
    def test_mmr_diversity(self):
        """
        Test 6: MMR search for diverse results
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 6: MMR Diversity Search")
        logger.info("="*70)
        
        try:
            if not self.pipeline.search_engine:
                self.pipeline.db_manager = ChromaDBManager()
                collection = self.pipeline.db_manager.create_collection("test_collection", reset=False)
                
                if collection.count() == 0:
                    logger.warning("⚠️  No documents in database")
                    return
                
                self.pipeline.search_engine = UnifiedSearchEngine(
                    self.pipeline.db_manager,
                    enable_routing=True
                )
            
            query = "artificial intelligence applications"
            
            # Compare semantic vs MMR
            logger.info(f"\n🔍 Query: '{query}'")
            
            # Semantic search (may have redundant results)
            logger.info("\n   Semantic Search:")
            semantic_results = self.pipeline.search_engine.search(
                query=query,
                n_results=5,
                strategy="semantic"
            )
            
            # MMR search (diverse results)
            logger.info("\n   MMR Search (diverse):")
            mmr_results = self.pipeline.search_engine.search(
                query=query,
                n_results=5,
                strategy="mmr",
                strategy_params={"lambda_param": 0.5}  # Balance relevance & diversity
            )
            
            print(f"\n{'='*70}")
            print("DIVERSITY COMPARISON")
            print('='*70)
            print(f"\nSemantic (may have similar results):")
            for i, r in enumerate(semantic_results[:3], 1):
                print(f"  [{i}] {r['content'][:80]}...")
            
            print(f"\nMMR (more diverse):")
            for i, r in enumerate(mmr_results[:3], 1):
                print(f"  [{i}] {r['content'][:80]}...")
            
            logger.info("\n✅ MMR diversity test complete")
            
        except Exception as e:
            logger.error(f"❌ MMR test failed: {e}")
    
    def test_local_files(self):
        """
        Test 7: Extract from local files
        """
        logger.info("\n" + "="*70)
        logger.info("TEST 7: Local File Extraction")
        logger.info("="*70)
        
        # Create sample text file
        test_file = Path("test_sample.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("""
            Machine Learning Fundamentals
            
            Machine learning is a subset of artificial intelligence that enables 
            systems to learn and improve from experience without being explicitly 
            programmed.
            
            Types of Machine Learning:
            1. Supervised Learning - Learning from labeled data
            2. Unsupervised Learning - Finding patterns in unlabeled data
            3. Reinforcement Learning - Learning through trial and error
            
            Applications include image recognition, natural language processing,
            and recommendation systems.
            """)
        
        logger.info(f"Created test file: {test_file}")
        
        # Extract from local file
        documents = self.pipeline.run_extraction_only(
            sources=[str(test_file)],
            output_dir="data/test_local",
            chunk=True,
            use_scrapy=False
        )
        
        if documents:
            logger.info(f"✓ Extracted {len(documents)} chunks from local file")
            logger.info(f"  First chunk: {documents[0].content[:150]}...")
        
        # Cleanup
        test_file.unlink()
        logger.info("\n✅ Local file test complete")


def main():
    """Run tests"""
    parser = argparse.ArgumentParser(description="RAG Pipeline Tests")
    parser.add_argument(
        '--test',
        type=str,
        choices=['basic', 'extraction', 'search', 'comparison', 'filtered', 'mmr', 'local', 'all'],
        default='basic',
        help='Which test to run'
    )
    
    args = parser.parse_args()
    
    tester = RAGTester()
    
    logger.info("\n" + "="*70)
    logger.info("RAG PIPELINE TEST SUITE")
    logger.info("="*70)
    
    try:
        if args.test == 'basic' or args.test == 'all':
            tester.test_basic_flow()
        
        if args.test == 'extraction' or args.test == 'all':
            tester.test_extraction_only()
        
        if args.test == 'search' or args.test == 'all':
            tester.test_search_with_existing_db()
        
        if args.test == 'comparison' or args.test == 'all':
            tester.test_strategy_comparison()
        
        if args.test == 'filtered' or args.test == 'all':
            tester.test_filtered_search()
        
        if args.test == 'mmr' or args.test == 'all':
            tester.test_mmr_diversity()
        
        if args.test == 'local' or args.test == 'all':
            tester.test_local_files()
        
        logger.info("\n" + "="*70)
        logger.info("✅ ALL TESTS COMPLETE")
        logger.info("="*70)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Tests interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}", exc_info=True)


if __name__ == "__main__":
    main()