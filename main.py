"""
RAG System Orchestrator - Main Entry Point

Complete pipeline: Extract → Chunk → Embed → Search

Usage:
    # Full pipeline
    python main.py --mode full --sources "url1" "url2" "file.pdf"
    
    # Extract only
    python main.py --mode extract --sources "url" --output data/extracted
    
    # Search only (use existing database)
    python main.py --mode search --query "What is machine learning?"
    
    # Interactive mode
    python main.py --mode interactive
"""

import argparse
import sys
from typing import List, Optional
from pathlib import Path

from core.extraction import DocumentExtractor
from core.search import ChromaDBManager, UnifiedSearchEngine, load_documents_from_json
from config.settings import get_config_summary, EXTRACTION, DATABASE
from utils.logger import RAGLogger, get_logger, log_section
from utils.validators import ValidationError

# Initialize logging
RAGLogger.setup(log_level="INFO", log_to_file=True)
logger = get_logger("main")


class RAGPipeline:
    """
    Complete RAG pipeline orchestrator
    
    Manages the full workflow from document extraction to search.
    """
    
    def __init__(self):
        """Initialize RAG pipeline"""
        self.extractor = None
        self.db_manager = None
        self.search_engine = None
        
        logger.info("🚀 RAG Pipeline initialized")
    
    def run_full_pipeline(self,
                         sources: List[str],
                         collection_name: str = "documents",
                         reset_db: bool = False,
                         chunk: bool = True,
                         use_scrapy: bool = True,
                         save_extracted: bool = True) -> UnifiedSearchEngine:
        """
        Run complete pipeline: Extract → Embed → Index
        
        Args:
            sources: List of URLs or file paths
            collection_name: ChromaDB collection name
            reset_db: Reset existing database
            chunk: Enable chunking
            use_scrapy: Use Scrapy for web scraping
            save_extracted: Save extracted documents to JSON
        
        Returns:
            Configured search engine ready for queries
        """
        log_section(logger, "FULL RAG PIPELINE")
        
        # Step 1: Extract documents
        logger.info("\n" + "="*70)
        logger.info("STEP 1: Document Extraction")
        logger.info("="*70)
        
        self.extractor = DocumentExtractor()
        documents = self.extractor.extract_multiple(
            sources=sources,
            chunk=chunk,
            use_scrapy=use_scrapy,
            parallel=True
        )
        
        if not documents:
            logger.error("❌ No documents extracted. Exiting.")
            sys.exit(1)
        
        # Save extracted documents
        if save_extracted:
            self.extractor.save_documents(documents)
        
        # Step 2: Initialize database and add documents
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Embedding & Indexing")
        logger.info("="*70)
        
        self.db_manager = ChromaDBManager()
        self.db_manager.create_collection(collection_name, reset=reset_db)
        
        # Prepare documents for database
        db_docs = [
            {
                'id': doc.doc_id,
                'content': doc.content,
                'metadata': doc.metadata
            }
            for doc in documents
        ]
        
        self.db_manager.add_documents(db_docs)
        
        # Step 3: Initialize search engine
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Search Engine Initialization")
        logger.info("="*70)
        
        self.search_engine = UnifiedSearchEngine(
            self.db_manager,
            enable_routing=True
        )
        
        logger.info("\n✅ Pipeline complete! Ready for queries.")
        
        return self.search_engine
    
    def run_extraction_only(self,
                           sources: List[str],
                           output_dir: Optional[str] = None,
                           chunk: bool = True,
                           use_scrapy: bool = True) -> List:
        """
        Run extraction only (no database creation)
        
        Args:
            sources: List of URLs or file paths
            output_dir: Output directory for extracted documents
            chunk: Enable chunking
            use_scrapy: Use Scrapy for web scraping
        
        Returns:
            List of extracted documents
        """
        log_section(logger, "EXTRACTION ONLY")
        
        if output_dir:
            EXTRACTION.output_dir = output_dir
        
        self.extractor = DocumentExtractor()
        documents = self.extractor.extract_multiple(
            sources=sources,
            chunk=chunk,
            use_scrapy=use_scrapy,
            parallel=True
        )
        
        if documents:
            self.extractor.save_documents(documents)
            logger.info(f"\n✅ Extraction complete: {len(documents)} documents")
        else:
            logger.warning("⚠️  No documents extracted")
        
        return documents
    
    def run_search_only(self,
                       query: str,
                       collection_name: str = "documents",
                       n_results: int = 5,
                       strategy: Optional[str] = None,
                       compare_strategies: bool = False):
        """
        Run search only (use existing database)
        
        Args:
            query: Search query
            collection_name: ChromaDB collection name
            n_results: Number of results
            strategy: Force specific strategy
            compare_strategies: Compare all strategies
        
        Returns:
            Search results
        """
        log_section(logger, "SEARCH ONLY")
        
        # Initialize database
        self.db_manager = ChromaDBManager()
        collection = self.db_manager.create_collection(collection_name, reset=False)
        
        if collection.count() == 0:
            logger.warning("⚠️  Database is empty. Loading from extracted files...")
            
            # Try to load from extracted files
            logger.info("Attempting to load from extracted files...")
            documents = load_documents_from_json(EXTRACTION.output_dir)
            
            if documents:
                self.db_manager.add_documents(documents)
            else:
                logger.error("❌ No documents found. Run extraction first.")
                sys.exit(1)
        
        # Initialize search
        self.search_engine = UnifiedSearchEngine(
            self.db_manager,
            enable_routing=(strategy is None)
        )
        
        # Search
        if compare_strategies:
            comparison = self.search_engine.compare_strategies(query, n_results)
            return comparison
        else:
            results = self.search_engine.search(
                query,
                n_results=n_results,
                strategy=strategy
            )
            
            # Display results
            self._display_results(query, results)
            
            return results
    
    def run_interactive(self, collection_name: str = "documents"):
        """
        Run interactive search mode
        
        Args:
            collection_name: ChromaDB collection name
        """
        log_section(logger, "INTERACTIVE MODE")
        
        # Initialize
        self.db_manager = ChromaDBManager()
        collection = self.db_manager.create_collection(collection_name, reset=False)
        
        if collection.count() == 0:
            logger.warning("⚠️  Database is empty. Loading from extracted files...")
            documents = load_documents_from_json(EXTRACTION.output_dir)
            
            if documents:
                self.db_manager.add_documents(documents)
            else:
                logger.error("❌ No documents found. Run extraction first.")
                return
        
        self.search_engine = UnifiedSearchEngine(
            self.db_manager,
            enable_routing=True
        )
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Interactive Search Mode")
        logger.info(f"Database: {collection.count()} documents")
        logger.info(f"Type 'quit' or 'exit' to stop")
        logger.info(f"Type 'compare: <query>' to compare strategies")
        logger.info(f"{'='*70}\n")
        
        # Interactive loop
        while True:
            try:
                query = input("\n🔍 Query: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    logger.info("👋 Goodbye!")
                    break
                
                # Check for compare mode
                if query.lower().startswith('compare:'):
                    actual_query = query[8:].strip()
                    comparison = self.search_engine.compare_strategies(actual_query, n_results=3)
                    
                    # Display comparison
                    print(f"\n{'='*70}")
                    print(f"Strategy Comparison")
                    print('='*70)
                    
                    for strategy, results in comparison.items():
                        print(f"\n{strategy.upper()}:")
                        if results:
                            top = results[0]
                            print(f"  Score: {top.get('final_score', 0):.3f}")
                            print(f"  {top['content'][:150]}...")
                
                else:
                    # Normal search
                    results = self.search_engine.search(query, n_results=5)
                    self._display_results(query, results)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")
    
    def _display_results(self, query: str, results: List):
        """Display search results in a readable format"""
        print(f"\n{'='*70}")
        print(f"Results for: '{query}'")
        print(f"{'='*70}")
        
        if not results:
            print("\n❌ No results found")
            return
        
        for result in results:
            rank = result.get('rank', 0)
            score = result.get('final_score', 0)
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            
            print(f"\n[{rank}] Score: {score:.3f}")
            print(f"    {content}")
            print(f"    Source: {source[:60]}...")
            
            # Show score breakdown if available
            if 'hybrid_score' in result:
                print(f"    Scores: vector={result['vector_score']:.3f}, "
                     f"bm25={result['bm25_score']:.3f}, "
                     f"hybrid={result['hybrid_score']:.3f}")
        
        print(f"\n{'='*70}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RAG System - Extract, Index, and Search Documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with URLs
  python main.py --mode full --sources "https://example.com" "https://example2.com"
  
  # Full pipeline with local files
  python main.py --mode full --sources "document.pdf" "data.xlsx"
  
  # Extract only
  python main.py --mode extract --sources "https://example.com"
  
  # Search existing database
  python main.py --mode search --query "What is machine learning?"
  
  # Interactive mode
  python main.py --mode interactive
  
  # Compare strategies
  python main.py --mode search --query "machine learning" --compare
        """
    )
    
    # Mode
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['full', 'extract', 'search', 'interactive'],
        help='Pipeline mode'
    )
    
    # Sources (for extract/full)
    parser.add_argument(
        '--sources',
        nargs='+',
        help='URLs or file paths to process'
    )
    
    # Query (for search)
    parser.add_argument(
        '--query',
        type=str,
        help='Search query'
    )
    
    # Optional parameters
    parser.add_argument(
        '--collection',
        type=str,
        default='documents',
        help='ChromaDB collection name (default: documents)'
    )
    
    parser.add_argument(
        '--n-results',
        type=int,
        default=5,
        help='Number of search results (default: 5)'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['semantic', 'hybrid', 'mmr', 'bm25_heavy'],
        help='Force specific search strategy'
    )
    
    parser.add_argument(
        '--no-chunk',
        action='store_true',
        help='Disable chunking'
    )
    
    parser.add_argument(
        '--no-scrapy',
        action='store_true',
        help='Disable Scrapy (use BeautifulSoup)'
    )
    
    parser.add_argument(
        '--reset-db',
        action='store_true',
        help='Reset database before adding documents'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all search strategies'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for extracted documents'
    )
    
    parser.add_argument(
        '--config',
        action='store_true',
        help='Show configuration and exit'
    )
    
    args = parser.parse_args()
    
    # Show config if requested
    if args.config:
        print(get_config_summary())
        sys.exit(0)
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    try:
        # Route based on mode
        if args.mode == 'full':
            if not args.sources:
                logger.error("❌ --sources required for full mode")
                sys.exit(1)
            
            pipeline.run_full_pipeline(
                sources=args.sources,
                collection_name=args.collection,
                reset_db=args.reset_db,
                chunk=not args.no_chunk,
                use_scrapy=not args.no_scrapy
            )
            
            # Enter interactive mode after setup
            logger.info("\n🎯 Pipeline complete. Entering interactive mode...")
            pipeline.run_interactive(args.collection)
        
        elif args.mode == 'extract':
            if not args.sources:
                logger.error("❌ --sources required for extract mode")
                sys.exit(1)
            
            pipeline.run_extraction_only(
                sources=args.sources,
                output_dir=args.output,
                chunk=not args.no_chunk,
                use_scrapy=not args.no_scrapy
            )
        
        elif args.mode == 'search':
            if not args.query:
                logger.error("❌ --query required for search mode")
                sys.exit(1)
            
            pipeline.run_search_only(
                query=args.query,
                collection_name=args.collection,
                n_results=args.n_results,
                strategy=args.strategy,
                compare_strategies=args.compare
            )
        
        elif args.mode == 'interactive':
            pipeline.run_interactive(args.collection)
    
    except ValidationError as e:
        logger.error(f"❌ Validation error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":    
    print(get_config_summary())
        
    main()