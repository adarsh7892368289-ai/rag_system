"""
Production RAG System - Complete Pipeline

Usage:
    # Build collection from sources
    python main.py build --sources "url1" "url2" "file.pdf"
    
    # Search with parallel strategies
    python main.py search "your question" --top-n 5
    
    # Interactive mode
    python main.py interactive
    
    # View analytics
    python main.py analytics
"""

import argparse
import sys
import json
from typing import List, Optional
from pathlib import Path

from core.extraction import DocumentExtractor
from core.search import ChromaDBManager, UnifiedSearchEngine, load_documents_from_json
from config.settings import DATABASE, print_config_summary
from utils.logger import get_logger, PerformanceLogger

logger = get_logger("main")


class ProductionRAG:
    """
    Complete production RAG system with parallel search
    """
    
    def __init__(self):
        """Initialize RAG system"""
        self.extractor = DocumentExtractor()
        self.db_manager = ChromaDBManager()
        self.search_engine = None
        
        logger.info("🚀 Production RAG initialized")
    
    def build_collection(self,
                        sources: List[str],
                        collection_name: Optional[str] = None,
                        chunk: bool = True,
                        use_scrapy: bool = True,
                        reset: bool = False) -> bool:
        """
        Build searchable collection from sources
        
        Args:
            sources: URLs or file paths
            collection_name: Collection name
            chunk: Enable chunking
            use_scrapy: Use Scrapy for web
            reset: Reset existing collection
        
        Returns:
            Success status
        """
        logger.info("\n" + "="*70)
        logger.info("📦 BUILDING COLLECTION")
        logger.info("="*70)
        
        try:
            # Step 1: Extract documents
            logger.info("\n📥 STEP 1: Extracting documents...")
            with PerformanceLogger("Document extraction"):
                documents = self.extractor.extract_multiple(
                    sources=sources,
                    chunk=chunk,
                    use_scrapy=use_scrapy,
                    parallel=True
                )
            
            if not documents:
                logger.error("❌ No documents extracted")
                return False
            
            # Save extracted documents
            self.extractor.save_documents(documents)
            
            # Step 2: Create collection
            logger.info("\n💾 STEP 2: Creating collection...")
            collection = self.db_manager.create_collection(
                collection_name=collection_name,
                reset=reset
            )
            
            # Step 3: Index documents
            logger.info("\n🔄 STEP 3: Indexing documents...")
            docs_for_indexing = [
                {
                    'id': doc.doc_id,
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'document_id': doc.document_id
                }
                for doc in documents
            ]
            
            with PerformanceLogger("Document indexing"):
                self.db_manager.add_documents(docs_for_indexing)
            
            logger.info("\n" + "="*70)
            logger.info("✅ COLLECTION BUILD COMPLETE")
            logger.info("="*70)
            logger.info(f"   Sources processed: {len(sources)}")
            logger.info(f"   Documents indexed: {len(documents)}")
            logger.info(f"   Collection: {collection.name}")
            logger.info("="*70 + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Collection build failed: {e}")
            return False
    
    def initialize_search(self,
                         collection_name: Optional[str] = None,
                         confidence_threshold: float = 0.3) -> bool:
        """
        Initialize search engine
        
        Args:
            collection_name: Collection to search
            confidence_threshold: Minimum confidence score
        
        Returns:
            Success status
        """
        try:
            # Create/load collection
            collection = self.db_manager.create_collection(
                collection_name=collection_name,
                reset=False
            )
            
            if collection.count() == 0:
                logger.warning("⚠️  Collection is empty!")
                logger.info("   Run 'build' command first or load existing data")
                
                # Try loading from extracted files
                logger.info("\n🔄 Attempting to load from extracted files...")
                docs = load_documents_from_json()
                if docs:
                    self.db_manager.add_documents(docs)
                    logger.info(f"✅ Loaded {len(docs)} documents")
                else:
                    logger.error("❌ No documents found")
                    return False
            
            # Initialize search engine with parallel mode
            self.search_engine = UnifiedSearchEngine(
                self.db_manager,
                confidence_threshold=confidence_threshold
            )
            
            logger.info("✅ Search engine ready (parallel mode enabled)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Search initialization failed: {e}")
            return False
    
    def search(self,
               query: str,
               top_n: int = 5,
               mode: str = 'parallel',
               display: bool = True,
               export_format: Optional[str] = None) -> Optional[str]:
        """
        Search with parallel multi-strategy execution
        
        Args:
            query: Search query
            top_n: Number of results
            mode: 'parallel' (default), 'fast', 'accurate'
            display: Display results
            export_format: Export format ('context', 'qa', 'chunks')
        
        Returns:
            Result file path or None
        """
        if not self.search_engine:
            if not self.initialize_search():
                return None
        
        try:
            # Execute search with parallel strategies
            results, metadata = self.search_engine.search(
                query=query,
                top_n=top_n,
                mode=mode,
                save_results=True
            )
            
            # Display results
            if display:
                self._display_results(query, results, metadata)
            
            # Export for NLP
            result_file = metadata.get('result_file')
            if export_format and result_file:
                result_id = Path(result_file).stem.replace('result_', '')
                exported = self.search_engine.export_for_nlp(result_id, export_format)
                
                if exported:
                    export_file = result_file.replace('.json', f'_{export_format}.txt')
                    with open(export_file, 'w', encoding='utf-8') as f:
                        f.write(exported)
                    logger.info(f"📤 Exported to: {export_file}")
            
            return result_file
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return None
    
    def interactive_mode(self):
        """Interactive search mode"""
        if not self.search_engine:
            if not self.initialize_search():
                return
        
        logger.info("\n" + "="*70)
        logger.info("🔍 INTERACTIVE SEARCH MODE (Parallel Strategies)")
        logger.info("="*70)
        logger.info("Commands:")
        logger.info("  • Enter query to search")
        logger.info("  • 'mode:[parallel|fast|accurate]' - Change mode")
        logger.info("  • 'top:N' - Set number of results")
        logger.info("  • 'analytics' - View search analytics")
        logger.info("  • 'quit' or 'exit' - Exit")
        logger.info("="*70 + "\n")
        
        current_mode = 'parallel'
        current_top_n = 5
        
        while True:
            try:
                user_input = input("\n🔍 Query: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    logger.info("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'analytics':
                    analytics = self.search_engine.get_analytics()
                    print(f"\n{'='*70}")
                    print("📊 SEARCH ANALYTICS")
                    print('='*70)
                    print(json.dumps(analytics, indent=2))
                    continue
                
                elif user_input.lower().startswith('mode:'):
                    new_mode = user_input.split(':', 1)[1].strip()
                    if new_mode in ['parallel', 'fast', 'accurate']:
                        current_mode = new_mode
                        logger.info(f"✅ Mode changed to: {current_mode}")
                    else:
                        logger.warning(f"❌ Invalid mode: {new_mode}")
                    continue
                
                elif user_input.lower().startswith('top:'):
                    try:
                        new_top_n = int(user_input.split(':', 1)[1].strip())
                        if 1 <= new_top_n <= 50:
                            current_top_n = new_top_n
                            logger.info(f"✅ Top N changed to: {current_top_n}")
                        else:
                            logger.warning("❌ Top N must be between 1 and 50")
                    except ValueError:
                        logger.warning("❌ Invalid number")
                    continue
                
                # Execute search
                self.search(
                    query=user_input,
                    top_n=current_top_n,
                    mode=current_mode,
                    display=True
                )
                
            except KeyboardInterrupt:
                logger.info("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")
    
    def show_analytics(self):
        """Display search analytics"""
        if not self.search_engine:
            if not self.initialize_search():
                return
        
        analytics = self.search_engine.get_analytics()
        
        print(f"\n{'='*70}")
        print("📊 SEARCH ANALYTICS")
        print('='*70)
        print(json.dumps(analytics, indent=2))
        print('='*70 + "\n")
    
    def _display_results(self, query: str, results, metadata):
        """Display search results with all scores"""
        print(f"\n{'='*70}")
        print(f"🔍 SEARCH RESULTS: '{query}'")
        print('='*70)
        print(f"Strategy: {metadata['strategy']} | Mode: {metadata['mode']}")
        print(f"Results: {len(results)} | Time: {metadata['execution_time']:.3f}s")
        print('='*70)
        
        if not results:
            print("\n❌ No results found")
            return
        
        for result in results:
            print(f"\n[Rank {result.rank}] Confidence: {result.confidence:.3f}")
            print(f"{'─'*70}")
            
            # Show content (truncated)
            content = result.content
            if len(content) > 300:
                content = content[:300] + "..."
            print(content)
            
            # Show all strategy scores
            if result.scores:
                print(f"\n📊 Strategy Scores:")
                for strategy, score in result.scores.items():
                    print(f"   • {strategy}: {score:.3f}")
            
            # Show metadata
            meta = result.metadata
            meta_info = []
            if 'source' in meta:
                source = meta['source']
                if len(source) > 60:
                    source = source[:60] + "..."
                meta_info.append(f"Source: {source}")
            if 'source_type' in meta:
                meta_info.append(f"Type: {meta['source_type']}")
            if 'chunk_index' in meta:
                meta_info.append(f"Chunk: {meta['chunk_index']}")
            
            if meta_info:
                print(f"\n📋 {' | '.join(meta_info)}")
        
        print(f"\n{'='*70}")
        if metadata.get('result_file'):
            print(f"💾 Results saved: {metadata['result_file']}")
        print('='*70 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Production RAG System with Parallel Multi-Strategy Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build collection from sources
    python main.py build --sources "https://example.com" "doc.pdf"
    
    # Search with parallel strategies (default)
    python main.py search "What is machine learning?" --top-n 5
    
    # Fast search (semantic only)
    python main.py search "deep learning" --mode fast
    
    # Accurate search (hybrid + rerank)
    python main.py search "transformers" --mode accurate --top-n 10
    
    # Interactive mode
    python main.py interactive
    
    # View analytics
    python main.py analytics
    
    # Export for NLP pipeline
    python main.py search "AI" --export context
        """
    )
    
    parser.add_argument(
        'command',
        choices=['build', 'search', 'interactive', 'analytics', 'config'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'query',
        nargs='?',
        help='Search query (for search command)'
    )
    
    parser.add_argument(
        '--sources',
        nargs='+',
        help='Sources to extract (URLs or file paths)'
    )
    
    parser.add_argument(
        '--collection',
        type=str,
        default=None,
        help='Collection name'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of results (default: 5)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['parallel', 'fast', 'accurate'],
        default='parallel',
        help='Search mode (default: parallel - runs all 4 strategies)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.3,
        help='Minimum confidence threshold (default: 0.3)'
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
        '--reset',
        action='store_true',
        help='Reset collection before building'
    )
    
    parser.add_argument(
        '--export',
        choices=['context', 'qa', 'chunks'],
        help='Export format for NLP pipeline'
    )
    
    args = parser.parse_args()
    
    # Handle config command
    if args.command == 'config':
        print_config_summary()
        return
    
    # Initialize RAG system
    rag = ProductionRAG()
    
    # Execute command
    if args.command == 'build':
        if not args.sources:
            logger.error("❌ --sources required for build command")
            parser.print_help()
            sys.exit(1)
        
        success = rag.build_collection(
            sources=args.sources,
            collection_name=args.collection,
            chunk=not args.no_chunk,
            use_scrapy=not args.no_scrapy,
            reset=args.reset
        )
        
        sys.exit(0 if success else 1)
    
    elif args.command == 'search':
        if not args.query:
            logger.error("❌ Query required for search command")
            parser.print_help()
            sys.exit(1)
        
        result_file = rag.search(
            query=args.query,
            top_n=args.top_n,
            mode=args.mode,
            display=True,
            export_format=args.export
        )
        
        sys.exit(0 if result_file else 1)
    
    elif args.command == 'interactive':
        rag.interactive_mode()
    
    elif args.command == 'analytics':
        rag.show_analytics()


if __name__ == "__main__":
    main()