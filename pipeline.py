from typing import List, Dict, Any
from core.extraction import DocumentExtractor
from core.database import ChromaDBManager
from strategies.search_strategies import AdvancedSearchStrategies
from strategies.search_engine import UnifiedSearchEngine

class RAGPipeline:
    """Main RAG pipeline"""
    
    def __init__(self):
        self.extractor = DocumentExtractor()
        self.db_manager = ChromaDBManager()
        self.db_manager.initialize()
        
        self.search_strategies = AdvancedSearchStrategies(self.db_manager)
        self.search_engine = UnifiedSearchEngine(self.db_manager, self.search_strategies)
    
    def ingest(self, sources: List[str], reset: bool = False, 
               save_extracted: bool = True, update_mode: str = 'skip'):
        """Ingest documents from sources"""
        
        # Extract documents
        documents = self.extractor.extract_multiple(sources, chunk=True, parallel=True)
        
        if not documents:
            print("⚠️  No documents extracted")
            return
        
        # Save to JSON
        if save_extracted:
            self.extractor.save_documents(documents)
        
        # Initialize database
        if reset or not self.db_manager.collection:
            self.db_manager.initialize(reset=reset)
        
        # Add documents
        self.db_manager.add_documents(documents, update_mode=update_mode)
        
        # Invalidate caches
        self.search_strategies.invalidate_caches()
        
        print(f"\n✅ Ingestion complete: {self.db_manager.collection.count()} documents")
    
    def load_from_json(self, folder_path: str, reset: bool = False, 
                      update_mode: str = 'skip'):
        """Load documents from saved JSON files"""
        
        documents = self.extractor.load_from_folder(folder_path)
        
        if not documents:
            print("⚠️  No documents loaded")
            return
        
        if reset or not self.db_manager.collection:
            self.db_manager.initialize(reset=reset)
        
        self.db_manager.add_documents(documents, update_mode=update_mode)
        self.search_strategies.invalidate_caches()
        
        print(f"\n✅ Loaded: {self.db_manager.collection.count()} documents")
    
    def query(self, query_text: str, top_k: int = 5, mode: str = None, 
             auto_route: bool = True) -> List[Dict[str, Any]]:
        """Search for documents"""
        return self.search_engine.search(query_text, top_n=top_k, mode=mode, auto_route=auto_route)
    
    def clear_database(self):
        """Clear all documents"""
        self.db_manager.clear()
        self.db_manager.initialize()
        self.search_strategies.invalidate_caches()
        print("✅ Database cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return self.db_manager.get_stats()