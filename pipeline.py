import json
from pathlib import Path
from typing import List, Dict, Any

from core.extraction import DocumentExtractor
from core.document_processor import DocumentProcessor
from core.database import ChromaDBManager
from strategies.query_processor import QueryProcessor
from strategies.search_engine import AdvancedSearchStrategies
from strategies.result_fusion import UnifiedSearchEngine
from config.settings import EXTRACTION

class RAGPipeline:
    def __init__(self):
        print("\n" + "="*70)
        print("🚀 INITIALIZING RAG SYSTEM")
        print("="*70)
        
        self.extractor = DocumentExtractor()
        self.processor = DocumentProcessor()
        self.db_manager = ChromaDBManager()
        self.query_processor = QueryProcessor()
        self.search_strategies = None
        self.search_engine = None
        
        print("✅ RAG Pipeline initialized\n")
    
    def ingest(self, sources: List[str], reset: bool = False, save_extracted: bool = True):
        print("\n" + "="*70)
        print("📥 INGESTION PIPELINE")
        print("="*70)
        
        documents = self.extractor.extract_multiple(sources, chunk=True, parallel=True)
        
        if not documents:
            print("❌ No documents extracted")
            return
        
        if save_extracted:
            self.extractor.save_documents(documents)
        
        processed = self.processor.process([doc.to_dict() for doc in documents])
        
        self.db_manager.initialize(reset=reset)
        
        db_documents = []
        for doc in processed:
            db_documents.append({
                'id': doc['doc_id'],
                'content': doc['content'],
                'metadata': doc['metadata']
            })
        
        self.db_manager.add_documents(db_documents)
        
        self.search_strategies = AdvancedSearchStrategies(self.db_manager)
        self.search_engine = UnifiedSearchEngine(self.db_manager, self.search_strategies)
        
        print("\n" + "="*70)
        print("✅ INGESTION COMPLETE")
        print("="*70)
        print(f"   Total documents: {len(db_documents)}")
        print(f"   Database ready for search")
        print("="*70 + "\n")
    
    def load_from_json(self, folder_path: str = None, reset: bool = False):
        if folder_path is None:
            folder_path = EXTRACTION.output_dir
        
        print("\n" + "="*70)
        print(f"📂 LOADING FROM: {folder_path}")
        print("="*70)
        
        json_files = list(Path(folder_path).glob("*.json"))
        
        if not json_files:
            print(f"❌ No JSON files found in {folder_path}")
            return
        
        all_documents = []
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for chunk in data.get('chunks', []):
                doc = {
                    'id': chunk['doc_id'],
                    'content': chunk['content'],
                    'metadata': chunk.get('metadata', {})
                }
                all_documents.append(doc)
        
        print(f"✅ Loaded {len(all_documents)} chunks from {len(json_files)} files")
        
        self.db_manager.initialize(reset=reset)
        self.db_manager.add_documents(all_documents)
        
        self.search_strategies = AdvancedSearchStrategies(self.db_manager)
        self.search_engine = UnifiedSearchEngine(self.db_manager, self.search_strategies)
        
        print("\n" + "="*70)
        print("✅ LOADING COMPLETE")
        print("="*70)
        print(f"   Database ready for search")
        print("="*70 + "\n")
    
    def query(self, query_text: str, top_k: int = 5, mode: str = 'parallel') -> List[Dict[str, Any]]:
        if not self.search_engine:
            raise ValueError("Please run ingest() or load_from_json() first")
        
        processed_query = self.query_processor.process(query_text)
        
        query_to_use = processed_query['expanded'] if mode in ['parallel', 'accurate'] else processed_query['cleaned']
        
        results = self.search_engine.search(query_to_use, top_n=top_k, mode=mode)
        
        self._print_results(results, query_text)
        
        return results
    
    def _print_results(self, results: List[Dict], query: str):
        print("\n" + "="*70)
        print(f"📊 RESULTS FOR: '{query}'")
        print("="*70)
        
        if not results:
            print("   No results found")
        else:
            for i, result in enumerate(results, 1):
                score = result.get('confidence', result.get('final_score', 0))
                content_preview = result['content'][:150].replace('\n', ' ')
                
                print(f"\n{i}. Score: {score:.4f}")
                print(f"   ID: {result['id']}")
                print(f"   Preview: {content_preview}...")
                
                if 'scores' in result:
                    print(f"   Scores: {result['scores']}")
        
        print("\n" + "="*70 + "\n")
    
    def clear_database(self):
        self.db_manager.clear()
        print("🗑️  Database cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        stats = self.db_manager.get_stats()
        print("\n" + "="*70)
        print("📊 SYSTEM STATISTICS")
        print("="*70)
        print(f"   Collection: {stats.get('collection_name', 'N/A')}")
        print(f"   Documents: {stats.get('count', 0)}")
        print(f"   Cached: {stats.get('cached_documents', 0)}")
        print("="*70 + "\n")
        return stats

if __name__ == "__main__":
    pipeline = RAGPipeline()
    
    sources = [
        'path/to/your/document.pdf',
    ]
    
    pipeline.ingest(sources, reset=True)
    
    results = pipeline.query(
        "How do I fix my car's engine?",
        top_k=5,
        mode='parallel'
    )
    
    pipeline.get_stats()