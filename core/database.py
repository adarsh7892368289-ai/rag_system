import hashlib
import chromadb
from typing import List, Dict, Any, Set
from config.settings import DATABASE
from core.embedding import EmbeddingGenerator
from utils.helpers import flatten_metadata


class ChromaDBManager:
    """Simplified ChromaDB manager with proper similarity scoring"""
    
    def __init__(self):
        self.persist_directory = DATABASE.persist_directory
        self.collection_name = DATABASE.collection_name
        self.batch_size = DATABASE.batch_size
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.documents_cache = []
    
    def initialize(self, reset: bool = False):
        """Initialize ChromaDB connection"""
        print(f"\n🔵 Initializing ChromaDB: {self.persist_directory}")
        
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedding_model = EmbeddingGenerator()
        
        if reset:
            try:
                self.client.delete_collection(self.collection_name)
                print(f"🗑️  Deleted collection: {self.collection_name}")
            except:
                pass
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self._refresh_cache()
        print(f"✅ Collection ready ({self.collection.count()} documents)")
    
    def add_documents(self, documents: List[Any], update_mode: str = 'skip'):
        """Add documents to database"""
        if not self.collection:
            self.initialize()
        
        print(f"\n📥 Adding documents (mode: {update_mode})...")
        
        # Convert documents to dict format
        doc_dicts = self._to_dict_list(documents)
        
        # Handle duplicates
        if update_mode == 'skip':
            doc_dicts = self._skip_existing(doc_dicts)
        elif update_mode == 'replace':
            self._delete_by_ids([d['id'] for d in doc_dicts])
        elif update_mode == 'merge':
            doc_dicts = self._skip_duplicate_content(doc_dicts)
        
        if not doc_dicts:
            print("⚠️  No new documents to add")
            return
        
        # Prepare data
        texts = [d['content'] for d in doc_dicts]
        ids = [d['id'] for d in doc_dicts]
        metadatas = [flatten_metadata(d.get('metadata', {})) for d in doc_dicts]
        
        # Generate embeddings
        print("🔄 Generating embeddings...")
        embeddings = self.embedding_model.encode_batch(texts).tolist()
        
        # Add to database in batches
        print("💾 Storing in database...")
        for i in range(0, len(doc_dicts), self.batch_size):
            end = min(i + self.batch_size, len(doc_dicts))
            
            self.collection.add(
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
            
            if end % 100 == 0 or end == len(doc_dicts):
                print(f"   ✓ {end}/{len(doc_dicts)}")
        
        # Update cache
        self.documents_cache.extend(doc_dicts)
        print(f"✅ Added {len(doc_dicts)} documents (Total: {self.collection.count()})")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.collection:
            raise ValueError("Initialize collection first")
        
        query_embedding = self.embedding_model.encode(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        return self._format_results(results, query)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all documents"""
        if not self.collection:
            return {'ids': [], 'documents': [], 'metadatas': []}
        
        try:
            return self.collection.get()
        except:
            return {'ids': [], 'documents': [], 'metadatas': []}
    
    def clear(self):
        """Clear collection"""
        if self.client:
            try:
                self.client.delete_collection(self.collection_name)
                print(f"🗑️  Deleted collection")
            except:
                pass
            self.collection = None
            self.documents_cache = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        if not self.collection:
            return {'count': 0, 'cached_documents': 0, 'collection_name': self.collection_name}
        
        return {
            'count': self.collection.count(),
            'cached_documents': len(self.documents_cache),
            'collection_name': self.collection_name
        }
    
    def _to_dict_list(self, documents: List[Any]) -> List[Dict]:
        """Convert documents to dict format (handles both dict and object)"""
        result = []
        for doc in documents:
            if isinstance(doc, dict):
                result.append(doc)
            else:
                # Handle Document objects
                result.append({
                    'id': getattr(doc, 'id', str(doc)),
                    'content': getattr(doc, 'content', str(doc)),
                    'metadata': getattr(doc, 'metadata', {})
                })
        return result
    
    def _refresh_cache(self):
        """Refresh cache from database"""
        if not self.collection:
            return
        
        try:
            result = self.collection.get()
            if result and 'ids' in result:
                self.documents_cache = [
                    {
                        'id': result['ids'][i],
                        'content': result['documents'][i],
                        'metadata': result['metadatas'][i]
                    }
                    for i in range(len(result['ids']))
                ]
        except:
            self.documents_cache = []
    
    def _skip_existing(self, documents: List[Dict]) -> List[Dict]:
        """Skip documents with existing IDs"""
        existing_ids = set(self.collection.get()['ids']) if self.collection.count() > 0 else set()
        new_docs = [d for d in documents if d['id'] not in existing_ids]
        
        skipped = len(documents) - len(new_docs)
        if skipped > 0:
            print(f"⚠️  Skipped {skipped} existing documents")
        
        return new_docs
    
    def _skip_duplicate_content(self, documents: List[Dict]) -> List[Dict]:
        """Skip documents with duplicate content"""
        existing_hashes = {
            hashlib.md5(d['content'].encode()).hexdigest() 
            for d in self.documents_cache
        }
        
        new_docs = []
        for doc in documents:
            doc_hash = hashlib.md5(doc['content'].encode()).hexdigest()
            if doc_hash not in existing_hashes:
                new_docs.append(doc)
                existing_hashes.add(doc_hash)
        
        skipped = len(documents) - len(new_docs)
        if skipped > 0:
            print(f"⚠️  Skipped {skipped} duplicate documents")
        
        return new_docs
    
    def _delete_by_ids(self, ids: List[str]):
        """Delete documents by IDs"""
        existing_ids = set(self.collection.get()['ids']) if self.collection.count() > 0 else set()
        to_delete = [id for id in ids if id in existing_ids]
        
        if to_delete:
            self.collection.delete(ids=to_delete)
            self.documents_cache = [d for d in self.documents_cache if d['id'] not in to_delete]
            print(f"🗑️  Deleted {len(to_delete)} documents")
    
    def _format_results(self, results: Dict, query: str) -> List[Dict[str, Any]]:
        """Format search results with proper similarity scores"""
        if not results or not results.get('ids') or not results['ids'][0]:
            return []
        
        ids = results['ids'][0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        formatted = []
        for i in range(len(ids)):
            # ChromaDB cosine distance: distance = 1 - similarity
            # So: similarity = 1 - distance
            similarity = max(0.0, min(1.0, 1.0 - distances[i]))
            
            formatted.append({
                'id': ids[i],
                'content': documents[i],
                'metadata': metadatas[i],
                'similarity_score': similarity,
                'distance': distances[i],
                'rank': i + 1,
                'query': query
            })
        
        return formatted