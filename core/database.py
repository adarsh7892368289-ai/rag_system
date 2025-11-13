import hashlib
import json
import chromadb
from typing import List, Dict, Any, Optional
from config.settings import DATABASE
from core.embedding import EmbeddingGenerator


class ChromaDBManager:
    
    def __init__(self):
        self.persist_directory = DATABASE.persist_directory
        self.collection_name = DATABASE.collection_name
        self.batch_size = DATABASE.batch_size
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.documents_cache = []
    
    def initialize(self, reset: bool = False):
        """Initialize ChromaDB collection"""
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedding_model = EmbeddingGenerator()
        
        if reset:
            try:
                self.client.delete_collection(self.collection_name)
            except:
                pass
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self._refresh_cache()
        print(f"✅ ChromaDB ready ({self.collection.count()} documents)")
    
    def add_documents(self, documents: List[Any], update_mode: str = 'skip'):
        """Add documents to collection"""
        if not self.collection:
            self.initialize()
        
        doc_dicts = self._to_dict_list(documents)
        
        if update_mode == 'skip':
            doc_dicts = self._skip_existing(doc_dicts)
        elif update_mode == 'replace':
            self._delete_by_ids([d['id'] for d in doc_dicts])
        elif update_mode == 'merge':
            doc_dicts = self._skip_duplicate_content(doc_dicts)
        
        if not doc_dicts:
            print("⚠️  No new documents to add")
            return
        
        texts = [d['content'] for d in doc_dicts]
        ids = [d['id'] for d in doc_dicts]
        
        # CRITICAL FIX: Properly flatten metadata preserving lists
        metadatas = [self._flatten_metadata_safe(d.get('metadata', {})) for d in doc_dicts]
        
        # Generate embeddings (silent mode)
        embeddings = self.embedding_model.encode_batch(texts, show_progress=False).tolist()
        
        # Store in batches
        for i in range(0, len(doc_dicts), self.batch_size):
            end = min(i + self.batch_size, len(doc_dicts))
            
            self.collection.add(
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )
        
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
    
    def get_chunk_with_context(self, chunk_id: str, window: int = 1) -> Optional[Dict[str, Any]]:
        """Get chunk with surrounding context"""
        chunk = self._get_by_id(chunk_id)
        if not chunk:
            return None
        
        chunk_index = chunk['metadata'].get('chunk_index')
        source = chunk['metadata'].get('source')
        
        if chunk_index is None or source is None:
            return {
                'id': chunk_id,
                'content': chunk['content'],
                'metadata': chunk['metadata'],
                'context_window': 0
            }
        
        prev_chunks = self._get_surrounding_chunks(source, chunk_index - window, chunk_index)
        next_chunks = self._get_surrounding_chunks(source, chunk_index + 1, chunk_index + window + 1)
        
        full_context = prev_chunks + [chunk['content']] + next_chunks
        
        return {
            'id': chunk_id,
            'content': '\n\n'.join(full_context),
            'main_chunk': chunk['content'],
            'metadata': chunk['metadata'],
            'context_window': window,
            'chunks_retrieved': len(full_context)
        }
    
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
            except:
                pass
            self.collection = None
            self.documents_cache = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.collection:
            return {'count': 0, 'cached_documents': 0, 'collection_name': self.collection_name}
        
        return {
            'count': self.collection.count(),
            'cached_documents': len(self.documents_cache),
            'collection_name': self.collection_name
        }
    
    def _flatten_metadata_safe(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten metadata for ChromaDB storage
        
        CRITICAL: ChromaDB only supports: str, int, float, bool
        - Lists/dicts must be JSON-encoded
        - None values must be removed
        """
        flattened = {}
        
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                flattened[key] = value
            elif isinstance(value, (list, dict)):
                # JSON encode lists and dicts
                flattened[key] = json.dumps(value)
            else:
                # Convert other types to string
                flattened[key] = str(value)
        
        return flattened
    
    def _unflatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Restore metadata from ChromaDB storage
        
        CRITICAL: Parse JSON-encoded fields back to native types
        """
        unflattened = {}
        
        for key, value in metadata.items():
            # Try to parse JSON strings back to lists/dicts
            if isinstance(value, str) and key in ['keywords', 'strategies_used']:
                try:
                    unflattened[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    unflattened[key] = value
            else:
                unflattened[key] = value
        
        return unflattened
    
    def _get_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Get single document by ID"""
        try:
            result = self.collection.get(ids=[chunk_id])
            if result and result['ids']:
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': self._unflatten_metadata(result['metadatas'][0])
                }
        except:
            pass
        return None
    
    def _get_surrounding_chunks(self, source: str, start_idx: int, end_idx: int) -> List[str]:
        """Get chunks in index range"""
        chunks = []
        for doc in self.documents_cache:
            metadata = doc.get('metadata', {})
            if (metadata.get('source') == source and 
                start_idx <= metadata.get('chunk_index', -1) < end_idx):
                chunks.append(doc['content'])
        
        return sorted(chunks, key=lambda x: self.documents_cache[
            next(i for i, d in enumerate(self.documents_cache) if d['content'] == x)
        ]['metadata'].get('chunk_index', 0))
    
    def _to_dict_list(self, documents: List[Any]) -> List[Dict]:
        """Convert documents to dict format"""
        result = []
        for doc in documents:
            if isinstance(doc, dict):
                result.append(doc)
            else:
                result.append({
                    'id': getattr(doc, 'id', str(doc)),
                    'content': getattr(doc, 'content', str(doc)),
                    'metadata': getattr(doc, 'metadata', {})
                })
        return result
    
    def _refresh_cache(self):
        """Refresh document cache"""
        if not self.collection:
            return
        
        try:
            result = self.collection.get()
            if result and 'ids' in result:
                self.documents_cache = [
                    {
                        'id': result['ids'][i],
                        'content': result['documents'][i],
                        'metadata': self._unflatten_metadata(result['metadatas'][i])
                    }
                    for i in range(len(result['ids']))
                ]
        except:
            self.documents_cache = []
    
    def _skip_existing(self, documents: List[Dict]) -> List[Dict]:
        """Skip documents that already exist"""
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
    
    def _format_results(self, results: Dict, query: str) -> List[Dict[str, Any]]:
        """Format ChromaDB results"""
        if not results or not results.get('ids') or not results['ids'][0]:
            return []
        
        ids = results['ids'][0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        formatted = []
        for i in range(len(ids)):
            # Proper distance to similarity conversion
            similarity = max(0.0, min(1.0, 1.0 - (distances[i] / 2.0)))
            
            # CRITICAL FIX: Unflatten metadata to restore lists
            unflattened_metadata = self._unflatten_metadata(metadatas[i])
            
            formatted.append({
                'id': ids[i],
                'content': documents[i],
                'metadata': unflattened_metadata,
                'similarity_score': similarity,
                'distance': distances[i],
                'rank': i + 1,
                'query': query
            })
        
        return formatted