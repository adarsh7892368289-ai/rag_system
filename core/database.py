import hashlib
import json
import chromadb
from typing import List, Dict, Any, Optional
from config.settings import DATABASE
from core.embedding import EmbeddingGenerator


class MultiCollectionManager:
    
    def __init__(self):
        self.persist_directory = DATABASE.persist_directory
        self.batch_size = DATABASE.batch_size
        self.client = None
        self.collections = {}
        self.embedding_model = None
        self.documents_cache = {}
        
        self.collection_names = {
            'sentence_aware': 'rag_sentence_aware',
            'semantic': 'rag_semantic',
            'paragraph': 'rag_paragraph',
            'fixed_size': 'rag_fixed_size'
        }
    
    def initialize(self, reset: bool = False):
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedding_model = EmbeddingGenerator()
        
        for strategy, collection_name in self.collection_names.items():
            if reset:
                try:
                    self.client.delete_collection(collection_name)
                except:
                    pass
            
            self.collections[strategy] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.documents_cache[strategy] = []
        
        self._refresh_all_caches()
        total_docs = sum(col.count() for col in self.collections.values())
        print(f"✅ Multi-collection ChromaDB ready ({total_docs} total documents)")
    
    def add_documents_by_strategy(self, strategy_documents: Dict[str, List[Dict]], 
                                  update_mode: str = 'skip'):
        if not self.collections:
            self.initialize()
        
        total_added = 0
        
        for strategy, documents in strategy_documents.items():
            if strategy not in self.collections:
                print(f"⚠️  Unknown strategy: {strategy}")
                continue
            
            if not documents:
                continue
            
            collection = self.collections[strategy]
            doc_dicts = self._to_dict_list(documents)
            
            if update_mode == 'skip':
                doc_dicts = self._skip_existing(doc_dicts, collection)
            elif update_mode == 'replace':
                self._delete_by_ids([d['id'] for d in doc_dicts], collection)
            elif update_mode == 'merge':
                doc_dicts = self._skip_duplicate_content(doc_dicts, strategy)
            
            if not doc_dicts:
                continue
            
            texts = [d['content'] for d in doc_dicts]
            ids = [d['id'] for d in doc_dicts]
            metadatas = [self._flatten_metadata_safe(d.get('metadata', {})) for d in doc_dicts]
            
            embeddings = self.embedding_model.encode_batch(texts, show_progress=False).tolist()
            
            for i in range(0, len(doc_dicts), self.batch_size):
                end = min(i + self.batch_size, len(doc_dicts))
                
                collection.add(
                    embeddings=embeddings[i:end],
                    documents=texts[i:end],
                    metadatas=metadatas[i:end],
                    ids=ids[i:end]
                )
            
            self.documents_cache[strategy].extend(doc_dicts)
            total_added += len(doc_dicts)
            print(f"   ✓ {strategy}: {len(doc_dicts)} chunks")
        
        total_docs = sum(col.count() for col in self.collections.values())
        print(f"✅ Added {total_added} documents (Total: {total_docs})")
    
    def search(self, query: str, strategy: str, n_results: int = 5) -> List[Dict[str, Any]]:
        if strategy not in self.collections:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        collection = self.collections[strategy]
        query_embedding = self.embedding_model.encode(query)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        return self._format_results(results, query, strategy)
    
    def clear_all(self):
        if self.client:
            for collection_name in self.collection_names.values():
                try:
                    self.client.delete_collection(collection_name)
                except:
                    pass
            self.collections = {}
            self.documents_cache = {}
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {}
        for strategy, collection in self.collections.items():
            stats[strategy] = {
                'count': collection.count(),
                'cached': len(self.documents_cache.get(strategy, []))
            }
        stats['total'] = sum(col.count() for col in self.collections.values())
        return stats
    
    def _flatten_metadata_safe(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        flattened = {}
        
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                flattened[key] = value
            elif isinstance(value, (list, dict)):
                flattened[key] = json.dumps(value)
            else:
                flattened[key] = str(value)
        
        return flattened
    
    def _unflatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        unflattened = {}
        
        for key, value in metadata.items():
            if isinstance(value, str) and key in ['keywords', 'strategies_used']:
                try:
                    unflattened[key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    unflattened[key] = value
            else:
                unflattened[key] = value
        
        return unflattened
    
    def _refresh_all_caches(self):
        for strategy, collection in self.collections.items():
            try:
                result = collection.get()
                if result and 'ids' in result:
                    self.documents_cache[strategy] = [
                        {
                            'id': result['ids'][i],
                            'content': result['documents'][i],
                            'metadata': self._unflatten_metadata(result['metadatas'][i])
                        }
                        for i in range(len(result['ids']))
                    ]
            except:
                self.documents_cache[strategy] = []
    
    def _to_dict_list(self, documents: List[Any]) -> List[Dict]:
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
    
    def _skip_existing(self, documents: List[Dict], collection) -> List[Dict]:
        existing_ids = set(collection.get()['ids']) if collection.count() > 0 else set()
        new_docs = [d for d in documents if d['id'] not in existing_ids]
        
        skipped = len(documents) - len(new_docs)
        if skipped > 0:
            print(f"      ⚠️  Skipped {skipped} existing")
        
        return new_docs
    
    def _skip_duplicate_content(self, documents: List[Dict], strategy: str) -> List[Dict]:
        existing_hashes = {
            hashlib.md5(d['content'].encode()).hexdigest() 
            for d in self.documents_cache.get(strategy, [])
        }
        
        new_docs = []
        for doc in documents:
            doc_hash = hashlib.md5(doc['content'].encode()).hexdigest()
            if doc_hash not in existing_hashes:
                new_docs.append(doc)
                existing_hashes.add(doc_hash)
        
        skipped = len(documents) - len(new_docs)
        if skipped > 0:
            print(f"      ⚠️  Skipped {skipped} duplicates")
        
        return new_docs
    
    def _delete_by_ids(self, ids: List[str], collection):
        existing_ids = set(collection.get()['ids']) if collection.count() > 0 else set()
        to_delete = [id for id in ids if id in existing_ids]
        
        if to_delete:
            collection.delete(ids=to_delete)
    
    def _format_results(self, results: Dict, query: str, strategy: str) -> List[Dict[str, Any]]:
        if not results or not results.get('ids') or not results['ids'][0]:
            return []
        
        ids = results['ids'][0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        formatted = []
        for i in range(len(ids)):
            similarity = max(0.0, min(1.0, 1.0 - (distances[i] / 2.0)))
            unflattened_metadata = self._unflatten_metadata(metadatas[i])
            
            formatted.append({
                'id': ids[i],
                'content': documents[i],
                'metadata': unflattened_metadata,
                'similarity_score': similarity,
                'distance': distances[i],
                'rank': i + 1,
                'query': query,
                'chunking_strategy': strategy
            })
        
        return formatted