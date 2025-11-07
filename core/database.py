import json
import chromadb
from typing import List, Dict, Any
from config.settings import DATABASE
from core.embedding import EmbeddingGenerator
from utils.helpers import flatten_metadata

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
        print(f"✅ Collection '{self.collection_name}' ready ({self.collection.count()} documents)")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        if not self.collection:
            self.initialize()
        
        print(f"\n📥 Adding {len(documents)} documents...")
        
        self.documents_cache = documents
        
        texts = [doc['content'] for doc in documents]
        ids = [doc['id'] for doc in documents]
        metadatas = [flatten_metadata(doc.get('metadata', {})) for doc in documents]
        
        print("🔄 Generating embeddings...")
        embeddings = self.embedding_model.encode_batch(texts).tolist()
        
        print(f"💾 Storing in database...")
        for i in range(0, len(documents), self.batch_size):
            end_idx = min(i + self.batch_size, len(documents))
            
            self.collection.add(
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            
            if end_idx % (self.batch_size * 5) == 0 or end_idx == len(documents):
                print(f"   ✓ {end_idx}/{len(documents)}")
        
        print(f"✅ Added {self.collection.count()} documents")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        if not self.collection:
            raise ValueError("Initialize collection first")
        
        query_embedding = self.embedding_model.encode(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        return self._format_results(results, query)
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        return self.documents_cache
    
    def clear(self):
        if self.client:
            try:
                self.client.delete_collection(self.collection_name)
                print(f"🗑️  Deleted collection: {self.collection_name}")
            except:
                pass
            self.collection = None
            self.documents_cache = []
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.collection:
            return {'count': 0}
        
        count = self.collection.count()
        return {
            'count': count,
            'collection_name': self.collection_name,
            'cached_documents': len(self.documents_cache)
        }
    
    def _format_results(self, results: Dict, query: str) -> List[Dict[str, Any]]:
        if not results or not results.get('ids') or not results['ids'][0]:
            return []
        
        ids = results['ids'][0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        formatted_results = []
        for i in range(len(ids)):
            similarity_score = 1 / (1 + distances[i])
            
            formatted_results.append({
                'id': ids[i],
                'content': documents[i],
                'metadata': metadatas[i],
                'similarity_score': similarity_score,
                'distance': distances[i],
                'rank': i + 1,
                'query': query
            })
        
        return formatted_results