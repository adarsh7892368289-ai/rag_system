"""
Production-Ready Vector Search System with Multiple Search Techniques

Demonstrates various search methods:
- Basic Semantic Search (cosine similarity)
- Hybrid Search (vector + keyword)
- MMR (Maximal Marginal Relevance - diversity)
- Re-ranking (cross-encoder for accuracy)
- Filtered Search (metadata-based)

Installation:
    pip install chromadb sentence-transformers scikit-learn numpy rank-bm25

Usage:
    python vector_search.py
"""

import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import Counter
import numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


class EmbeddingGenerator:
    """
    Generate vector embeddings from text
    
    Args:
        model_name: Pre-trained model from Hugging Face
            - 'all-MiniLM-L6-v2': Fast, 384 dim (default)
            - 'all-mpnet-base-v2': Better quality, 768 dim
            - 'paraphrase-multilingual-MiniLM-L12-v2': 50+ languages
    """
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        print(f"🔄 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded ({self.dimension} dimensions)")
    
    def encode(self, text: str) -> List[float]:
        """Convert single text to vector"""
        return self.model.encode(text).tolist()
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Convert multiple texts to vectors (batch processing)"""
        return self.model.encode(texts)

class ChromaDBManager:
    """Manage ChromaDB vector database with search capabilities"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        print(f"\n🔵 Initializing ChromaDB: {persist_directory}")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_model = EmbeddingGenerator()
        self.collection = None
        self.documents_cache = []  # Store for BM25 keyword search
        print("✅ ChromaDB initialized")
    
    def create_collection(self, collection_name: str, reset: bool = False):
        """Create or get collection"""
        if reset:
            try:
                self.client.delete_collection(collection_name)
                print(f"🗑️  Deleted existing collection: {collection_name}")
            except:
                pass
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document embeddings"}
        )
        print(f"✅ Collection '{collection_name}' ready")
        return self.collection
    
    def add_documents(self, documents: List[Dict]):
        """Add documents with embeddings to ChromaDB"""
        if not self.collection:
            raise ValueError("Create collection first")
        
        print(f"\n📥 Adding {len(documents)} documents...")
        
        self.documents_cache = documents
        
        texts = [doc['content'] for doc in documents]
        ids = [doc['id'] for doc in documents]
        metadatas = [doc.get('metadata', {}) for doc in documents]
        
        print("🔄 Generating embeddings...")
        embeddings = self.embedding_model.encode_batch(texts).tolist()
        
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            
            self.collection.add(
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            print(f"   ✓ {end_idx}/{len(documents)}")
        
        print(f"✅ Added {self.collection.count()} documents")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Basic semantic search using cosine similarity
        
        Returns top-k most similar documents
        """
        if not self.collection:
            raise ValueError("Create collection first")
        
        query_embedding = self.embedding_model.encode(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Type-safe checks
        if not results:
            return []
        
        ids = results.get('ids')
        documents = results.get('documents')
        metadatas = results.get('metadatas')
        distances = results.get('distances')
        
        if not ids or not ids[0] or not documents or not metadatas or not distances:
            return []
        
        formatted_results = []
        for i in range(len(ids[0])):
            content = documents[0][i]
            distance = distances[0][i]
            
            formatted_results.append({
                'id': ids[0][i],
                'content': content[:200] + "..." if len(content) > 200 else content,
                'full_content': content,
                'metadata': metadatas[0][i],
                'similarity_score': 1 / (1 + distance),
                'distance': distance
            })
        
        return formatted_results
    
    def filtered_search(self, query: str, filters: Dict, n_results: int = 5) -> List[Dict]:
        """
        Search with metadata filters
        
        Example filters:
            {"source_type": "pdf"}
            {"source_type": {"$in": ["pdf", "web"]}}
            {"chunk_word_count": {"$gt": 100}}
        """
        if not self.collection:
            raise ValueError("Create collection first")
        
        query_embedding = self.embedding_model.encode(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filters,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Type-safe checks
        if not results:
            return []
        
        ids = results.get('ids')
        documents = results.get('documents')
        metadatas = results.get('metadatas')
        distances = results.get('distances')
        
        if not ids or not ids[0] or not documents or not metadatas or not distances:
            return []
        
        formatted_results = []
        for i in range(len(ids[0])):
            content = documents[0][i]
            distance = distances[0][i]
            
            formatted_results.append({
                'id': ids[0][i],
                'content': content[:200] + "..." if len(content) > 200 else content,
                'full_content': content,
                'metadata': metadatas[0][i],
                'similarity_score': 1 / (1 + distance),
                'distance': distance
            })
        
        return formatted_results


class AdvancedSearchTechniques:
    """
    Advanced search methods for better retrieval quality
    
    Implements:
    1. Re-ranking with Cross-Encoder
    2. Hybrid Search (BM25 + Vector)
    3. MMR (Maximal Marginal Relevance)
    """
    
    def __init__(self):
        self.cross_encoder = None
        self.bm25 = None
    
    def rerank_with_cross_encoder(self, query: str, results: List[Dict], 
                                   model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2') -> List[Dict]:
        """
        Re-rank results using Cross-Encoder for higher accuracy
        
        Cross-encoders process query+document together (slower but more accurate)
        Bi-encoders (regular embeddings) encode separately (fast but less accurate)
        
        Use this as a second-stage ranker on top-k results
        """
        print(f"\n🔄 Re-ranking {len(results)} results with Cross-Encoder...")
        
        if not self.cross_encoder:
            self.cross_encoder = CrossEncoder(model_name)
        
        pairs = [[query, result['full_content']] for result in results]
        scores = self.cross_encoder.predict(pairs)
        
        for i, result in enumerate(results):
            result['rerank_score'] = float(scores[i])
        
        reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        print("✅ Re-ranking complete")
        return reranked
    
    def hybrid_search(self, query: str, db_manager: ChromaDBManager, 
                     n_results: int = 5, alpha: float = 0.7) -> List[Dict]:
        """
        Combine BM25 (keyword) + Vector (semantic) search
        
        Args:
            query: Search query
            db_manager: ChromaDB instance
            n_results: Results to return
            alpha: Weight for vector search (0.0=pure BM25, 1.0=pure vector)
        
        BM25 is good for exact matches, vectors for semantic meaning
        """
        print(f"\n🔄 Hybrid Search (α={alpha})...")
        
        # Initialize BM25 if needed
        if not self.bm25:
            print("   Building BM25 index...")
            corpus = [doc['content'] for doc in db_manager.documents_cache]
            tokenized_corpus = [doc.lower().split() for doc in corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Get vector search results
        vector_results = db_manager.search(query, n_results=n_results * 2)
        
        # Get BM25 scores for the same documents
        query_tokens = query.lower().split()
        
        # Create mapping of doc_id to vector result
        vector_dict = {r['id']: r for r in vector_results}
        
        # Calculate BM25 scores for all cached documents
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Normalize scores
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        
        hybrid_results = []
        for result in vector_results:
            doc_idx = next(
                (i for i, doc in enumerate(db_manager.documents_cache) 
                 if doc['id'] == result['id']), 
                None
            )
            
            if doc_idx is not None:
                bm25_score = bm25_scores[doc_idx] / max_bm25
                vector_score = result['similarity_score']
                
                hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score
                
                result['bm25_score'] = bm25_score
                result['vector_score'] = vector_score
                result['hybrid_score'] = hybrid_score
                hybrid_results.append(result)
        
        # Sort by hybrid score
        hybrid_results = sorted(hybrid_results, key=lambda x: x['hybrid_score'], reverse=True)
        print(f"✅ Hybrid search complete")
        return hybrid_results[:n_results]
    
    def mmr_search(self, query: str, db_manager: ChromaDBManager, 
                   n_results: int = 5, lambda_param: float = 0.5) -> List[Dict]:
        """
        Maximal Marginal Relevance - Balance relevance and diversity
        
        Args:
            query: Search query
            db_manager: ChromaDB instance
            n_results: Results to return
            lambda_param: Trade-off (0.0=max diversity, 1.0=max relevance)
        
        Prevents returning multiple similar documents
        Useful for diverse search results
        """
        print(f"\n🔄 MMR Search (λ={lambda_param})...")
        
        # Get more candidates than needed
        candidates = db_manager.search(query, n_results=n_results * 3)
        
        if not candidates:
            return []
        
        # Get embeddings for all candidates
        candidate_texts = [c['full_content'] for c in candidates]
        candidate_embeddings = db_manager.embedding_model.encode_batch(candidate_texts)
        
        selected = []
        selected_embeddings = []
        remaining = list(range(len(candidates)))
        
        # Select first (most relevant)
        selected.append(0)
        selected_embeddings.append(candidate_embeddings[0])
        remaining.remove(0)
        
        # Iteratively select diverse documents
        while len(selected) < n_results and remaining:
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining:
                # Relevance to query
                relevance = candidates[idx]['similarity_score']
                
                # Max similarity to already selected documents
                # Reshape for cosine_similarity: needs 2D arrays
                candidate_emb = candidate_embeddings[idx].reshape(1, -1)
                selected_embs = np.array(selected_embeddings)
                
                similarities = cosine_similarity(candidate_emb, selected_embs)[0]
                max_sim = float(np.max(similarities))
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                selected_embeddings.append(candidate_embeddings[best_idx])
                remaining.remove(best_idx)
        
        mmr_results = [candidates[i] for i in selected]
        for result in mmr_results:
            result['mmr_score'] = result['similarity_score']  # Simplified
        
        print(f"✅ MMR search complete")
        return mmr_results


class SearchComparison:
    """Compare different search techniques side-by-side"""
    
    @staticmethod
    def compare_all_methods(query: str, db_manager: ChromaDBManager, 
                           advanced_search: AdvancedSearchTechniques,
                           n_results: int = 3) -> Dict:
        """
        Run all search methods and compare results
        
        Returns comprehensive comparison for analysis
        """
        print("\n" + "="*70)
        print(f"Comparing Search Methods: '{query}'")
        print("="*70)
        
        results = {}
        
        # 1. Basic Semantic Search
        print("\n1️⃣  Basic Semantic Search")
        semantic_results = db_manager.search(query, n_results=n_results)
        results['semantic'] = semantic_results
        SearchComparison._print_results(semantic_results, "similarity_score")
        
        # 2. Hybrid Search
        print("\n2️⃣  Hybrid Search (BM25 + Vector)")
        hybrid_results = advanced_search.hybrid_search(query, db_manager, n_results=n_results, alpha=0.7)
        results['hybrid'] = hybrid_results
        SearchComparison._print_results(hybrid_results, "hybrid_score")
        
        # 3. MMR Search
        print("\n3️⃣  MMR Search (Diverse Results)")
        mmr_results = advanced_search.mmr_search(query, db_manager, n_results=n_results, lambda_param=0.7)
        results['mmr'] = mmr_results
        SearchComparison._print_results(mmr_results, "similarity_score")
        
        # 4. Re-ranked Search
        print("\n4️⃣  Re-ranked Search (Cross-Encoder)")
        reranked_results = advanced_search.rerank_with_cross_encoder(query, semantic_results[:5])
        results['reranked'] = reranked_results[:n_results]
        SearchComparison._print_results(reranked_results[:n_results], "rerank_score")
        
        return results
    
    @staticmethod
    def _print_results(results: List[Dict], score_key: str):
        """Helper to print search results"""
        for i, result in enumerate(results, 1):
            score = result.get(score_key, 0)
            metadata = result.get('metadata', {})
            source = metadata.get('source', 'Unknown') if isinstance(metadata, dict) else 'Unknown'
            content = result.get('content', '')[:100]
            print(f"\n   [{i}] Score: {score:.3f}")
            print(f"       {content}...")
            print(f"       Source: {source[:60]}")
    
    @staticmethod
    def analyze_overlap(results: Dict) -> Dict:
        """Analyze how much different methods overlap"""
        print("\n" + "="*70)
        print("Search Method Overlap Analysis")
        print("="*70)
        
        methods = list(results.keys())
        overlap = {}
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                method1_results = results.get(method1, [])
                method2_results = results.get(method2, [])
                
                ids1 = set(r.get('id', '') for r in method1_results if isinstance(r, dict))
                ids2 = set(r.get('id', '') for r in method2_results if isinstance(r, dict))
                
                common = len(ids1 & ids2)
                total = len(ids1 | ids2)
                overlap_pct = (common / total * 100) if total > 0 else 0
                
                overlap[f"{method1}_vs_{method2}"] = {
                    'common': common,
                    'overlap_percentage': overlap_pct
                }
                
                print(f"\n{method1.upper()} ↔️ {method2.upper()}")
                print(f"   Common results: {common}/{len(ids1)}")
                print(f"   Overlap: {overlap_pct:.1f}%")
        
        return overlap


def load_extracted_documents(folder_path: str = "extracted_docs") -> List[Dict]:
    """Load documents from JSON files"""
    print(f"\n📂 Loading documents from: {folder_path}")
    
    json_files = [f for f in Path(folder_path).glob("*.json") if f.name != "_index.json"]
    
    if not json_files:
        print(f"⚠️  No JSON files found in {folder_path}")
        return []
    
    all_documents = []
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for chunk in data.get('chunks', []):
            doc = {
                'id': chunk['doc_id'],
                'content': chunk['content'],
                'metadata': {
                    'source': chunk['source'],
                    'source_type': chunk['source_type'],
                    'chunk_id': chunk['chunk_id'],
                    **chunk['metadata']
                }
            }
            all_documents.append(doc)
    
    print(f"✅ Loaded {len(all_documents)} document chunks")
    return all_documents


def save_search_results(results: Dict, query: str, output_dir: str = "search_results"):
    """Save search results for analysis"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Sanitize filename by removing invalid characters
    safe_query = query[:30].replace(' ', '_')
    # Remove invalid filename characters for Windows
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        safe_query = safe_query.replace(char, '')
    
    output_file = Path(output_dir) / f"comparison_{safe_query}.json"
    
    # Convert results to serializable format
    serializable_results = {}
    for method, method_results in results.items():
        serializable_results[method] = [
            {k: v for k, v in r.items() if k != 'full_content'}
            for r in method_results
        ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'query': query,
            'results': serializable_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Search results saved: {output_file}")


if __name__ == "__main__":
    print("="*70)
    print("Vector Search System")
    print("="*70)
    
    # Load documents
    documents = load_extracted_documents("extracted_docs")
    
    if not documents:
        print("\n⚠️  No documents found. Run document_extractor.py first.")
        exit(1)
    
    # Initialize ChromaDB
    chroma = ChromaDBManager()
    chroma.create_collection("my_documents", reset=True)
    chroma.add_documents(documents)
    
    # Initialize advanced search
    advanced = AdvancedSearchTechniques()
    
    # Test queries
    test_queries = [
        "What is embedding?",
        "Explain BERT model",
        "How does Word2vec work?"
    ]
    
    # Compare all methods for each query
    all_comparisons = {}
    for query in test_queries:
        comparison_results = SearchComparison.compare_all_methods(
            query, chroma, advanced, n_results=3
        )
        all_comparisons[query] = comparison_results
        
        # Analyze overlap
        SearchComparison.analyze_overlap(comparison_results)
        
        # Save results
        save_search_results(comparison_results, query)
    
    # Show collection statistics
    print("\n" + "="*70)
    print("Collection Statistics")
    print("="*70)
    
    print(f"Total documents: {len(documents)}")
    
    source_types = Counter(
        doc.get('metadata', {}).get('source_type', 'unknown') 
        if isinstance(doc.get('metadata'), dict) 
        else 'unknown' 
        for doc in documents
    )
    print(f"\nDocument types:")
    for doc_type, count in source_types.items():
        print(f"   {doc_type}: {count} chunks")
    
    avg_chars = sum(len(doc.get('content', '')) for doc in documents) / len(documents) if documents else 0
    print(f"\nAverage chunk size: {avg_chars:.0f} characters")
    
    print("\n" + "="*70)
    print("✅ Search comparison complete")
    print("📁 Results saved in: search_results/")
    print("="*70)