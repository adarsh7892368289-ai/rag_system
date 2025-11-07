import numpy as np
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from config.settings import SEARCH

class AdvancedSearchStrategies:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.cross_encoder = None
        self.bm25 = None
    
    def semantic_search(self, query: str, n_results: int = 5) -> List[Dict]:
        results = self.db_manager.search(query, n_results=n_results)
        for r in results:
            r['final_score'] = r['similarity_score']
        return results
    
    def bm25_search(self, query: str, n_results: int = 5) -> List[Dict]:
        if not self.bm25:
            self._initialize_bm25()
        
        if not self.bm25:
            return []
        
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(bm25_scores)[::-1][:n_results]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            doc = self.db_manager.documents_cache[idx]
            results.append({
                'id': doc['id'],
                'content': doc['content'],
                'metadata': doc.get('metadata', {}),
                'bm25_score': float(bm25_scores[idx]),
                'final_score': float(bm25_scores[idx]),
                'rank': rank,
                'query': query
            })
        
        return results
    
    def hybrid_search(self, query: str, n_results: int = 5, alpha: float = None) -> List[Dict]:
        if alpha is None:
            alpha = SEARCH.hybrid_alpha
        
        if not self.bm25:
            self._initialize_bm25()
        
        vector_results = self.db_manager.search(query, n_results=n_results * 2)
        
        if not vector_results or not self.bm25:
            return vector_results[:n_results]
        
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        max_bm25 = max(bm25_scores) if bm25_scores.size > 0 else 1
        min_bm25 = min(bm25_scores) if bm25_scores.size > 0 else 0
        
        hybrid_results = []
        for result in vector_results:
            doc_idx = next(
                (i for i, doc in enumerate(self.db_manager.documents_cache)
                 if doc['id'] == result['id']),
                None
            )
            
            if doc_idx is not None:
                bm25_score = ((bm25_scores[doc_idx] - min_bm25) / (max_bm25 - min_bm25) 
                             if max_bm25 > min_bm25 else 0)
                vector_score = result['similarity_score']
                
                hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score
                
                result['bm25_score'] = float(bm25_score)
                result['vector_score'] = float(vector_score)
                result['hybrid_score'] = float(hybrid_score)
                result['final_score'] = float(hybrid_score)
                hybrid_results.append(result)
        
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        for i, result in enumerate(hybrid_results[:n_results], 1):
            result['rank'] = i
        
        return hybrid_results[:n_results]
    
    def mmr_search(self, query: str, n_results: int = 5, lambda_param: float = None) -> List[Dict]:
        if lambda_param is None:
            lambda_param = SEARCH.mmr_lambda
        
        candidates = self.db_manager.search(
            query,
            n_results=min(n_results * SEARCH.mmr_candidates_multiplier, SEARCH.max_top_k)
        )
        
        if not candidates:
            return []
        
        candidate_texts = [c['content'] for c in candidates]
        candidate_embeddings = self.db_manager.embedding_model.encode_batch(candidate_texts)
        
        selected = []
        selected_embeddings = []
        remaining = list(range(len(candidates)))
        
        selected.append(0)
        selected_embeddings.append(candidate_embeddings[0])
        remaining.remove(0)
        
        while len(selected) < n_results and remaining:
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining:
                relevance = candidates[idx]['similarity_score']
                
                candidate_emb = candidate_embeddings[idx].reshape(1, -1)
                selected_embs = np.array(selected_embeddings)
                similarities = cosine_similarity(candidate_emb, selected_embs)[0]
                max_sim = float(np.max(similarities))
                
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                selected_embeddings.append(candidate_embeddings[best_idx])
                remaining.remove(best_idx)
        
        mmr_results = [candidates[i] for i in selected]
        for i, result in enumerate(mmr_results, 1):
            result['mmr_score'] = result['similarity_score']
            result['final_score'] = result['similarity_score']
            result['rank'] = i
        
        return mmr_results
    
    def rerank_search(self, query: str, n_results: int = 5,
                     initial_results: Optional[List[Dict]] = None) -> List[Dict]:
        if initial_results is None:
            initial_results = self.db_manager.search(
                query,
                n_results=min(SEARCH.rerank_top_k, SEARCH.max_top_k)
            )
        
        if not initial_results:
            return []
        
        if not self.cross_encoder:
            print(f"🔄 Loading Cross-Encoder: {SEARCH.rerank_model}")
            self.cross_encoder = CrossEncoder(SEARCH.rerank_model)
        
        pairs = [[query, result['content']] for result in initial_results]
        scores = self.cross_encoder.predict(pairs)
        
        for i, result in enumerate(initial_results):
            result['rerank_score'] = float(scores[i])
            result['final_score'] = float(scores[i])
        
        reranked = sorted(initial_results, key=lambda x: x['rerank_score'], reverse=True)
        
        for i, result in enumerate(reranked[:n_results], 1):
            result['rank'] = i
        
        return reranked[:n_results]
    
    def _initialize_bm25(self):
        if not self.db_manager.documents_cache:
            return
        
        print("🔄 Building BM25 index...")
        corpus = [doc['content'] for doc in self.db_manager.documents_cache]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"✅ BM25 index built")