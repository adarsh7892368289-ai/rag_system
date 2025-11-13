import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING


class EmbeddingGenerator:
    """Generate embeddings with fixed model loading"""
    
    def __init__(self):
        self.model_name = EMBEDDING.model_name
        self.batch_size = EMBEDDING.batch_size
        # FIX: Load model immediately with explicit device
        self._model = SentenceTransformer(self.model_name, device='cpu')
    
    @property
    def model(self):
        """Get the model instance"""
        return self._model
    
    def encode(self, text: str) -> np.ndarray:
        """Encode single text"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def encode_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode batch of texts
        
        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress bar
        """
        if not texts:
            return np.array([])
        
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))