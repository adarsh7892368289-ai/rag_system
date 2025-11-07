import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING

class EmbeddingGenerator:
    def __init__(self):
        self.model_name = EMBEDDING.model_name
        self.batch_size = EMBEDDING.batch_size
        self.normalize = EMBEDDING.normalize_embeddings
        self.device = EMBEDDING.device
        self.model = None
    
    def load_model(self):
        if self.model is None:
            print(f"🔄 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=EMBEDDING.cache_dir
            )
            print(f"✅ Model loaded")
        return self.model
    
    def encode(self, text: str) -> List[float]:
        model = self.load_model()
        embedding = model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        model = self.load_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True
        )
        return embeddings