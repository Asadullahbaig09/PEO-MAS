import numpy as np
from typing import List, Union
from pathlib import Path


class EmbeddingEngine:
    """Handles text vectorization for semantic similarity"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "./models"):
        self.model_name = model_name
        self.dimension = 384
        self.cache_dir = Path(cache_dir)
        
        # Load actual sentence transformer model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, cache_folder=str(self.cache_dir))
            print(f"✓ Loaded embedding model: {model_name}")
        except ImportError:
            print("⚠️  sentence-transformers not installed, using random embeddings")
            print("   Install with: pip install sentence-transformers")
            self.model = None
        
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text into vector embeddings
        """
        if self.model is None:
            # Fallback to random embeddings if model not loaded
            if isinstance(texts, str):
                return np.random.rand(self.dimension)
            return np.random.rand(len(texts), self.dimension)
        
        # Use real model
        return self.model.encode(texts, convert_to_numpy=True)
        """Compute cosine similarity between two embeddings"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))

