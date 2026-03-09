import numpy as np
from typing import List
from src.models.signal import EthicalSignal


class AttentionMechanism:
    """Multi-head attention to identify critical signals"""
    
    def __init__(self, embedding_dim: int = 384, num_heads: int = 8):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
    def compute_attention(self, signals: List[EthicalSignal]) -> List[float]:
        """
        Compute attention weights for signals
        Returns normalized attention scores
        """
        if not signals:
            return []
        
        embeddings = np.array([s.vector_embedding for s in signals])
        
        # Compute similarity to mean (simplified attention)
        mean_embedding = embeddings.mean(axis=0)
        attention_scores = []
        
        for emb in embeddings:
            similarity = np.dot(emb, mean_embedding) / (
                np.linalg.norm(emb) * np.linalg.norm(mean_embedding) + 1e-8
            )
            attention_scores.append(float(similarity))
        
        # Normalize using softmax
        attention_scores = self._softmax(np.array(attention_scores))
        return attention_scores.tolist()
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax for normalization"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def filter_critical_signals(
        self, 
        signals: List[EthicalSignal], 
        threshold: float = 0.5
    ) -> List[EthicalSignal]:
        """Filter signals based on attention scores"""
        if not signals:
            return []
        
        attention_scores = self.compute_attention(signals)
        
        critical_signals = [
            signal for signal, score in zip(signals, attention_scores)
            if score > threshold
        ]
        
        return critical_signals
