from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any
import numpy as np


@dataclass
class EthicalSignal:
    """Raw ethical signal from external sources"""
    signal_id: str
    source: str
    content: str
    vector_embedding: np.ndarray
    severity: float  # 0.0 to 1.0
    timestamp: datetime
    category: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'signal_id': self.signal_id,
            'source': self.source,
            'content': self.content,
            'severity': float(self.severity),
            'timestamp': self.timestamp.isoformat(),
            'category': self.category,
            'metadata': self.metadata,
            'embedding_shape': self.vector_embedding.shape
        }
    
    def __repr__(self) -> str:
        return (f"EthicalSignal(id={self.signal_id[:8]}, "
                f"category={self.category}, severity={self.severity:.2f})")
