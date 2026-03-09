from abc import ABC, abstractmethod
from typing import List
import re
import numpy as np
from datetime import datetime

from src.models.signal import EthicalSignal


class EthicalSignalScraper(ABC):
    """Base class for all signal scrapers"""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.last_scrape = None
        
    @abstractmethod
    def fetch_signals(self) -> List[EthicalSignal]:
        """Fetch and process signals from source"""
        pass

    @staticmethod
    def strip_html(text: str) -> str:
        """Remove HTML tags and collapse whitespace from raw text."""
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Decode common HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>') \
                   .replace('&nbsp;', ' ').replace('&#39;', "'").replace('&quot;', '"')
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def vectorize_content(self, content: str) -> np.ndarray:
        """Vectorize content using the real SentenceTransformer embedding model."""
        if not hasattr(EthicalSignalScraper, '_shared_embedding_engine'):
            from src.knowledge.embeddings import EmbeddingEngine
            EthicalSignalScraper._shared_embedding_engine = EmbeddingEngine()
        return EthicalSignalScraper._shared_embedding_engine.encode(content)
    
    def categorize_content(self, content: str) -> str:
        """Categorize signal based on content keywords"""
        content_lower = content.lower()
        
        categories = {
            'bias': ['bias', 'discrimination', 'fairness', 'equity'],
            'privacy': ['privacy', 'gdpr', 'data protection', 'pii'],
            'transparency': ['transparency', 'explainability', 'interpretability'],
            'accountability': ['accountability', 'liability', 'responsibility'],
            'safety': ['safety', 'harm', 'risk'],
            'security': ['security', 'breach', 'vulnerability']
        }
        
        for category, keywords in categories.items():
            if any(kw in content_lower for kw in keywords):
                return category
        
        return 'general'
    
    def calculate_severity(self, content: str, metadata: dict = None) -> float:
        """Calculate signal severity (0.0 to 1.0)"""
        severity = 0.5
        
        high_severity_keywords = [
            'breach', 'violation', 'illegal', 'harmful', 
            'discriminatory', 'critical', 'urgent'
        ]
        
        content_lower = content.lower()
        for keyword in high_severity_keywords:
            if keyword in content_lower:
                severity += 0.1
        
        return min(1.0, severity)

