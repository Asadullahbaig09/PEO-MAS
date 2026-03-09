"""
Document Retriever for Multi-Agent RAG System

Retrieves relevant ethical documents for agent assessments
Supports multiple retrieval strategies and ranking methods
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from src.rag.vector_store import VectorStore
from src.models.signal import EthicalSignal

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    scores: List[float]
    query: str
    domain: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'query': self.query,
            'domain': self.domain,
            'num_results': len(self.documents),
            'documents': self.documents,
            'metadatas': self.metadatas,
            'scores': self.scores
        }
    
    def get_top_k(self, k: int) -> 'RetrievalResult':
        """Get top k results"""
        return RetrievalResult(
            documents=self.documents[:k],
            metadatas=self.metadatas[:k],
            scores=self.scores[:k],
            query=self.query,
            domain=self.domain
        )
    
    def get_context_string(self, max_length: int = 2000) -> str:
        """
        Get concatenated context string for LLM
        
        Args:
            max_length: Maximum total character length
        
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, (doc, meta) in enumerate(zip(self.documents, self.metadatas)):
            # Format document with metadata
            source = meta.get('source', 'Unknown')
            title = meta.get('title', f'Document {i+1}')
            
            doc_text = f"[{source}] {title}:\n{doc}\n"
            
            if current_length + len(doc_text) > max_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n---\n".join(context_parts)


class DocumentRetriever:
    """
    Retrieves relevant documents from vector store
    
    Features:
    - Domain-specific retrieval
    - Hybrid search (semantic + keyword)
    - Metadata filtering
    - Result ranking and re-ranking
    - Context augmentation
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        default_k: int = 5,
        use_reranking: bool = True
    ):
        """
        Initialize document retriever
        
        Args:
            vector_store: Vector store instance
            default_k: Default number of results
            use_reranking: Whether to use re-ranking
        """
        self.vector_store = vector_store
        self.default_k = default_k
        self.use_reranking = use_reranking
        
        logger.info("✓ Document retriever initialized")
    
    def retrieve_for_signal(
        self,
        signal: EthicalSignal,
        domain: str,
        k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve documents relevant to an ethical signal
        
        Args:
            signal: Ethical signal to retrieve context for
            domain: Ethical domain to search
            k: Number of results (uses default if None)
            metadata_filter: Optional metadata filters
        
        Returns:
            Retrieval results
        """
        k = k or self.default_k
        
        # Build query from signal
        query_text = self._build_query_from_signal(signal)
        
        # Retrieve documents
        results = self.vector_store.search(
            domain=domain,
            query_text=query_text,
            query_embedding=signal.vector_embedding,
            n_results=k * 2 if self.use_reranking else k,  # Get more for re-ranking
            metadata_filter=metadata_filter
        )
        
        # Convert distances to similarity scores
        scores = [1.0 / (1.0 + d) for d in results['distances']]
        
        # Re-rank if enabled
        if self.use_reranking and len(results['documents']) > 0:
            results, scores = self._rerank_results(
                query_text=query_text,
                documents=results['documents'],
                metadatas=results['metadatas'],
                scores=scores,
                signal=signal,
                k=k
            )
        
        return RetrievalResult(
            documents=results['documents'][:k],
            metadatas=results['metadatas'][:k],
            scores=scores[:k],
            query=query_text,
            domain=domain
        )
    
    def retrieve_for_query(
        self,
        query: str,
        domain: str,
        k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """
        Retrieve documents for text query
        
        Args:
            query: Query text
            domain: Ethical domain
            k: Number of results
            metadata_filter: Optional metadata filters
        
        Returns:
            Retrieval results
        """
        k = k or self.default_k
        
        results = self.vector_store.search(
            domain=domain,
            query_text=query,
            n_results=k,
            metadata_filter=metadata_filter
        )
        
        scores = [1.0 / (1.0 + d) for d in results['distances']]
        
        return RetrievalResult(
            documents=results['documents'],
            metadatas=results['metadatas'],
            scores=scores,
            query=query,
            domain=domain
        )
    
    def retrieve_multi_domain(
        self,
        signal: EthicalSignal,
        domains: List[str],
        k_per_domain: int = 3
    ) -> Dict[str, RetrievalResult]:
        """
        Retrieve from multiple domains
        
        Args:
            signal: Ethical signal
            domains: List of domains to search
            k_per_domain: Results per domain
        
        Returns:
            Dictionary mapping domain to results
        """
        results = {}
        
        for domain in domains:
            try:
                domain_results = self.retrieve_for_signal(
                    signal=signal,
                    domain=domain,
                    k=k_per_domain
                )
                results[domain] = domain_results
            except Exception as e:
                logger.warning(f"Error retrieving from {domain}: {e}")
                results[domain] = RetrievalResult(
                    documents=[],
                    metadatas=[],
                    scores=[],
                    query="",
                    domain=domain
                )
        
        return results
    
    def _build_query_from_signal(self, signal: EthicalSignal) -> str:
        """Build query text from signal"""
        # Combine content and category for better retrieval
        query_parts = [signal.content]
        
        if signal.category:
            query_parts.append(f"Category: {signal.category}")
        
        if signal.severity > 0.8:
            query_parts.append("high severity critical urgent")
        
        return " ".join(query_parts)
    
    def _rerank_results(
        self,
        query_text: str,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        scores: List[float],
        signal: EthicalSignal,
        k: int
    ) -> tuple:
        """
        Re-rank results using multiple signals
        
        Combines:
        - Semantic similarity score
        - Keyword overlap
        - Recency (if available)
        - Source credibility
        """
        reranked_items = []
        
        query_words = set(query_text.lower().split())
        
        for doc, meta, score in zip(documents, metadatas, scores):
            # Start with base semantic score
            final_score = score
            
            # Keyword overlap bonus
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            keyword_bonus = min(0.2, overlap * 0.01)
            final_score += keyword_bonus
            
            # Recency bonus (if date available)
            if 'date' in meta:
                try:
                    from datetime import datetime
                    doc_date = datetime.fromisoformat(meta['date'])
                    days_old = (datetime.now() - doc_date).days
                    recency_bonus = max(0, 0.1 * (1 - days_old / 365))
                    final_score += recency_bonus
                except:
                    pass
            
            # Source credibility bonus
            source = meta.get('source', '').lower()
            if any(credible in source for credible in ['official', 'policy', 'law', 'regulation']):
                final_score += 0.15
            
            # Severity alignment bonus
            if signal.severity > 0.8 and 'critical' in doc.lower():
                final_score += 0.1
            
            reranked_items.append((final_score, doc, meta))
        
        # Sort by final score
        reranked_items.sort(reverse=True, key=lambda x: x[0])
        
        # Extract top k
        top_items = reranked_items[:k]
        
        return (
            {
                'documents': [item[1] for item in top_items],
                'metadatas': [item[2] for item in top_items]
            },
            [item[0] for item in top_items]
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        domains = self.vector_store.list_all_domains()
        
        stats = {
            'total_domains': len(domains),
            'default_k': self.default_k,
            'use_reranking': self.use_reranking,
            'domains': {}
        }
        
        for domain in domains:
            domain_stats = self.vector_store.get_collection_stats(domain)
            stats['domains'][domain] = domain_stats
        
        return stats
