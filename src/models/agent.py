from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class AgentSpecification:
    """Specification for a specialized ethical agent with RAG support"""
    agent_id: str
    name: str
    domain: str
    capabilities: List[str]
    prompt_template: str
    success_metrics: Dict[str, float]
    tools: List[str]
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # RAG-specific fields
    use_rag: bool = True  # Enable RAG by default
    retrieval_k: int = 5  # Number of documents to retrieve
    context_window: int = 3000  # Max context characters for LLM
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'domain': self.domain,
            'capabilities': self.capabilities,
            'prompt_template': self.prompt_template,
            'success_metrics': self.success_metrics,
            'tools': self.tools,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata,
            'use_rag': self.use_rag,
            'retrieval_k': self.retrieval_k,
            'context_window': self.context_window
        }
    
    def __repr__(self) -> str:
        return f"AgentSpec(id={self.agent_id}, name={self.name}, domain={self.domain})"


@dataclass
class AgentPerformanceMetrics:
    """Track agent performance over time"""
    agent_id: str
    signals_processed: int = 0
    successful_assessments: int = 0
    failed_assessments: int = 0
    average_confidence: float = 0.0
    coverage_score: float = 0.7
    last_updated: datetime = field(default_factory=datetime.now)
    
    # RAG-specific metrics
    total_documents_retrieved: int = 0
    avg_retrieval_relevance: float = 0.0
    rag_assessments_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.successful_assessments + self.failed_assessments
        return self.successful_assessments / total if total > 0 else 0.0
    
    @property
    def avg_docs_per_assessment(self) -> float:
        """Average documents retrieved per assessment"""
        return self.total_documents_retrieved / self.rag_assessments_count if self.rag_assessments_count > 0 else 0.0
    
    def update_metrics(self, success: bool, confidence: float):
        """Update metrics after processing a signal"""
        self.signals_processed += 1
        if success:
            self.successful_assessments += 1
        else:
            self.failed_assessments += 1
        
        n = self.signals_processed
        self.average_confidence = (
            (self.average_confidence * (n - 1) + confidence) / n
        )
        self.last_updated = datetime.now()
    
    def update_rag_metrics(self, num_docs_retrieved: int, avg_relevance: float):
        """Update RAG-specific metrics"""
        self.total_documents_retrieved += num_docs_retrieved
        self.rag_assessments_count += 1
        
        n = self.rag_assessments_count
        self.avg_retrieval_relevance = (
            (self.avg_retrieval_relevance * (n - 1) + avg_relevance) / n
        )

