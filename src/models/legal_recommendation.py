"""
Legal Recommendation Model

Represents AI compliance laws/regulations that should be imposed by governments
based on detected ethical issues.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional


@dataclass
class LegalRecommendation:
    """
    Represents a proposed law/regulation for AI compliance.
    
    Generated when the system detects an ethical issue without 
    existing legal framework.
    """
    
    # Core identification
    id: str
    title: str
    issue_domain: str  # e.g., "privacy", "bias", "transparency"
    
    # Issue details
    detected_issue: str  # Description of the ethical issue found
    severity: float  # 0.0 to 1.0
    affected_signals: List[str]  # Signal IDs that triggered this
    
    # Legal recommendation
    proposed_law: str  # Full text of proposed regulation
    rationale: str  # Why this law is needed
    scope: str  # Who/what it applies to (e.g., "All AI systems processing personal data")
    enforcement_mechanism: str  # How it should be enforced
    
    # Supporting evidence
    evidence: List[Dict[str, str]] = field(default_factory=list)  # From RAG retrieval
    related_regulations: List[str] = field(default_factory=list)  # Existing laws (GDPR, etc.)
    case_studies: List[str] = field(default_factory=list)  # Real-world examples
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    confidence: float = 0.0  # How confident the system is in this recommendation
    source_apis: List[str] = field(default_factory=list)  # Which APIs provided data
    
    # Review status
    status: str = "proposed"  # proposed, under_review, accepted, rejected
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "issue_domain": self.issue_domain,
            "detected_issue": self.detected_issue,
            "severity": self.severity,
            "affected_signals": self.affected_signals,
            "proposed_law": self.proposed_law,
            "rationale": self.rationale,
            "scope": self.scope,
            "enforcement_mechanism": self.enforcement_mechanism,
            "evidence": self.evidence,
            "related_regulations": self.related_regulations,
            "case_studies": self.case_studies,
            "generated_at": self.generated_at.isoformat(),
            "confidence": self.confidence,
            "source_apis": self.source_apis,
            "status": self.status
        }
    
    def __repr__(self) -> str:
        return (
            f"LegalRecommendation(id={self.id}, title='{self.title}', "
            f"domain={self.issue_domain}, severity={self.severity:.2f})"
        )


@dataclass
class ExistingLaw:
    """
    Represents an existing law/regulation relevant to the detected issue.
    
    Retrieved from the RAG system's legal policy collection.
    """
    
    name: str
    jurisdiction: str  # e.g., "EU", "USA", "Global"
    description: str
    relevance_score: float  # How relevant to current issue
    coverage_gap: Optional[str] = None  # What aspect is not covered
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "jurisdiction": self.jurisdiction,
            "description": self.description,
            "relevance_score": self.relevance_score,
            "coverage_gap": self.coverage_gap
        }
