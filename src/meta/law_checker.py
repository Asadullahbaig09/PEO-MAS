"""
Law Checker Module

Checks if existing laws/regulations cover detected ethical issues.
Uses RAG system to retrieve and analyze existing legal frameworks.
"""

import logging
from typing import List, Optional, Dict
from datetime import datetime

from src.models.legal_recommendation import ExistingLaw
from src.rag.retriever import DocumentRetriever

logger = logging.getLogger(__name__)


class LawChecker:
    """
    Checks if existing laws/regulations cover detected ethical issues.
    
    Uses RAG system to:
    1. Retrieve relevant existing laws from vector DB
    2. Analyze coverage of the detected issue
    3. Identify coverage gaps
    """
    
    def __init__(self, retriever: DocumentRetriever):
        """
        Initialize law checker.
        
        Args:
            retriever: RAG document retriever for finding existing laws
        """
        self.retriever = retriever
        
        # Known major AI regulations (for quick reference)
        self.major_regulations = {
            "GDPR": "EU General Data Protection Regulation - Privacy",
            "AI Act": "EU AI Act - AI System Classification and Requirements",
            "CCPA": "California Consumer Privacy Act - Data Privacy",
            "PIPEDA": "Personal Information Protection and Electronic Documents Act (Canada)",
            "Algorithmic Accountability Act": "USA - AI System Auditing and Impact Assessments",
            "AI Bill of Rights": "USA Blueprint for AI Bill of Rights",
        }
        
        logger.info("LawChecker initialized")
    
    def check_existing_coverage(
        self, 
        issue_domain: str, 
        issue_description: str,
        severity: float
    ) -> Dict[str, any]:
        """
        Check if existing laws cover the detected issue.
        
        Args:
            issue_domain: Domain of issue (privacy, bias, transparency, etc.)
            issue_description: Detailed description of the issue
            severity: Issue severity (0.0 to 1.0)
            
        Returns:
            Dict with:
                - has_coverage: bool
                - existing_laws: List[ExistingLaw]
                - coverage_gaps: List[str]
                - needs_new_law: bool
        """
        logger.info(f"🔍 Checking existing laws for {issue_domain} issue (severity: {severity:.2f})")
        
        # Create search query combining domain and description
        search_query = f"{issue_domain} {issue_description}"
        
        # Retrieve relevant documents from RAG system
        # Search across all collections, but prioritize legal/regulatory documents
        try:
            retrieval_result = self.retriever.vector_store.search(
                domain=issue_domain,  # Search in domain-specific collection
                query_text=search_query,
                n_results=5  # Get top 5 relevant documents
            )
            
            # logger.info(f"📚 Retrieved {len(retrieval_result['documents'])} relevant policy documents")
            
        except Exception as e:
            logger.warning(f"⚠️ RAG retrieval failed: {e}. Using fallback.")
            retrieval_result = {"documents": [], "metadatas": [], "distances": []}
        
        # Analyze retrieved documents for existing law coverage
        existing_laws = self._analyze_coverage(
            retrieval_result,
            issue_description
        )
        
        # Identify coverage gaps
        coverage_gaps = self._identify_gaps(
            existing_laws,
            issue_description,
            severity
        )
        
        # Determine if new law is needed
        has_coverage = len(existing_laws) > 0 and len(coverage_gaps) == 0
        needs_new_law = not has_coverage or severity >= 0.8  # High severity always needs review
        
        result = {
            "has_coverage": has_coverage,
            "existing_laws": existing_laws,
            "coverage_gaps": coverage_gaps,
            "needs_new_law": needs_new_law
        }
        
        if needs_new_law:
            logger.info(f"⚖️ NEW LAW NEEDED: {len(coverage_gaps)} gaps identified")
        else:
            logger.info(f"✓ Adequate coverage found: {len(existing_laws)} existing laws")
        
        return result
    
    def _analyze_coverage(
        self, 
        retrieval_result: Dict,
        issue_description: str
    ) -> List[ExistingLaw]:
        """
        Analyze retrieved documents to identify existing laws.
        Marks coverage_gap only when relevance is genuinely low (not just because
        the signal text doesn't appear verbatim in the legal document).
        """
        existing_laws = []
        
        documents = retrieval_result.get("documents", [])
        metadatas = retrieval_result.get("metadatas", [])
        distances = retrieval_result.get("distances", [])
        
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            # Convert L2 distance to similarity (0-1)
            relevance_score = max(0.0, 1.0 - distance / 2.0)
            
            if relevance_score < 0.15:  # Only skip truly irrelevant documents
                continue
            
            law_name = metadata.get("title", f"Regulation {i+1}")
            jurisdiction = metadata.get("source", "Unknown")
            
            # Coverage gap only when semantic relevance is moderate-low (< 0.45).
            # Never check verbatim string match — regulations use systematic language
            # that won't contain the specific signal text.
            coverage_gap = None
            if relevance_score < 0.45:
                coverage_gap = f"Low semantic match ({relevance_score:.2f}) — may not fully address this specific issue"
            
            existing_law = ExistingLaw(
                name=law_name,
                jurisdiction=jurisdiction,
                description=doc[:500],
                relevance_score=relevance_score,
                coverage_gap=coverage_gap
            )
            
            existing_laws.append(existing_law)
            logger.debug(f"  - Found: {law_name} (relevance: {relevance_score:.2f})")
        
        return existing_laws
    
    def _identify_gaps(
        self,
        existing_laws: List[ExistingLaw],
        issue_description: str,
        severity: float
    ) -> List[str]:
        """
        Identify coverage gaps in existing laws.
        Returns empty list when adequate coverage exists (relevance >= 0.45).
        """
        gaps = []
        
        # Check for strong coverage: at least one law with high relevance
        high_relevance = [l for l in existing_laws if l.relevance_score >= 0.35]
        
        if len(existing_laws) == 0:
            gaps.append("No existing regulations found for this issue")
        elif len(high_relevance) == 0:
            # All retrieved laws have low relevance
            gaps.append("Existing regulations have low relevance to this specific issue")
        
        # Add gaps only from laws with genuine partial coverage (not low-relevance noise)
        for law in high_relevance:
            if law.coverage_gap:
                gaps.append(f"{law.name}: {law.coverage_gap}")
        
        # High severity with zero or one matching regulation needs a stronger framework
        if severity >= 0.8 and len(high_relevance) < 2:
            gaps.append("High severity issue requires comprehensive regulatory framework")
        
        return gaps
    
    def get_major_regulations_summary(self) -> List[Dict[str, str]]:
        """
        Get summary of major known AI regulations.
        
        Returns:
            List of regulation summaries
        """
        return [
            {"name": name, "description": desc}
            for name, desc in self.major_regulations.items()
        ]
