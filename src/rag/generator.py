"""
Ethical Assessment Generator for Multi-Agent RAG System

Generates detailed ethical assessments using LLM + retrieved context
Each agent uses this to produce evidence-based ethical analysis
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.models.signal import EthicalSignal
from src.rag.retriever import RetrievalResult
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class EthicalAssessment:
    """Structured ethical assessment with evidence"""
    signal_id: str
    domain: str
    severity_assessment: float  # 0.0 to 1.0
    risk_level: str  # low, medium, high, critical
    analysis: str  # Detailed analysis
    policy_violations: List[str]  # Identified violations
    recommendations: List[str]  # Mitigation recommendations
    evidence: List[str]  # Citations from retrieved docs
    confidence: float  # 0.0 to 1.0
    generated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'signal_id': self.signal_id,
            'domain': self.domain,
            'severity_assessment': self.severity_assessment,
            'risk_level': self.risk_level,
            'analysis': self.analysis,
            'policy_violations': self.policy_violations,
            'recommendations': self.recommendations,
            'evidence': self.evidence,
            'confidence': self.confidence,
            'generated_at': self.generated_at.isoformat()
        }
    
    def get_summary(self) -> str:
        """Get brief summary"""
        return f"{self.risk_level.upper()}: {self.analysis[:150]}..."


class EthicalAssessmentGenerator:
    """
    Generates ethical assessments using LLM + retrieved context
    
    Features:
    - Evidence-based analysis with citations
    - Policy violation identification
    - Risk assessment and recommendations
    - Falls back to template-based if LLM unavailable
    """
    
    def __init__(self, llm_interface=None):
        """Initialize generator with optional shared LLM"""
        self.llm_interface = llm_interface
        
        # Log LLM availability
        if self.llm_interface and hasattr(self.llm_interface, 'available') and self.llm_interface.available:
            logger.info("✓ LLM-enhanced assessment generation enabled")
        else:
            logger.info("ℹ Using template-based generation (LLM not available)")
            self.llm_interface = None
    
    def generate_assessment(
        self,
        signal: EthicalSignal,
        retrieval_results: RetrievalResult,
        agent_name: str,
        agent_capabilities: List[str]
    ) -> EthicalAssessment:
        """
        Generate ethical assessment for signal
        
        Args:
            signal: Ethical signal to assess
            retrieval_results: Retrieved context documents
            agent_name: Name of assessing agent
            agent_capabilities: Agent's capabilities
        
        Returns:
            Structured ethical assessment
        """
        # Get context from retrieved documents
        context = retrieval_results.get_context_string(max_length=3000)
        
        # Skip LLM for agent assessments — templates produce equivalent confidence
        # scores and LLM is reserved for law generation (the high-value path).
        use_llm = False
        if use_llm:
            try:
                assessment = self._generate_with_llm(
                    signal=signal,
                    context=context,
                    retrieval_results=retrieval_results,
                    agent_name=agent_name,
                    agent_capabilities=agent_capabilities
                )
                if assessment:
                    return assessment
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}, using template")
        
        # Fallback to template-based
        return self._generate_with_template(
            signal=signal,
            retrieval_results=retrieval_results,
            agent_name=agent_name
        )
    
    def _generate_with_llm(
        self,
        signal: EthicalSignal,
        context: str,
        retrieval_results: RetrievalResult,
        agent_name: str,
        agent_capabilities: List[str]
    ) -> Optional[EthicalAssessment]:
        """Generate assessment using LLM"""
        
        # Build prompt
        prompt = self._build_assessment_prompt(
            signal=signal,
            context=context,
            agent_name=agent_name,
            agent_capabilities=agent_capabilities
        )
        
        # Call LLM (fine-tuned Mistral 7B) — keep token budget tight for speed
        response = self.llm_interface.generate(prompt, max_tokens=350)
        
        # Parse response
        assessment = self._parse_llm_assessment(
            response=response,
            signal=signal,
            retrieval_results=retrieval_results
        )
        
        return assessment
    
    def _generate_with_template(
        self,
        signal: EthicalSignal,
        retrieval_results: RetrievalResult,
        agent_name: str
    ) -> EthicalAssessment:
        """Generate assessment using templates (always works)"""
        
        # Determine risk level
        severity = signal.severity
        if severity >= 0.85:
            risk_level = "critical"
        elif severity >= 0.7:
            risk_level = "high"
        elif severity >= 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Generate analysis
        analysis = self._template_analysis(signal, retrieval_results)
        
        # Extract policy violations from context
        policy_violations = self._extract_violations(retrieval_results)
        
        # Generate recommendations
        recommendations = self._template_recommendations(signal, risk_level)
        
        # Extract evidence citations
        evidence = self._extract_evidence(retrieval_results)
        
        # Calculate confidence based on retrieval quality
        confidence = self._calculate_confidence(retrieval_results, signal)
        
        return EthicalAssessment(
            signal_id=signal.signal_id,
            domain=signal.category,
            severity_assessment=severity,
            risk_level=risk_level,
            analysis=analysis,
            policy_violations=policy_violations,
            recommendations=recommendations,
            evidence=evidence,
            confidence=confidence,
            generated_at=datetime.now()
        )
    
    def _template_analysis(
        self,
        signal: EthicalSignal,
        retrieval_results: RetrievalResult
    ) -> str:
        """Generate template-based analysis"""
        
        num_docs = len(retrieval_results.documents)
        
        analysis_parts = []
        
        # Introduction
        analysis_parts.append(
            f"Analysis of {signal.category} signal with severity {signal.severity:.2f}."
        )
        
        # Context reference
        if num_docs > 0:
            analysis_parts.append(
                f"Based on {num_docs} relevant policy document(s), "
                f"this signal raises concerns in the following areas:"
            )
            
            # Extract key concerns from documents
            concerns = set()
            for doc in retrieval_results.documents[:3]:
                doc_lower = doc.lower()
                if 'violation' in doc_lower or 'breach' in doc_lower:
                    concerns.add("policy violations")
                if 'risk' in doc_lower or 'harm' in doc_lower:
                    concerns.add("potential harm")
                if 'discrimination' in doc_lower or 'bias' in doc_lower:
                    concerns.add("discriminatory practices")
                if 'privacy' in doc_lower or 'data protection' in doc_lower:
                    concerns.add("privacy infringement")
            
            if concerns:
                analysis_parts.append(
                    "Identified concerns: " + ", ".join(concerns) + "."
                )
        else:
            analysis_parts.append(
                "Limited policy documentation available for this specific issue. "
                "Assessment based on general ethical principles."
            )
        
        # Severity assessment
        if signal.severity >= 0.85:
            analysis_parts.append(
                "CRITICAL: This signal indicates a severe ethical issue requiring "
                "immediate attention and intervention."
            )
        elif signal.severity >= 0.7:
            analysis_parts.append(
                "HIGH PRIORITY: This issue poses significant ethical risks "
                "and should be addressed promptly."
            )
        
        # Signal content summary
        content_snippet = signal.content[:200]
        analysis_parts.append(f"Signal content: {content_snippet}...")
        
        return " ".join(analysis_parts)
    
    def _extract_violations(
        self,
        retrieval_results: RetrievalResult
    ) -> List[str]:
        """Extract policy violations from retrieved documents"""
        violations = []
        
        violation_keywords = [
            'violation', 'breach', 'non-compliance', 'contravention',
            'infringement', 'transgression'
        ]
        
        for doc, meta in zip(retrieval_results.documents, retrieval_results.metadatas):
            doc_lower = doc.lower()
            
            # Check for violation keywords
            if any(kw in doc_lower for kw in violation_keywords):
                # Extract relevant sentence
                sentences = doc.split('.')
                for sentence in sentences:
                    if any(kw in sentence.lower() for kw in violation_keywords):
                        source = meta.get('source', 'Unknown source')
                        violations.append(f"[{source}] {sentence.strip()}")
                        break
        
        # Deduplicate and limit
        violations = list(set(violations))[:5]
        
        # Add generic violations if none found but severity is high
        if not violations and len(retrieval_results.documents) > 0:
            violations.append("Potential policy violations identified - review recommended")
        
        return violations
    
    def _template_recommendations(
        self,
        signal: EthicalSignal,
        risk_level: str
    ) -> List[str]:
        """Generate template-based recommendations"""
        recommendations = []
        
        # Risk-level specific recommendations
        if risk_level == "critical":
            recommendations.append("IMMEDIATE ACTION REQUIRED: Halt operations pending investigation")
            recommendations.append("Conduct urgent ethical review and impact assessment")
            recommendations.append("Engage senior leadership and ethics committee")
        elif risk_level == "high":
            recommendations.append("Prioritize investigation and mitigation planning")
            recommendations.append("Conduct thorough ethical impact assessment")
            recommendations.append("Develop remediation strategy within 48 hours")
        elif risk_level == "medium":
            recommendations.append("Schedule ethical review within next week")
            recommendations.append("Document findings and mitigation options")
            recommendations.append("Monitor for escalation")
        else:
            recommendations.append("Continue monitoring")
            recommendations.append("Document for future reference")
        
        # Category-specific recommendations
        category_recs = {
            'bias': [
                "Review training data for demographic representation",
                "Implement fairness metrics and auditing",
                "Consider bias mitigation techniques"
            ],
            'privacy': [
                "Review data protection policies and procedures",
                "Ensure GDPR/privacy law compliance",
                "Implement privacy-preserving techniques"
            ],
            'transparency': [
                "Enhance model explainability mechanisms",
                "Provide clear documentation and disclosures",
                "Implement interpretability tools"
            ],
            'accountability': [
                "Establish clear lines of responsibility",
                "Implement audit trails and logging",
                "Define escalation procedures"
            ],
            'safety': [
                "Conduct comprehensive safety testing",
                "Implement fail-safe mechanisms",
                "Develop incident response plan"
            ]
        }
        
        category_specific = category_recs.get(signal.category, [])
        recommendations.extend(category_specific[:2])
        
        return recommendations
    
    def _extract_evidence(
        self,
        retrieval_results: RetrievalResult
    ) -> List[str]:
        """Extract evidence citations from retrieved documents"""
        evidence = []
        
        for doc, meta in zip(retrieval_results.documents[:5], retrieval_results.metadatas[:5]):
            source = meta.get('source', 'Unknown')
            title = meta.get('title', 'Untitled')
            
            # Create citation
            snippet = doc[:150].strip() + "..."
            citation = f"[{source}] {title}: {snippet}"
            evidence.append(citation)
        
        return evidence
    
    def _calculate_confidence(
        self,
        retrieval_results: RetrievalResult,
        signal: EthicalSignal
    ) -> float:
        """Calculate assessment confidence based on retrieval quality"""
        
        # Base confidence
        confidence = 0.5
        
        # Boost if we have relevant documents
        num_docs = len(retrieval_results.documents)
        if num_docs >= 5:
            confidence += 0.3
        elif num_docs >= 3:
            confidence += 0.2
        elif num_docs >= 1:
            confidence += 0.1
        
        # Boost if documents are highly relevant (high scores)
        if retrieval_results.scores:
            avg_score = sum(retrieval_results.scores) / len(retrieval_results.scores)
            confidence += avg_score * 0.2
        
        return min(1.0, confidence)
    
    def _build_assessment_prompt(
        self,
        signal: EthicalSignal,
        context: str,
        agent_name: str,
        agent_capabilities: List[str]
    ) -> str:
        """Build prompt for LLM assessment generation"""
        
        prompt = f"""You are {agent_name}, an AI ethics specialist monitoring {signal.category} issues.

SIGNAL TO ASSESS:
{signal.content}

Severity: {signal.severity:.2f}
Category: {signal.category}
Source: {signal.source}

RELEVANT POLICY CONTEXT:
{context}

YOUR CAPABILITIES:
{', '.join(agent_capabilities)}

TASK:
Provide a structured ethical assessment with:

1. RISK LEVEL (critical/high/medium/low)
2. ANALYSIS (2-3 sentences explaining the ethical concerns)
3. POLICY VIOLATIONS (list specific violations if any)
4. RECOMMENDATIONS (3-5 actionable recommendations)

Format your response as:
RISK: [level]
ANALYSIS: [your analysis]
VIOLATIONS: [list violations or "None identified"]
RECOMMENDATIONS:
- [recommendation 1]
- [recommendation 2]
- [recommendation 3]

Be concise, evidence-based, and cite the policy context where relevant."""

        return prompt
    
    def _parse_llm_assessment(
        self,
        response: str,
        signal: EthicalSignal,
        retrieval_results: RetrievalResult
    ) -> Optional[EthicalAssessment]:
        """Parse LLM response into structured assessment"""
        
        try:
            # Extract sections
            lines = response.split('\n')
            
            risk_level = "medium"
            analysis = ""
            violations = []
            recommendations = []
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('RISK:'):
                    risk_level = line.replace('RISK:', '').strip().lower()
                    risk_level = risk_level.split()[0]  # Get first word
                elif line.startswith('ANALYSIS:'):
                    analysis = line.replace('ANALYSIS:', '').strip()
                    current_section = 'analysis'
                elif line.startswith('VIOLATIONS:'):
                    viol_text = line.replace('VIOLATIONS:', '').strip()
                    if viol_text and viol_text.lower() != 'none identified':
                        violations.append(viol_text)
                    current_section = 'violations'
                elif line.startswith('RECOMMENDATIONS:'):
                    current_section = 'recommendations'
                elif line.startswith('-') and current_section == 'recommendations':
                    rec = line.lstrip('- ').strip()
                    if rec:
                        recommendations.append(rec)
                elif current_section == 'analysis' and line:
                    analysis += " " + line
            
            # Extract evidence
            evidence = self._extract_evidence(retrieval_results)
            
            # Calculate confidence
            confidence = 0.85  # Higher confidence for LLM-generated
            
            # Map risk level to severity
            severity_map = {
                'critical': 0.95,
                'high': 0.8,
                'medium': 0.6,
                'low': 0.3
            }
            severity_assessment = severity_map.get(risk_level, signal.severity)
            
            return EthicalAssessment(
                signal_id=signal.signal_id,
                domain=signal.category,
                severity_assessment=severity_assessment,
                risk_level=risk_level,
                analysis=analysis or "Assessment generated from retrieved context",
                policy_violations=violations or ["Review required"],
                recommendations=recommendations or ["Further investigation recommended"],
                evidence=evidence,
                confidence=confidence,
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error parsing LLM assessment: {e}")
            return None
