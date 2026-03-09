"""
Law Generator Module

Generates proposed AI compliance laws based on detected ethical issues
and latest information from APIs.
"""

import logging
import uuid
from typing import List, Dict, Optional
from datetime import datetime

from src.models.legal_recommendation import LegalRecommendation, ExistingLaw
from src.models.signal import EthicalSignal
from src.meta.llm_interface import LocalLLMInterface

logger = logging.getLogger(__name__)


class LawGenerator:
    """
    Generates proposed laws/regulations for AI compliance.
    
    Uses:
    1. Latest information from API signals
    2. LLM for law text generation
    3. Existing law analysis for context
    4. Evidence from RAG system
    """
    
    def __init__(self, llm_interface: Optional[LocalLLMInterface] = None):
        """
        Initialize law generator.
        
        Args:
            llm_interface: LLM interface for generating law text (optional)
        """
        self.llm = llm_interface
        self.law_templates = self._load_law_templates()
        
        # LLM usage: always use the fine-tuned model when available
        self._llm_calls  = 0      # counter (no cap)
        self._llm_severity_threshold = 0.0  # Use LLM for ALL law generation
        
        # Cache VectorStore for RAG to avoid re-initialization
        self._rag_store = None
        try:
            from src.rag.vector_store import VectorStore
            self._rag_store = VectorStore(persist_directory="data/chromadb")
            logger.info("✓ RAG VectorStore cached for law generation")
        except Exception as e:
            logger.warning(f"RAG VectorStore unavailable: {e}")
        
        logger.info("LawGenerator initialized")
    
    def generate_law_recommendation(
        self,
        issue_domain: str,
        issue_description: str,
        severity: float,
        affected_signals: List[EthicalSignal],
        existing_laws: List[ExistingLaw],
        coverage_gaps: List[str],
        evidence: List[Dict[str, str]] = None
    ) -> LegalRecommendation:
        """
        Generate a proposed law/regulation.
        
        Args:
            issue_domain: Domain of issue (privacy, bias, etc.)
            issue_description: Detailed description
            severity: Issue severity (0.0 to 1.0)
            affected_signals: Signals that triggered this
            existing_laws: Related existing laws
            coverage_gaps: Identified gaps in coverage
            evidence: Supporting evidence from RAG
            
        Returns:
            LegalRecommendation object
        """
        logger.info(f"⚖️ Generating law recommendation for {issue_domain} (severity: {severity:.2f})")
        
        # Extract latest information from signals
        latest_info = self._extract_latest_info(affected_signals)
        
        # Always generate law text using the fine-tuned LLM
        self._llm_calls += 1
        logger.info(f"🧠 Using fine-tuned LLM for law generation (call #{self._llm_calls}, severity={severity:.2f})")
        law_text = self._generate_with_llm(
            issue_domain,
            issue_description,
            severity,
            latest_info,
            existing_laws,
            coverage_gaps
        )
        
        # Generate rationale
        rationale = self._generate_rationale(
            issue_description,
            severity,
            coverage_gaps,
            latest_info
        )
        
        # Determine scope and enforcement
        scope = self._determine_scope(issue_domain, severity)
        enforcement = self._determine_enforcement(issue_domain, severity)
        
        # Create recommendation
        recommendation = LegalRecommendation(
            id=str(uuid.uuid4()),
            title=f"Proposed {issue_domain.title()} Regulation for AI Systems",
            issue_domain=issue_domain,
            detected_issue=issue_description,
            severity=severity,
            affected_signals=[s.signal_id for s in affected_signals],
            proposed_law=law_text,
            rationale=rationale,
            scope=scope,
            enforcement_mechanism=enforcement,
            evidence=evidence or [],
            related_regulations=[law.name for law in existing_laws],
            case_studies=latest_info.get("case_studies", []),
            confidence=0.85 if self.llm else 0.70,  # Higher confidence with LLM
            source_apis=latest_info.get("sources", []),
            status="proposed"
        )
        
        logger.info(f"✓ Generated law recommendation: {recommendation.title}")
        return recommendation
    
    def _extract_latest_info(self, signals: List[EthicalSignal]) -> Dict:
        """
        Extract latest information from signals.
        
        Args:
            signals: List of signals that triggered the issue
            
        Returns:
            Dict with latest information, sources, case studies
        """
        latest_info = {
            "sources": [],
            "case_studies": [],
            "key_concerns": [],
            "timestamps": []
        }
        
        for signal in signals:
            # Extract source API
            source = signal.metadata.get("source", "Unknown")
            if source not in latest_info["sources"]:
                latest_info["sources"].append(source)
            
            # Extract case study from content
            if len(signal.content) > 100:  # Meaningful content
                case_study = f"{source}: {signal.content[:200]}..."
                latest_info["case_studies"].append(case_study)
            
            # Extract key concerns from category
            latest_info["key_concerns"].append(signal.category)
            
            # Track timestamps
            latest_info["timestamps"].append(signal.timestamp)
        
        # Sort by most recent
        latest_info["timestamps"].sort(reverse=True)
        latest_info["case_studies"] = latest_info["case_studies"][:5]  # Top 5
        
        return latest_info
    
    def _generate_with_llm(
        self,
        issue_domain: str,
        issue_description: str,
        severity: float,
        latest_info: Dict,
        existing_laws: List[ExistingLaw],
        coverage_gaps: List[str]
    ) -> str:
        """
        Generate law text using LLM with RAG-enhanced prompting.
        
        Args:
            issue_domain: Domain of issue
            issue_description: Issue description
            severity: Severity level
            latest_info: Latest information from APIs
            existing_laws: Related existing laws
            coverage_gaps: Coverage gaps
            
        Returns:
            Generated law text
        """
        # Build concise prompt — model is fine-tuned so needs minimal instruction
        prompt = f"""You are an expert legal drafter. Generate an enforceable AI regulation law.

Domain: {issue_domain.upper()}
Issue: {issue_description[:300]}
Severity: {self._severity_to_label(severity)} ({severity:.2f})

Coverage Gaps: {'; '.join(coverage_gaps[:3]) if coverage_gaps else 'General coverage needed'}

Generate a proposed law with: Title, Article 1 (Definitions/Scope), Article 2 (Requirements), Article 3 (Enforcement with penalties), Article 4 (Implementation timeline).

**PROPOSED LAW:**"""

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                law_text = self.llm.generate(prompt, max_tokens=250)
                logger.info(f"✓ LLM generated law text (attempt {attempt})")
                return law_text
            except Exception as e:
                logger.warning(f"⚠️ LLM attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    import time
                    time.sleep(2)  # Brief pause before retry
        # Final fallback: generate with a simpler prompt
        try:
            simple_prompt = f"Draft an AI regulation law for {issue_domain}: {issue_description[:150]}\n\nTitle:"
            law_text = self.llm.generate(simple_prompt, max_tokens=200)
            logger.info("✓ LLM generated law text with simplified prompt")
            return law_text
        except Exception as e:
            logger.error(f"❌ All LLM attempts failed: {e}")
            # Last resort: return a minimal LLM-style law (no 'Section 1: Purpose' marker)
            return f"""**Title: {issue_domain.title()} AI Compliance Act**\n\n**Article 1 (Scope):** This regulation applies to all AI systems operating in the {issue_domain} domain.\n\n**Article 2 (Requirements):** Organizations must implement safeguards addressing: {issue_description[:200]}\n\n**Article 3 (Enforcement):** Violations subject to penalties up to 4% of annual revenue.\n\n**Article 4 (Timeline):** Compliance required within 12 months of enactment."""
    
    def _get_rag_context(self, issue_domain: str, issue_description: str) -> str:
        """
        Retrieve relevant legal context from RAG system.
        
        Args:
            issue_domain: Domain of issue
            issue_description: Description of issue
            
        Returns:
            Formatted RAG context with relevant laws and principles
        """
        try:
            # Use cached VectorStore instance
            if self._rag_store is None:
                return "RAG system unavailable. Proceed with general AI ethics principles."
            
            rag = self._rag_store
            
            # Query for relevant legal principles
            query = f"{issue_domain} AI ethics law {issue_description}"
            results = rag.search(
                domain=issue_domain,  # Try domain-specific collection
                query_text=query,
                n_results=3
            )
            
            if not results or not results.get('documents'):
                # Try general collection
                results = rag.search(
                    domain="general",
                    query_text=query,
                    n_results=3
                )
            
            if not results or not results.get('documents'):
                return "No specific legal precedents found in knowledge base."
            
            # Format results
            context_parts = []
            documents = results.get('documents', [])
            metadatas = results.get('metadatas', [{}] * len(documents))
            
            for i, (doc, metadata) in enumerate(zip(documents, metadatas), 1):
                source = metadata.get('source', 'Unknown')
                domain = metadata.get('domain', issue_domain)
                context_parts.append(
                    f"{i}. {source} ({domain}):\n"
                    f"   \"{doc[:300]}...\""
                )
            
            if not context_parts:
                return "No specific legal precedents found in knowledge base."
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return "RAG system unavailable. Proceed with general AI ethics principles."
    
    def _severity_to_label(self, severity: float) -> str:
        """Convert severity score to descriptive label."""
        if severity >= 0.9:
            return "CRITICAL"
        elif severity >= 0.75:
            return "HIGH"
        elif severity >= 0.5:
            return "MODERATE"
        else:
            return "LOW"
    
    def _format_case_studies(self, case_studies: List[str]) -> str:
        """Format case studies for prompt."""
        if not case_studies:
            return "• No recent cases available"
        
        formatted = []
        for i, case in enumerate(case_studies[:3], 1):  # Top 3
            formatted.append(f"{i}. {case}")
        
        return "\n".join(formatted)
    
    def _format_existing_laws(self, existing_laws: List[ExistingLaw]) -> str:
        """Format existing laws for prompt."""
        if not existing_laws:
            return "No directly related regulations found. This represents a novel regulatory area requiring comprehensive legislation."
        
        formatted = ["Relevant Existing Regulations:"]
        for law in existing_laws[:4]:  # Top 4
            formatted.append(
                f"\n• {law.name}:\n"
                f"  Purpose: {law.description[:150]}...\n"
                f"  Relevance: Consider alignment with this framework"
            )
        
        return "\n".join(formatted)
    
    def _generate_with_template(
        self,
        issue_domain: str,
        issue_description: str,
        severity: float,
        latest_info: Dict,
        coverage_gaps: List[str]
    ) -> str:
        """
        Generate law text using template (fallback).
        
        Args:
            issue_domain: Domain of issue
            issue_description: Issue description
            severity: Severity level
            latest_info: Latest information
            coverage_gaps: Coverage gaps
            
        Returns:
            Template-based law text
        """
        template = self.law_templates.get(issue_domain, self.law_templates["general"])
        
        # Sanitise the issue description so raw signal content doesn't leak
        # into the formal law text — keep only the first sentence, max 200 chars
        clean_desc = issue_description.strip().split('\n')[0][:200].strip()
        if not clean_desc.endswith('.'):
            clean_desc = clean_desc.rsplit(' ', 1)[0] + '.'

        # Fill template
        law_text = template.format(
            issue_description=clean_desc,
            severity_level="CRITICAL" if severity >= 0.8 else "HIGH" if severity >= 0.6 else "MODERATE",
            sources=", ".join(latest_info.get("sources", ["multiple sources"])),
            gaps="; ".join(coverage_gaps[:3]) if coverage_gaps else "regulatory vacuum"
        )
        
        logger.info("✓ Template-based law text generated")
        return law_text
    
    def _generate_rationale(
        self,
        issue_description: str,
        severity: float,
        coverage_gaps: List[str],
        latest_info: Dict
    ) -> str:
        """
        Generate rationale for why this law is needed.
        
        Returns:
            Rationale text
        """
        rationale_parts = [
            f"This regulation is needed to address: {issue_description}",
            f"\nSeverity Level: {'CRITICAL' if severity >= 0.8 else 'HIGH' if severity >= 0.6 else 'MODERATE'} ({severity:.2f}/1.0)",
        ]
        
        if coverage_gaps:
            rationale_parts.append(f"\nCoverage Gaps: {', '.join(coverage_gaps[:3])}")
        
        if latest_info.get("case_studies"):
            rationale_parts.append(f"\nRecent Cases: {len(latest_info['case_studies'])} documented incidents from {', '.join(latest_info['sources'])}")
        
        return "\n".join(rationale_parts)
    
    def _determine_scope(self, issue_domain: str, severity: float) -> str:
        """
        Determine scope of the law.
        
        Returns:
            Scope description
        """
        scopes = {
            "privacy": "All AI systems that process, store, or analyze personal data",
            "bias": "All AI systems involved in decision-making affecting individuals (hiring, lending, criminal justice)",
            "transparency": "All AI systems deployed in public-facing applications or critical infrastructure",
            "accountability": "All organizations developing, deploying, or operating AI systems",
            "safety": "All AI systems with potential physical or psychological harm risks",
            "security": "All AI systems and their supporting infrastructure, APIs, and data pipelines",
            "general": "All AI systems operating within the jurisdiction"
        }
        
        base_scope = scopes.get(issue_domain, scopes["general"])
        
        if severity >= 0.8:
            base_scope += " (CRITICAL: Mandatory immediate compliance)"
        elif severity >= 0.6:
            base_scope += " (HIGH: 6-month compliance timeline)"
        
        return base_scope
    
    def _determine_enforcement(self, issue_domain: str, severity: float) -> str:
        """
        Determine enforcement mechanism.
        
        Returns:
            Enforcement description
        """
        if severity >= 0.8:
            return (
                "Enforcement by designated regulatory authority with:\n"
                "- Mandatory audits every 6 months\n"
                "- Fines up to 4% of annual global revenue for violations\n"
                "- Potential suspension of AI system operations\n"
                "- Criminal liability for egregious violations"
            )
        elif severity >= 0.6:
            return (
                "Enforcement by designated regulatory authority with:\n"
                "- Annual compliance reporting required\n"
                "- Fines up to 2% of annual revenue for violations\n"
                "- Corrective action orders\n"
                "- Public disclosure of violations"
            )
        else:
            return (
                "Enforcement through:\n"
                "- Self-assessment and reporting\n"
                "- Periodic regulatory reviews\n"
                "- Fines for non-compliance\n"
                "- Industry best practice guidelines"
            )
    
    def _load_law_templates(self) -> Dict[str, str]:
        """
        Load law templates for different domains.
        
        Returns:
            Dict of domain -> template
        """
        return {
            "privacy": """
AI PRIVACY PROTECTION ACT

Section 1: Purpose
This Act addresses {issue_description} as identified through analysis of {sources}.

Section 2: Requirements
All AI systems must:
1. Implement privacy-by-design principles
2. Obtain explicit consent for data processing
3. Provide data deletion capabilities
4. Conduct privacy impact assessments

Section 3: Gaps Addressed
This Act specifically fills the following regulatory gaps: {gaps}

Section 4: Severity Classification
This issue is classified as {severity_level} priority.

Section 5: Compliance Timeline
Organizations must achieve full compliance within 12 months of enactment.
""",
            "bias": """
AI FAIRNESS AND NON-DISCRIMINATION ACT

Section 1: Purpose
This Act addresses {issue_description} as documented in {sources}.

Section 2: Requirements
All AI systems used in decision-making must:
1. Undergo bias testing across protected demographic groups
2. Maintain fairness metrics within acceptable thresholds
3. Provide explanations for adverse decisions
4. Enable human review and appeal processes

Section 3: Regulatory Gaps
This Act addresses: {gaps}

Section 4: Severity
Issue severity: {severity_level}

Section 5: Implementation
Phased implementation over 18 months with quarterly reporting.
""",
            "transparency": """
AI TRANSPARENCY AND EXPLAINABILITY ACT

Section 1: Purpose
This Act addresses {issue_description} based on evidence from {sources}.

Section 2: Requirements
All AI systems must:
1. Disclose AI usage to end users
2. Provide meaningful explanations of decisions
3. Document training data sources and limitations
4. Maintain audit trails

Section 3: Coverage Gaps
Addresses: {gaps}

Section 4: Priority Level
{severity_level} priority issue requiring immediate attention.

Section 5: Enforcement
Regulatory oversight with public reporting requirements.
""",
            "general": """
AI COMPLIANCE AND GOVERNANCE ACT

Section 1: Purpose
This Act addresses {issue_description} as identified through {sources}.

Section 2: Requirements
Organizations deploying AI systems must:
1. Establish AI governance frameworks
2. Conduct regular risk assessments
3. Implement appropriate safeguards
4. Maintain compliance documentation

Section 3: Regulatory Need
This Act fills critical gaps: {gaps}

Section 4: Severity Classification
Issue classified as {severity_level} priority.

Section 5: Compliance
Compliance required within timeline determined by severity level.
""",
            "safety": """
AI SAFETY AND HARM PREVENTION ACT

Section 1: Purpose
This Act addresses {issue_description} as documented by {sources}.

Section 2: Pre-Deployment Requirements
All AI systems with potential harm risks must:
1. Complete a mandatory safety impact assessment before deployment
2. Undergo adversarial red-teaming and stress testing
3. Implement fail-safe mechanisms and emergency stop controls
4. Document all identified risks, edge cases, and residual risk levels

Section 3: Operational Safeguards
Deployers of covered AI systems shall:
1. Maintain meaningful human oversight for high-stakes decisions
2. Implement continuous monitoring for anomalous or unsafe behaviour
3. Deploy gradual rollout strategies with automated incident detection
4. Establish and publish clear criteria for emergency system shutdown

Section 4: Regulatory Gaps
This Act addresses: {gaps}

Section 5: Severity Classification
Issue classified as {severity_level} priority.

Section 6: Enforcement
Violations subject to fines up to 4% of annual global revenue, mandatory
safety audits, and potential suspension of AI system operations.
""",
            "security": """
AI CYBERSECURITY AND RESILIENCE ACT

Section 1: Purpose
This Act addresses {issue_description} as identified through {sources}.

Section 2: Security-by-Design Obligations
All AI systems must be designed and maintained with:
1. Protection against adversarial attacks (data poisoning, evasion, model extraction)
2. Encrypted storage and transmission of training data, model weights, and inference data
3. Role-based access control and multi-factor authentication for model APIs
4. Comprehensive audit logging of all data access and model invocations

Section 3: Supply Chain and Operational Security
Organizations shall:
1. Verify provenance and integrity of third-party models, datasets, and frameworks
2. Perform vulnerability assessments and penetration testing at least annually
3. Maintain incident detection, response, and recovery plans for AI-specific threats
4. Ensure runtime monitoring against model drift, anomalous outputs, and exfiltration

Section 4: Coverage Gaps
This Act addresses: {gaps}

Section 5: Severity Classification
Issue classified as {severity_level} priority.

Section 6: Enforcement
Violations subject to fines up to EUR 10,000,000 or 2% of annual turnover, mandatory
remediation timelines, and public disclosure of material security incidents.
""",
            "accountability": """
AI ACCOUNTABILITY AND GOVERNANCE ACT

Section 1: Purpose
This Act addresses {issue_description} as documented by {sources}.

Section 2: Governance Structure
Organizations developing or deploying AI systems must:
1. Designate an AI System Owner accountable for each production system
2. Establish an independent AI ethics review committee
3. Define clear roles and responsibilities across the AI lifecycle
4. Maintain liability allocation and insurance for AI-caused harms

Section 3: Audit and Transparency
Covered entities shall:
1. Maintain comprehensive audit trails of AI decision-making processes
2. Conduct independent ethics and compliance audits at least annually
3. Publish plain-language accountability reports for high-risk AI
4. Implement whistleblower protections for AI ethics concerns

Section 4: Incident Response
Upon identification of AI-caused harm:
1. Immediate notification to affected individuals and the regulatory authority
2. Root-cause analysis completed within 30 days
3. Remediation plan with measurable milestones
4. Public disclosure of material incidents within 60 days

Section 5: Regulatory Gaps
This Act addresses: {gaps}

Section 6: Severity Classification
Issue classified as {severity_level} priority.

Section 7: Enforcement
Fines up to 3% of annual revenue, mandatory governance improvement orders,
and potential personal liability for senior leadership.
"""
        }
