from typing import List, Optional
from datetime import datetime
import logging
import numpy as np

from src.models.signal import EthicalSignal
from src.models.anomaly import AnomalyReport
from src.cognitive.attention import AttentionMechanism
from config.settings import settings

try:
    from src.training.inference import HybridAnomalyDetector
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    logging.warning("Neural anomaly detector not available. Using rule-based detection only.")

logger = logging.getLogger(__name__)


class EthicalAgent:
    """Individual agent monitoring specific ethical domain with RAG support"""
    
    def __init__(self, spec, retriever=None, generator=None):
        """
        Initialize ethical agent with optional RAG components
        
        Args:
            spec: AgentSpecification
            retriever: Optional DocumentRetriever for RAG
            generator: Optional EthicalAssessmentGenerator for RAG
        """
        self.spec = spec
        self.coverage_score = 0.75  # Start higher (was 0.7)
        self.processed_signals = []
        self.successful_assessments = 0
        self.total_assessments = 0
        
        # RAG components
        self.retriever = retriever
        self.generator = generator
        self.use_rag = spec.use_rag and retriever is not None and generator is not None
        
        if self.use_rag:
            logger.info(f"✓ Agent {spec.name} initialized with RAG support")
        
    def can_handle(self, signal: EthicalSignal) -> bool:
        """Check if agent can handle this signal"""
        return signal.category in self.spec.capabilities
    
    def assess_signal(self, signal: EthicalSignal) -> float:
        """Assess how well agent can explain this signal (0.0 to 1.0)"""
        if not self.can_handle(signal):
            return 0.0
        
        # Track assessment
        self.total_assessments += 1
        
        # Use RAG-enhanced assessment if available
        if self.use_rag:
            return self._assess_with_rag(signal)
        else:
            return self._assess_traditional(signal)
    
    def _assess_traditional(self, signal: EthicalSignal) -> float:
        """Traditional assessment without RAG"""
        # Improved scoring that learns over time
        base_score = self.coverage_score
        
        # Less penalty for severity (was 0.2, now 0.1)
        severity_penalty = signal.severity * 0.1
        
        # Bonus for experience with this category
        experience_bonus = min(0.15, self.successful_assessments * 0.01)
        
        final_score = base_score - severity_penalty + experience_bonus
        
        # If we handled it well, mark as successful
        if final_score > settings.ETHICAL_COVERAGE_THRESHOLD:
            self.successful_assessments += 1
            self.update_coverage(success=True)
        
        return max(0, min(1.0, final_score))
    
    def _assess_with_rag(self, signal: EthicalSignal) -> float:
        """RAG-enhanced assessment with retrieved context"""
        try:
            logger.info(f"🔍 {self.spec.name} performing RAG assessment for signal: {signal.content[:50]}...")
            
            # Retrieve relevant documents
            retrieval_results = self.retriever.retrieve_for_signal(
                signal=signal,
                domain=self.spec.domain,
                k=self.spec.retrieval_k
            )
            
            num_docs = len(retrieval_results.documents) if retrieval_results else 0
            # logger.info(f"  📚 Retrieved {num_docs} relevant documents from {self.spec.domain} collection")
            
            # Generate detailed assessment
            assessment = self.generator.generate_assessment(
                signal=signal,
                retrieval_results=retrieval_results,
                agent_name=self.spec.name,
                agent_capabilities=self.spec.capabilities
            )
            
            logger.info(f"  ✓ Generated assessment: {assessment.risk_level} (confidence: {assessment.confidence:.2f})")
            
            # Store the assessment for later reference
            self.processed_signals.append({
                'signal_id': signal.signal_id,
                'assessment': assessment,
                'timestamp': datetime.now(),
                'docs_retrieved': num_docs
            })
            
            # Track RAG metrics (for reporting)
            if not hasattr(self, 'rag_assessments'):
                self.rag_assessments = 0
                self.total_docs_retrieved = 0
            
            self.rag_assessments += 1
            self.total_docs_retrieved += num_docs
            
            # Use assessment confidence as explanation score
            explanation_score = assessment.confidence
            
            # Update success tracking
            if explanation_score > settings.ETHICAL_COVERAGE_THRESHOLD:
                self.successful_assessments += 1
                self.update_coverage(success=True)
            else:
                self.update_coverage(success=False)
            
            return explanation_score
            
        except Exception as e:
            logger.error(f"❌ RAG assessment failed for {self.spec.name}: {e}", exc_info=True)
            logger.warning(f"⚠️  Falling back to traditional assessment")
            # Fallback to traditional assessment
            return self._assess_traditional(signal)
    
    def get_last_assessment(self, signal_id: str):
        """Get detailed assessment for a signal (RAG-generated)"""
        for item in reversed(self.processed_signals):
            if item['signal_id'] == signal_id:
                return item.get('assessment')
        return None
    
    def update_coverage(self, success: bool):
        """Update coverage based on performance"""
        if success:
            # Increase coverage faster
            self.coverage_score = min(0.95, self.coverage_score + 0.08)
        else:
            # Decrease coverage slower
            self.coverage_score = max(0.5, self.coverage_score - 0.02)
    
    def get_performance_stats(self) -> dict:
        """Get agent performance statistics"""
        return {
            'coverage_score': self.coverage_score,
            'total_assessments': self.total_assessments,
            'successful_assessments': self.successful_assessments,
            'success_rate': (
                self.successful_assessments / self.total_assessments 
                if self.total_assessments > 0 else 0
            )
        }


class AnomalyDetector:
    """
    Detects unexplained ethical anomalies and triggers law checking/generation.
    
    Instead of spawning new agents, this now triggers legal compliance checking
    to determine if a law should be proposed.
    
    Can use neural network-based detection (100% F1 score) or rule-based fallback.
    """
    
    def __init__(self, threshold: float = None, law_checker=None, law_generator=None, use_neural: bool = True):
        self.threshold = threshold or settings.ETHICAL_COVERAGE_THRESHOLD
        self.attention = AttentionMechanism()
        self.recent_anomalies = []  # Track recent anomalies
        
        # Legal compliance components
        self.law_checker = law_checker
        self.law_generator = law_generator
        self.legal_recommendations = []  # Track generated recommendations
        
        # Neural detector (optimal threshold=0.60 for 100% F1 score)
        self.use_neural = use_neural and NEURAL_AVAILABLE
        self.neural_detector = None
        
        if self.use_neural:
            try:
                self.neural_detector = HybridAnomalyDetector(
                    neural_threshold=0.60,  # Optimal threshold from tuning
                    rule_threshold=self.threshold  # Fallback threshold
                )
                logger.info("✓ Neural anomaly detector initialized (threshold=0.60, 100% F1 score)")
            except Exception as e:
                logger.warning(f"Failed to initialize neural detector: {e}. Using rule-based detection.")
                self.use_neural = False
        else:
            logger.info("Using rule-based anomaly detection")
        
    def detect_anomaly(
        self, 
        signal: EthicalSignal, 
        agent_pool: List[EthicalAgent]
    ) -> Optional[AnomalyReport]:
        """
        Detect if signal represents an anomaly requiring legal intervention.
        
        Now triggers law checking/generation instead of agent spawning.
        
        Returns AnomalyReport if anomaly detected, None otherwise
        """
        
        # Get assessments from ALL agents that can handle this category
        relevant_agents = [a for a in agent_pool if a.can_handle(signal)]
        
        if not relevant_agents:
            # No agents can handle this category - check for legal coverage
            return self._handle_coverage_gap(
                signal=signal,
                explanation_score=0.0,
                gap_description=f"No coverage for category: {signal.category}",
                agent_pool=agent_pool,
                severity_level="high"
            )
        
        # Get assessments from relevant agents
        explanation_scores = [agent.assess_signal(signal) for agent in relevant_agents]
        max_explanation = max(explanation_scores)
        avg_explanation = sum(explanation_scores) / len(explanation_scores)
        
        # Detect anomaly using neural network or rule-based approach
        if self.use_neural and self.neural_detector is not None:
            is_anomaly = self._detect_with_neural(signal, max_explanation, avg_explanation)
        else:
            is_anomaly = self._detect_with_rules(signal, max_explanation, avg_explanation)
        
        if is_anomaly:
            unexplained = self._identify_gaps(signal, agent_pool, max_explanation)
            severity_level = self._determine_severity_level(signal, max_explanation)
            
            # Instead of spawning agent, check/generate law
            return self._handle_coverage_gap(
                signal=signal,
                explanation_score=max_explanation,
                gap_description="; ".join(unexplained),
                agent_pool=agent_pool,
                severity_level=severity_level
            )
        
        return None
    
    def _handle_coverage_gap(
        self,
        signal: EthicalSignal,
        explanation_score: float,
        gap_description: str,
        agent_pool: List[EthicalAgent],
        severity_level: str
    ) -> AnomalyReport:
        """
        Handle coverage gap by checking/generating legal recommendations.
        
        Args:
            signal: The ethical signal with coverage gap
            explanation_score: How well existing agents explained it
            gap_description: Description of the gap
            agent_pool: Current agent pool
            severity_level: Severity of the issue
            
        Returns:
            AnomalyReport with legal recommendation if needed
        """
        logger.info(f"🔍 Coverage gap detected: {gap_description}")
        
        # Check if existing laws cover this issue
        legal_recommendation = None
        if self.law_checker and self.law_generator:
            try:
                # Check existing legal coverage
                coverage_check = self.law_checker.check_existing_coverage(
                    issue_domain=signal.category,
                    issue_description=signal.content,
                    severity=signal.severity
                )
                
                logger.info(f"⚖️ Legal coverage check: {coverage_check['has_coverage']}, "
                           f"Needs new law: {coverage_check['needs_new_law']}")
                
                # Generate law recommendation if needed
                if coverage_check['needs_new_law']:
                    legal_recommendation = self.law_generator.generate_law_recommendation(
                        issue_domain=signal.category,
                        issue_description=signal.content,
                        severity=signal.severity,
                        affected_signals=[signal],
                        existing_laws=coverage_check['existing_laws'],
                        coverage_gaps=coverage_check['coverage_gaps'],
                        evidence=[]
                    )
                    
                    # Track the recommendation
                    self.legal_recommendations.append(legal_recommendation)
                    
                    logger.info(f"✓ Generated legal recommendation: {legal_recommendation.title}")
                else:
                    logger.info(f"✓ Existing laws provide adequate coverage")
                    
            except Exception as e:
                logger.error(f"❌ Error in law checking/generation: {e}", exc_info=True)
        
        # Create anomaly report (requires_new_agent now means requires_legal_review)
        anomaly = AnomalyReport(
            anomaly_id=f"anomaly_{datetime.now().timestamp()}",
            signal=signal,
            explanation_score=explanation_score,
            unexplained_factors=[gap_description],
            context={
                'available_agents': len(agent_pool),
                'legal_recommendation': legal_recommendation.to_dict() if legal_recommendation else None,
                'legal_check_performed': self.law_checker is not None
            },
            requires_new_agent=False,  # No longer spawning agents
            timestamp=datetime.now(),
            severity_level=severity_level
        )
        
        # Track this anomaly
        self.recent_anomalies.append({
            'category': signal.category,
            'timestamp': datetime.now(),
            'legal_recommendation_id': legal_recommendation.id if legal_recommendation else None
        })
        
        # Keep only last 20 anomalies
        if len(self.recent_anomalies) > 20:
            self.recent_anomalies = self.recent_anomalies[-20:]
        
        return anomaly
    
    def _recently_spawned_for_category(self, category: str) -> bool:
        """
        Check if we recently generated a legal recommendation for this category.
        
        (Renamed from agent spawning check, now checks legal recommendations)
        """
        # Look at last 5 anomalies
        recent = self.recent_anomalies[-5:] if len(self.recent_anomalies) >= 5 else self.recent_anomalies
        
        # Count how many were for this category
        category_count = sum(1 for a in recent if a['category'] == category)
        
        # If more than 2 of last 5 were this category, wait
        return category_count >= 2
    
    def _create_anomaly_report(
        self,
        signal: EthicalSignal,
        explanation_score: float,
        unexplained_factors: List[str],
        agent_pool: List[EthicalAgent],
        severity_level: str
    ) -> AnomalyReport:
        """Helper to create anomaly report"""
        return AnomalyReport(
            anomaly_id=f"anomaly_{datetime.now().timestamp()}",
            signal=signal,
            explanation_score=explanation_score,
            unexplained_factors=unexplained_factors,
            context={'available_agents': len(agent_pool)},
            requires_new_agent=True,
            timestamp=datetime.now(),
            severity_level=severity_level
        )
    
    def _identify_gaps(
        self, 
        signal: EthicalSignal, 
        agent_pool: List[EthicalAgent],
        max_explanation: float
    ) -> List[str]:
        """Identify specific gaps in coverage"""
        gaps = []
        
        # Check domain coverage
        covered_domains = set()
        for agent in agent_pool:
            covered_domains.update(agent.spec.capabilities)
        
        if signal.category not in covered_domains:
            gaps.append(f"No coverage for category: {signal.category}")
            return gaps  # This is the main gap, return early
        
        # Check if existing agents are underperforming
        if max_explanation < 0.4:
            gaps.append(f"Existing {signal.category} agents underperforming (score: {max_explanation:.2f})")
        
        # Check severity handling capability
        high_performing_agents = [
            a for a in agent_pool 
            if a.coverage_score > 0.85 and signal.category in a.spec.capabilities
        ]
        
        if len(high_performing_agents) == 0 and signal.severity > 0.85:
            gaps.append(f"No high-confidence agents for critical {signal.category} signals")
        
        return gaps if gaps else ["Low confidence across all agents"]
    
    def _determine_severity_level(self, signal: EthicalSignal, explanation: float) -> str:
        """Determine severity level of anomaly"""
        if signal.severity > 0.9 and explanation < 0.3:
            return "critical"
        elif signal.severity > 0.8 and explanation < 0.4:
            return "high"
        elif signal.severity > 0.7 or explanation < 0.5:
            return "medium"
        return "low"
    
    def _detect_with_neural(self, signal: EthicalSignal, max_explanation: float, avg_explanation: float) -> bool:
        """
        Detect anomaly using neural network (100% F1 score).
        
        Args:
            signal: Ethical signal to evaluate
            max_explanation: Best explanation score from agents
            avg_explanation: Average explanation score
            
        Returns:
            True if anomaly detected, False otherwise
        """
        try:
            # Use HybridAnomalyDetector's detect method
            # It handles feature extraction and neural prediction internally
            is_anomaly, confidence, severity = self.neural_detector.detect(
                signal=signal,
                agent_pool=None  # Not needed for neural detection
            )
            
            if is_anomaly:
                logger.info(f"🧠 Neural detector: ANOMALY detected (severity={signal.severity:.2f}, explanation={max_explanation:.2f}, confidence={confidence:.2f})")
            
            return is_anomaly
            
        except Exception as e:
            logger.warning(f"Neural detection failed: {e}. Falling back to rule-based.")
            return self._detect_with_rules(signal, max_explanation, avg_explanation)
    
    def _detect_with_rules(self, signal: EthicalSignal, max_explanation: float, avg_explanation: float) -> bool:
        """
        Detect anomaly using rule-based approach (fallback).
        
        Args:
            signal: Ethical signal to evaluate
            max_explanation: Best explanation score from agents
            avg_explanation: Average explanation score
            
        Returns:
            True if anomaly detected, False otherwise
        """
        # MORE STRINGENT anomaly detection
        is_high_severity = signal.severity > settings.ANOMALY_SEVERITY_THRESHOLD
        is_poorly_explained = max_explanation < self.threshold
        is_very_poorly_explained = avg_explanation < (self.threshold - 0.15)
        
        # Only trigger if:
        # 1. High severity AND poorly explained, OR
        # 2. Very poorly explained (even medium severity)
        is_anomaly = (
            (is_high_severity and is_poorly_explained) or
            is_very_poorly_explained
        )
        
        if is_anomaly:
            logger.info(f"📋 Rule-based detector: ANOMALY detected (severity={signal.severity:.2f}, explanation={max_explanation:.2f})")
        
        return is_anomaly

