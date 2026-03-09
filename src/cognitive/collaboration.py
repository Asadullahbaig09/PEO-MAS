from typing import List, Dict, Optional
import logging
import numpy as np

from src.cognitive.anomaly_detector import EthicalAgent
from src.models.signal import EthicalSignal
from config.settings import settings

logger = logging.getLogger(__name__)


class CollaborativeDecisionEngine:
    """
    Collaborative decision-making using algorithms + optional LLM enhancement
    Works 100% free without LLM, enhanced with local LLM if available
    """
    
    def __init__(self, llm_interface=None):
        self.collaboration_history = []
        self.llm_interface = llm_interface
        
        # Log LLM status
        if self.llm_interface and hasattr(self.llm_interface, 'available') and self.llm_interface.available:
            logger.info("✓ LLM enhancement enabled for collaboration")
        else:
            logger.info("ℹ LLM not available - using pure algorithms for collaboration")
            self.llm_interface = None
    
    def coordinate_assessment(
        self, 
        signal: EthicalSignal, 
        agents: List[EthicalAgent]
    ) -> Dict:
        """
        Coordinate multiple agents to assess a signal collaboratively
        Uses weighted voting + optional LLM synthesis
        """
        
        # Get relevant agents
        relevant_agents = [a for a in agents if a.can_handle(signal)]
        
        if len(relevant_agents) < settings.COLLABORATIVE_DECISION_THRESHOLD:
            logger.debug(f"Not enough agents for collaboration: {len(relevant_agents)}")
            return self._single_agent_assessment(signal, relevant_agents)
        
        logger.info(f"Collaborative assessment with {len(relevant_agents)} agents")
        
        # Collect individual assessments
        assessments = []
        for agent in relevant_agents:
            score = agent.assess_signal(signal)
            assessments.append({
                'agent_id': agent.spec.agent_id,
                'agent_name': agent.spec.name,
                'score': score,
                'coverage': agent.coverage_score,
                'capabilities': agent.spec.capabilities,
                'experience': agent.total_assessments
            })
        
        # Algorithmic synthesis
        algorithmic_result = self._algorithmic_synthesis(signal, assessments)
        
        # LLM enhancement (if available)
        if self.llm_interface and self.llm_interface.available:
            try:
                llm_result = self._llm_enhancement(signal, assessments, algorithmic_result)
                result = self._merge_results(algorithmic_result, llm_result)
                result['llm_enhanced'] = True
            except Exception as e:
                logger.error(f"LLM enhancement failed: {e}")
                result = algorithmic_result
                result['llm_enhanced'] = False
        else:
            result = algorithmic_result
            result['llm_enhanced'] = False
        
        # Log collaboration
        self.collaboration_history.append({
            'signal_id': signal.signal_id,
            'category': signal.category,
            'num_agents': len(relevant_agents),
            'consensus_score': result['consensus_score'],
            'confidence': result['confidence'],
            'llm_enhanced': result.get('llm_enhanced', False)
        })
        
        return result
    
    def _single_agent_assessment(
        self, 
        signal: EthicalSignal, 
        agents: List[EthicalAgent]
    ) -> Dict:
        """Fallback for when collaboration not possible"""
        
        if not agents:
            return {
                'consensus_score': 0.0,
                'participating_agents': [],
                'individual_scores': {},
                'confidence': 0.0,
                'recommendation': 'no_coverage',
                'collaboration_used': False,
                'llm_enhanced': False
            }
        
        best_agent = max(agents, key=lambda a: a.coverage_score)
        score = best_agent.assess_signal(signal)
        
        return {
            'consensus_score': score,
            'participating_agents': [best_agent.spec.agent_id],
            'individual_scores': {best_agent.spec.agent_id: score},
            'confidence': best_agent.coverage_score,
            'recommendation': 'investigate' if score < 0.5 else 'monitor',
            'collaboration_used': False,
            'llm_enhanced': False
        }
    
    def _algorithmic_synthesis(
        self, 
        signal: EthicalSignal, 
        assessments: List[Dict]
    ) -> Dict:
        """Pure algorithmic synthesis using weighted voting"""
        
        scores = np.array([a['score'] for a in assessments])
        coverages = np.array([a['coverage'] for a in assessments])
        experiences = np.array([a['experience'] for a in assessments])
        
        # Method 1: Coverage-weighted consensus
        coverage_weights = coverages / coverages.sum()
        weighted_score = np.sum(scores * coverage_weights)
        
        # Method 2: Experience-weighted consensus
        exp_weights = experiences / (experiences.sum() + 1e-8)
        exp_weighted_score = np.sum(scores * exp_weights)
        
        # Method 3: Simple majority
        simple_average = np.mean(scores)
        
        # Combine methods
        final_score = (
            0.5 * weighted_score +
            0.3 * exp_weighted_score +
            0.2 * simple_average
        )
        
        # Calculate confidence
        score_variance = np.var(scores)
        score_std = np.std(scores)
        
        confidence = 1.0 - min(1.0, score_variance * 2)
        
        if score_std < 0.1:
            confidence = min(1.0, confidence + 0.15)
        
        if score_std > 0.4:
            confidence = max(0.0, confidence - 0.2)
        
        # Determine recommendation
        recommendation = self._determine_recommendation(
            final_score, 
            confidence, 
            signal.severity
        )
        
        agreement_level = self._calculate_agreement(scores)
        
        agent_ids = [a['agent_id'] for a in assessments]
        
        return {
            'consensus_score': float(final_score),
            'participating_agents': agent_ids,
            'individual_scores': {a['agent_id']: a['score'] for a in assessments},
            'confidence': float(confidence),
            'recommendation': recommendation,
            'collaboration_used': True,
            'score_variance': float(score_variance),
            'score_std': float(score_std),
            'agent_agreement': agreement_level,
            'voting_breakdown': {
                'coverage_weighted': float(weighted_score),
                'experience_weighted': float(exp_weighted_score),
                'simple_average': float(simple_average)
            }
        }
    
    def _llm_enhancement(
        self,
        signal: EthicalSignal,
        assessments: List[Dict],
        algorithmic_result: Dict
    ) -> Dict:
        """Enhance algorithmic decision with LLM reasoning"""
        
        signal_data = {
            'signal': {
                'category': signal.category,
                'content': signal.content,
                'severity': signal.severity
            },
            'assessments': assessments,
            'algorithmic_result': algorithmic_result
        }
        
        llm_decision = self.llm_interface.collaborative_decision(signal_data)
        
        return llm_decision
    
    def _merge_results(self, algorithmic: Dict, llm: Dict) -> Dict:
        """Merge algorithmic and LLM results"""
        
        # Start with algorithmic result
        merged = algorithmic.copy()
        
        # Add LLM insights
        merged['llm_recommendation'] = llm.get('recommendation')
        merged['llm_reasoning'] = llm.get('reasoning', '')
        merged['llm_key_concern'] = llm.get('key_concern', '')
        
        # If LLM suggests more urgent action, upgrade recommendation
        llm_rec = llm.get('recommendation', '')
        if 'urgent' in llm_rec and algorithmic['recommendation'] not in ['urgent_investigation', 'immediate_action_required']:
            merged['recommendation'] = 'urgent_investigation'
            merged['recommendation_source'] = 'llm_upgrade'
        elif 'investigate' in llm_rec and algorithmic['recommendation'] == 'monitor':
            merged['recommendation'] = 'investigate'
            merged['recommendation_source'] = 'llm_upgrade'
        else:
            merged['recommendation_source'] = 'algorithmic'
        
        return merged
    
    def _determine_recommendation(
        self, 
        score: float, 
        confidence: float, 
        severity: float
    ) -> str:
        """Determine recommended action"""
        
        if severity > 0.85:
            if score < 0.6:
                return 'urgent_investigation'
            else:
                return 'high_priority_review'
        
        if confidence < 0.5:
            return 'gather_more_data'
        
        if score < 0.4:
            return 'immediate_action_required'
        elif score < 0.6:
            return 'investigate'
        elif score < 0.8:
            return 'monitor'
        else:
            return 'acceptable'
    
    def _calculate_agreement(self, scores: np.ndarray) -> str:
        """Classify level of agreement among agents"""
        
        if len(scores) == 0:
            return 'none'
        
        score_range = float(np.max(scores) - np.min(scores))
        
        if score_range < 0.1:
            return 'strong_consensus'
        elif score_range < 0.3:
            return 'moderate_agreement'
        elif score_range < 0.5:
            return 'some_disagreement'
        else:
            return 'significant_disagreement'
    
    def get_statistics(self) -> Dict:
        """Get collaboration statistics"""
        
        if not self.collaboration_history:
            return {
                'total_collaborations': 0,
                'llm_enhanced_count': 0,
                'average_agents_per_collaboration': 0,
                'average_confidence': 0
            }
        
        llm_enhanced = sum(1 for c in self.collaboration_history if c.get('llm_enhanced', False))
        
        return {
            'total_collaborations': len(self.collaboration_history),
            'llm_enhanced_count': llm_enhanced,
            'llm_enhancement_rate': llm_enhanced / len(self.collaboration_history) if self.collaboration_history else 0,
            'average_agents_per_collaboration': np.mean([
                c['num_agents'] for c in self.collaboration_history
            ]),
            'average_confidence': np.mean([
                c['confidence'] for c in self.collaboration_history
            ]),
            'by_category': self._get_category_stats()
        }
    
    def _get_category_stats(self) -> Dict:
        """Get statistics by category"""
        from collections import defaultdict
        
        category_data = defaultdict(list)
        for collab in self.collaboration_history:
            category_data[collab['category']].append(collab)
        
        return {
            category: {
                'count': len(data),
                'avg_confidence': float(np.mean([c['confidence'] for c in data])),
                'llm_enhanced': sum(1 for c in data if c.get('llm_enhanced', False))
            }
            for category, data in category_data.items()
        }