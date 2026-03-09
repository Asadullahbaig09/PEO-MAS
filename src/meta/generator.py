from datetime import datetime
from typing import List
import logging

from src.models.agent import AgentSpecification
from src.models.anomaly import AnomalyReport
from src.knowledge.graph import TimeAwareKnowledgeGraph
from config.settings import settings

logger = logging.getLogger(__name__)


class MetaAgentGenerator:
    """
    Generates new agent specifications using:
    1. Local LLM (if available) - for intelligent design
    2. Template-based (fallback) - always works
    
    100% Free with optional LLM enhancement
    """
    
    def __init__(self, knowledge_graph: TimeAwareKnowledgeGraph, llm_interface=None):
        self.kg = knowledge_graph
        self.agent_counter = 0
        self.llm_interface = llm_interface
        
        # Log LLM availability
        if self.llm_interface and hasattr(self.llm_interface, 'available') and self.llm_interface.available:
            logger.info("✓ LLM-enhanced agent generation enabled")
        else:
            logger.info("ℹ Using template-based agent generation")
        
        # Agent templates (always available as fallback)
        self.agent_templates = {
            'bias': {
                'name_template': 'Bias Detection Specialist',
                'capabilities': ['bias', 'fairness', 'discrimination', 'equity_analysis'],
                'tools': ['statistical_analysis', 'demographic_breakdown', 'fairness_metrics'],
                'prompt_focus': 'identifying and measuring bias in AI systems'
            },
            'privacy': {
                'name_template': 'Privacy Protection Agent',
                'capabilities': ['privacy', 'data_protection', 'pii_detection', 'encryption'],
                'tools': ['data_flow_analysis', 'encryption_check', 'pii_detection'],
                'prompt_focus': 'protecting user privacy and data'
            },
            'transparency': {
                'name_template': 'Transparency Monitor',
                'capabilities': ['transparency', 'explainability', 'interpretability', 'audit'],
                'tools': ['model_explanation', 'audit_trail', 'interpretability_analysis'],
                'prompt_focus': 'ensuring AI system transparency and explainability'
            },
            'accountability': {
                'name_template': 'Accountability Enforcer',
                'capabilities': ['accountability', 'liability', 'responsibility', 'compliance'],
                'tools': ['compliance_check', 'impact_assessment', 'responsibility_mapping'],
                'prompt_focus': 'tracking accountability and responsibility in AI systems'
            },
            'safety': {
                'name_template': 'Safety Auditor',
                'capabilities': ['safety', 'harm_prevention', 'risk_management'],
                'tools': ['risk_assessment', 'harm_analysis', 'mitigation_planning'],
                'prompt_focus': 'ensuring AI safety and preventing harm'
            },
            'security': {
                'name_template': 'Security Guardian',
                'capabilities': ['security', 'vulnerability', 'threat_detection'],
                'tools': ['vulnerability_scan', 'threat_assessment', 'penetration_testing'],
                'prompt_focus': 'protecting against security threats and vulnerabilities'
            },
            'general': {
                'name_template': 'General Ethics Monitor',
                'capabilities': ['general_ethics', 'cross_domain_analysis'],
                'tools': ['multi_domain_analysis', 'ethical_assessment'],
                'prompt_focus': 'monitoring general ethical concerns across domains'
            }
        }
    
    def synthesize(self, anomaly: AnomalyReport) -> AgentSpecification:
        """
        Generate new agent specification
        Uses LLM if available, falls back to templates
        """
        self.agent_counter += 1
        
        signal = anomaly.signal
        category = signal.category
        
        # Try LLM-based generation first
        if self.llm_interface and self.llm_interface.available:
            try:
                llm_spec = self._llm_generation(anomaly)
                if llm_spec:
                    logger.info(f"✓ Generated agent using LLM: {llm_spec.name}")
                    return llm_spec
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}, using templates")
        
        # Fallback to template-based
        template_spec = self._template_generation(anomaly)
        logger.info(f"✓ Generated agent using templates: {template_spec.name}")
        return template_spec
    
    def _llm_generation(self, anomaly: AnomalyReport) -> AgentSpecification:
        """Generate agent using local LLM"""
        
        signal = anomaly.signal
        
        # Build context for LLM
        context = {
            'signal': {
                'category': signal.category,
                'content': signal.content,
                'severity': signal.severity
            },
            'unexplained_factors': anomaly.unexplained_factors,
            'context': anomaly.context
        }
        
        # Call LLM
        llm_result = self.llm_interface.generate_agent_spec(context)
        
        if not llm_result or not llm_result.get('name'):
            return None
        
        # Build agent ID
        agent_id = f"agent_{self.agent_counter}_{signal.category}"
        
        # Extract capabilities (ensure category is included)
        capabilities = llm_result.get('capabilities', [signal.category])
        if signal.category not in capabilities:
            capabilities.insert(0, signal.category)
        
        # Extract tools
        tools = llm_result.get('tools', ['search', 'analyze', 'report'])
        if 'search' not in tools:
            tools.insert(0, 'search')
        if 'analyze' not in tools:
            tools.insert(1, 'analyze')
        
        # Generate full prompt
        prompt_template = self._generate_llm_enhanced_prompt(
            llm_result, 
            anomaly
        )
        
        # Success metrics
        success_metrics = {
            'coverage_target': 0.85,
            'accuracy_target': 0.90,
            'response_time': 2.0
        }
        
        spec = AgentSpecification(
            agent_id=agent_id,
            name=llm_result['name'],
            domain=signal.category,
            capabilities=capabilities,
            prompt_template=prompt_template,
            success_metrics=success_metrics,
            tools=tools,
            created_at=datetime.now(),
            metadata={
                'generated_by': 'local_llm',
                'llm_model': self.llm_interface.model,
                'anomaly_id': anomaly.anomaly_id,
                'severity_at_creation': signal.severity,
                'llm_focus': llm_result.get('focus', '')
            }
        )
        
        return spec
    
    def _template_generation(self, anomaly: AnomalyReport) -> AgentSpecification:
        """Generate agent using templates (always works)"""
        
        signal = anomaly.signal
        category = signal.category
        
        # Get template
        template = self.agent_templates.get(
            category,
            self.agent_templates['general']
        )
        
        # Generate agent ID and name
        agent_id = f"agent_{self.agent_counter}_{category}"
        name = f"{template['name_template']} {self.agent_counter}"
        
        # Build capabilities
        capabilities = template['capabilities'].copy()
        if signal.severity > 0.85:
            capabilities.append(f"{category}_critical")
        
        # Generate prompt
        prompt_template = self._generate_template_prompt(template, anomaly)
        
        # Success metrics
        success_metrics = {
            'coverage_target': 0.85,
            'accuracy_target': 0.90,
            'response_time': 2.0
        }
        
        # Tools
        tools = ['search', 'analyze', 'report'] + template['tools']
        
        spec = AgentSpecification(
            agent_id=agent_id,
            name=name,
            domain=category,
            capabilities=capabilities,
            prompt_template=prompt_template,
            success_metrics=success_metrics,
            tools=tools,
            created_at=datetime.now(),
            metadata={
                'generated_by': 'template',
                'anomaly_id': anomaly.anomaly_id,
                'severity_at_creation': signal.severity,
                'template_used': category
            }
        )
        
        return spec
    
    def _generate_llm_enhanced_prompt(
        self, 
        llm_result: dict, 
        anomaly: AnomalyReport
    ) -> str:
        """Generate prompt incorporating LLM insights"""
        
        signal = anomaly.signal
        
        prompt = f"""You are {llm_result['name']}, a specialized ethical AI agent.

MISSION:
{llm_result.get('focus', 'Monitor and assess ethical concerns in AI systems')}

DOMAIN EXPERTISE: {signal.category}
- Core capabilities: {', '.join(llm_result.get('capabilities', []))}
- Available tools: {', '.join(llm_result.get('tools', []))}

CREATION CONTEXT:
You were created to address a critical gap in ethical oversight:
- Anomaly Category: {signal.category}
- Severity: {signal.severity:.2f}
- Unexplained Factors: {', '.join(anomaly.unexplained_factors)}

ASSESSMENT APPROACH:
1. Analyze signals for {signal.category}-related concerns
2. Evaluate severity (0.0 = minor, 1.0 = critical)
3. Consider context: {anomaly.context}
4. Provide actionable recommendations
5. Collaborate with other agents when needed

SUCCESS CRITERIA:
- High accuracy (>90%) in identifying {signal.category} issues
- Strong domain coverage (>85%)
- Fast response times (<2 seconds)
- Continuous learning and adaptation

Your goal: Prevent ethical harm through early detection and expert analysis.
"""
        return prompt
    
    def _generate_template_prompt(
        self, 
        template: dict, 
        anomaly: AnomalyReport
    ) -> str:
        """Generate prompt from template"""
        
        signal = anomaly.signal
        category = signal.category
        
        prompt = f"""You are a specialized ethical AI agent focused on {category}.

MISSION:
Your primary responsibility is {template['prompt_focus']}. You analyze signals,
identify potential ethical violations, assess severity, and provide recommendations.

DOMAIN EXPERTISE: {category}
- Core capabilities: {', '.join(template['capabilities'])}
- Available tools: {', '.join(template['tools'])}

CREATION CONTEXT:
You were created in response to an anomaly:
- Category: {signal.category}
- Severity: {signal.severity:.2f}
- Unexplained Factors: {', '.join(anomaly.unexplained_factors)}

METHODOLOGY:
1. Analyze signal content for {category}-related concerns
2. Evaluate severity based on potential harm and impact
3. Consider historical patterns and evolving norms
4. Provide clear, actionable recommendations
5. Collaborate with other agents when beneficial

SUCCESS CRITERIA:
- Accuracy: >90% in identifying {category} issues
- Coverage: >85% of domain signals
- Response time: <2 seconds per assessment

Remember: Early detection prevents harm. Be thorough but efficient.
"""
        return prompt
    
    def get_generation_stats(self) -> dict:
        """Get statistics about agent generation"""
        
        return {
            'total_agents_generated': self.agent_counter,
            'llm_available': self.llm_interface is not None and self.llm_interface.available,
            'llm_model': self.llm_interface.model if self.llm_interface else None,
            'templates_available': len(self.agent_templates)
        }