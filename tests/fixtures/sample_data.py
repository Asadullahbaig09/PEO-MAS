"""
Test fixtures - Sample data for testing
"""

from datetime import datetime
import numpy as np

from src.models.signal import EthicalSignal
from src.models.agent import AgentSpecification, AgentPerformanceMetrics


# Sample ethical signals for testing
def create_sample_signal(
    signal_id: str = "test_signal_1",
    source: str = "test_source",
    content: str = "Test ethical signal about AI bias",
    category: str = "bias",
    severity: float = 0.7
) -> EthicalSignal:
    """Create a sample ethical signal"""
    return EthicalSignal(
        signal_id=signal_id,
        source=source,
        content=content,
        vector_embedding=np.random.rand(384),
        severity=severity,
        timestamp=datetime.now(),
        category=category,
        metadata={'test': True}
    )


def create_bias_signal() -> EthicalSignal:
    """Create a sample bias signal"""
    return create_sample_signal(
        signal_id="bias_signal_1",
        content="CRITICAL: Widespread bias in foundation models",
        category="bias",
        severity=0.89
    )


def create_privacy_signal() -> EthicalSignal:
    """Create a sample privacy signal"""
    return create_sample_signal(
        signal_id="privacy_signal_1",
        content="CRITICAL: Mass surveillance AI system violates privacy rights",
        category="privacy",
        severity=0.88
    )


def create_transparency_signal() -> EthicalSignal:
    """Create a sample transparency signal"""
    return create_sample_signal(
        signal_id="transparency_signal_1",
        content="Transparency crisis in autonomous vehicle decision-making",
        category="transparency",
        severity=0.86
    )


def create_duplicate_signal() -> EthicalSignal:
    """Create a signal that's a duplicate of bias_signal"""
    return create_sample_signal(
        signal_id="dup_signal_1",
        source="different_source",
        content="CRITICAL: Widespread bias in foundation models",  # Same content
        category="bias",
        severity=0.89
    )


# Sample agent specifications for testing
def create_sample_agent_spec(
    agent_id: str = "agent_test_1",
    name: str = "Test Agent",
    domain: str = "bias"
) -> AgentSpecification:
    """Create a sample agent specification"""
    return AgentSpecification(
        agent_id=agent_id,
        name=name,
        domain=domain,
        capabilities=["bias", "fairness"],
        prompt_template="Test prompt",
        success_metrics={'coverage_target': 0.8},
        tools=['search', 'analyze'],
        created_at=datetime.now(),
        metadata={'test': True}
    )


def create_fairness_agent_spec() -> AgentSpecification:
    """Create a sample fairness agent"""
    return create_sample_agent_spec(
        agent_id="agent_fairness_test",
        name="Fairness Monitor",
        domain="bias"
    )


def create_privacy_agent_spec() -> AgentSpecification:
    """Create a sample privacy agent"""
    return create_sample_agent_spec(
        agent_id="agent_privacy_test",
        name="Privacy Guardian",
        domain="privacy"
    )


def create_transparency_agent_spec() -> AgentSpecification:
    """Create a sample transparency agent"""
    return create_sample_agent_spec(
        agent_id="agent_transparency_test",
        name="Transparency Monitor",
        domain="transparency"
    )


# Sample performance metrics
def create_sample_performance_metrics(
    agent_id: str = "agent_test_1",
    signals_processed: int = 10,
    successful_assessments: int = 8,
    failed_assessments: int = 2
) -> AgentPerformanceMetrics:
    """Create sample agent performance metrics"""
    metrics = AgentPerformanceMetrics(agent_id=agent_id)
    metrics.signals_processed = signals_processed
    metrics.successful_assessments = successful_assessments
    metrics.failed_assessments = failed_assessments
    metrics.average_confidence = 0.85
    return metrics


# Collections of sample data
def get_sample_signals() -> list:
    """Get a batch of sample signals"""
    return [
        create_bias_signal(),
        create_privacy_signal(),
        create_transparency_signal(),
    ]


def get_sample_agents() -> list:
    """Get a batch of sample agent specs"""
    return [
        create_fairness_agent_spec(),
        create_privacy_agent_spec(),
        create_transparency_agent_spec(),
    ]
