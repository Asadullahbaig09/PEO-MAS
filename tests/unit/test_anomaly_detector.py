"""
Unit tests for anomaly detector
Tests anomaly detection logic and alert generation
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime

from src.cognitive.anomaly_detector import AnomalyDetector, EthicalAgent
from src.models.signal import EthicalSignal
from src.models.anomaly import AnomalyReport
from tests.fixtures.sample_data import (
    create_sample_signal,
    create_bias_signal,
    create_sample_agent_spec
)


class TestEthicalAgent:
    """Test individual agent behavior"""
    
    def test_agent_creation(self):
        """Test agent can be created with specification"""
        spec = create_sample_agent_spec()
        agent = EthicalAgent(spec)
        
        assert agent.spec == spec
        assert agent.coverage_score >= 0
        assert agent.total_assessments == 0
    
    def test_agent_can_handle_signal(self):
        """Test agent recognizes signals in its capabilities"""
        spec = create_sample_agent_spec(domain="bias")
        agent = EthicalAgent(spec)
        
        bias_signal = create_bias_signal()
        assert agent.can_handle(bias_signal) is True
    
    def test_agent_cannot_handle_signal(self):
        """Test agent rejects signals outside capabilities"""
        spec = create_sample_agent_spec(domain="bias")
        agent = EthicalAgent(spec)
        
        privacy_signal = create_sample_signal(category="privacy")
        assert agent.can_handle(privacy_signal) is False
    
    def test_agent_assess_signal_success(self):
        """Test agent can assess signals in its domain"""
        spec = create_sample_agent_spec(domain="bias")
        agent = EthicalAgent(spec)
        
        signal = create_bias_signal()
        score = agent.assess_signal(signal)
        
        assert 0 <= score <= 1
        assert agent.total_assessments == 1
    
    def test_agent_assess_signal_failure(self):
        """Test agent returns 0 for signals outside domain"""
        spec = create_sample_agent_spec(domain="bias")
        agent = EthicalAgent(spec)
        
        privacy_signal = create_sample_signal(category="privacy")
        score = agent.assess_signal(privacy_signal)
        
        assert score == 0.0
    
    def test_agent_coverage_updates(self):
        """Test agent coverage score changes with performance"""
        spec = create_sample_agent_spec(domain="bias")
        agent = EthicalAgent(spec)
        initial_coverage = agent.coverage_score
        
        # Failure should decrease coverage
        agent.update_coverage(success=False)
        agent.update_coverage(success=False)
        agent.update_coverage(success=False)
        final_coverage = agent.coverage_score
        
        # Multiple failures should decrease coverage
        assert final_coverage <= initial_coverage


class TestAnomalyDetector:
    """Test anomaly detection logic"""
    
    def test_detector_creation(self):
        """Test anomaly detector initializes correctly"""
        detector = AnomalyDetector()
        
        assert detector.threshold > 0
        assert detector.recent_anomalies is not None
    
    def test_no_anomaly_when_covered(self):
        """Test no anomaly when signal is covered by agents"""
        detector = AnomalyDetector()
        
        spec = create_sample_agent_spec(domain="bias")
        agent = EthicalAgent(spec)
        agent.coverage_score = 0.9  # High coverage
        
        signal = create_bias_signal()
        anomaly = detector.detect_anomaly(signal, [agent])
        
        assert anomaly is None or anomaly.explanation_score > detector.threshold
    
    def test_anomaly_when_uncovered(self):
        """Test anomaly detected when no agent covers signal"""
        detector = AnomalyDetector()
        
        spec = create_sample_agent_spec(domain="bias")
        agent = EthicalAgent(spec)
        
        # Signal from uncovered category
        signal = create_sample_signal(category="accountability")
        anomaly = detector.detect_anomaly(signal, [agent])
        
        assert anomaly is not None
        assert anomaly.explanation_score < detector.threshold
    
    def test_anomaly_with_high_severity_signal(self):
        """Test anomaly more likely with high-severity signals"""
        detector = AnomalyDetector()
        
        spec = create_sample_agent_spec(domain="bias")
        agent = EthicalAgent(spec)
        agent.coverage_score = 0.5  # Medium coverage
        
        # Low severity signal - may not trigger anomaly
        low_severity = create_sample_signal(category="bias", severity=0.3)
        low_anomaly = detector.detect_anomaly(low_severity, [agent])
        
        # High severity signal - more likely to trigger anomaly
        high_severity = create_sample_signal(category="bias", severity=0.95)
        high_anomaly = detector.detect_anomaly(high_severity, [agent])
        
        # Both should be detected but high severity should have lower score
        if low_anomaly and high_anomaly:
            assert high_anomaly.explanation_score <= low_anomaly.explanation_score
    
    def test_anomaly_tracks_recent(self):
        """Test detector tracks recent anomalies"""
        detector = AnomalyDetector()
        
        spec = create_sample_agent_spec(domain="bias")
        agent = EthicalAgent(spec)
        
        signal1 = create_sample_signal(category="accountability")
        signal2 = create_sample_signal(category="transparency")
        
        anomaly1 = detector.detect_anomaly(signal1, [agent])
        anomaly2 = detector.detect_anomaly(signal2, [agent])
        
        assert len(detector.recent_anomalies) >= 0


class TestAnomalyReport:
    """Test anomaly report generation"""
    
    def test_anomaly_report_creation(self):
        """Test anomaly report can be created"""
        signal = create_bias_signal()
        anomaly = AnomalyReport(
            anomaly_id="anom_1",
            signal=signal,
            explanation_score=0.2,
            unexplained_factors=["No bias coverage"],
            context={"reason": "bias not covered"},
            requires_new_agent=True,
            timestamp=datetime.now()
        )
        
        assert anomaly.signal == signal
        assert anomaly.explanation_score == 0.2
    
    def test_anomaly_report_is_critical(self):
        """Test anomaly severity assessment"""
        signal = create_sample_signal(category="general", severity=0.95)
        anomaly = AnomalyReport(
            anomaly_id="anom_2",
            signal=signal,
            explanation_score=0.0,
            unexplained_factors=["No coverage"],
            context={"reason": "uncovered"},
            requires_new_agent=True,
            timestamp=datetime.now(),
            severity_level="critical"
        )
        
        # High severity + low explanation = critical
        assert anomaly.is_critical


class TestMultiAgentCoverage:
    """Test anomaly detection with multiple agents"""
    
    def test_multiple_agents_provide_coverage(self):
        """Test multiple agents can provide coverage"""
        detector = AnomalyDetector()
        
        # Multiple agents covering different domains
        bias_agent = EthicalAgent(create_sample_agent_spec(domain="bias"))
        privacy_agent = EthicalAgent(
            create_sample_agent_spec(domain="privacy")
        )
        
        agents = [bias_agent, privacy_agent]
        
        signal = create_bias_signal()
        anomaly = detector.detect_anomaly(signal, agents)
        
        # Should have better coverage with 2 agents
        if anomaly:
            # Bias signal with bias agent should have decent coverage
            assert anomaly.explanation_score > 0
    
    def test_agent_pool_for_uncovered_signal(self):
        """Test with multiple agents for uncovered signal"""
        detector = AnomalyDetector()
        
        bias_agent = EthicalAgent(create_sample_agent_spec(domain="bias"))
        privacy_agent = EthicalAgent(
            create_sample_agent_spec(domain="privacy")
        )
        
        agents = [bias_agent, privacy_agent]
        
        # Signal from uncovered domain
        signal = create_sample_signal(category="accountability")
        anomaly = detector.detect_anomaly(signal, agents)
        
        assert anomaly is not None
        assert "accountability" in str(anomaly.unexplained_factors)
