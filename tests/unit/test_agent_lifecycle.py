"""
Unit tests for agent lifecycle management
Tests agent creation, performance tracking, and retirement
"""

import pytest
from datetime import datetime

from src.cognitive.agent_lifecycle import AgentLifecycleManager
from src.cognitive.anomaly_detector import EthicalAgent
from src.models.agent import AgentSpecification
from tests.fixtures.sample_data import (
    create_sample_agent_spec,
    create_fairness_agent_spec,
    create_privacy_agent_spec,
    create_transparency_agent_spec
)


class TestAgentLifecycleManager:
    """Test agent lifecycle management"""
    
    def test_manager_creation(self):
        """Test lifecycle manager initializes correctly"""
        manager = AgentLifecycleManager()
        
        assert manager.active_agent_count == 0
        assert len(manager.retired_agents) == 0
        assert len(manager.agent_performance) == 0
    
    def test_can_spawn_new_agent_when_space_available(self):
        """Test new agent can be spawned when space available"""
        manager = AgentLifecycleManager()
        
        new_spec = create_sample_agent_spec(domain="bias")
        existing_agents = []
        
        can_spawn = manager.should_spawn_new_agent(new_spec, existing_agents)
        assert can_spawn is True
    
    def test_cannot_spawn_duplicate_domain_agent(self):
        """Test duplicate domain agents are prevented"""
        manager = AgentLifecycleManager()
        
        # Create first bias agent
        bias_spec = create_fairness_agent_spec()
        bias_agent = EthicalAgent(bias_spec)
        
        existing_agents = [bias_agent]
        
        # Try to create second bias agent
        new_bias_spec = create_sample_agent_spec(
            agent_id="agent_bias_2",
            domain="bias"
        )
        
        can_spawn = manager.should_spawn_new_agent(new_bias_spec, existing_agents)
        assert can_spawn is False
    
    def test_can_spawn_different_domain_agent(self):
        """Test agents from different domains can coexist"""
        manager = AgentLifecycleManager()
        
        bias_agent = EthicalAgent(create_fairness_agent_spec())
        existing_agents = [bias_agent]
        
        # Create different domain agent
        privacy_spec = create_privacy_agent_spec()
        
        can_spawn = manager.should_spawn_new_agent(privacy_spec, existing_agents)
        assert can_spawn is True
    
    def test_register_agent_for_tracking(self):
        """Test agent registration for performance tracking"""
        manager = AgentLifecycleManager()
        
        spec = create_sample_agent_spec()
        agent = EthicalAgent(spec)
        
        manager.register_agent(agent)
        
        assert spec.agent_id in manager.agent_performance
        assert manager.active_agent_count == 1
    
    def test_update_agent_performance(self):
        """Test agent performance metrics are updated"""
        manager = AgentLifecycleManager()
        
        spec = create_sample_agent_spec()
        agent = EthicalAgent(spec)
        manager.register_agent(agent)
        
        # Record successful assessment
        manager.update_agent_performance(spec.agent_id, success=True, confidence=0.9)
        
        metrics = manager.agent_performance[spec.agent_id]
        assert metrics.signals_processed == 1
        assert metrics.successful_assessments == 1
    
    def test_identify_low_performers_for_retirement(self):
        """Test identification of low-performing agents for retirement"""
        manager = AgentLifecycleManager()
        
        spec = create_sample_agent_spec()
        agent = EthicalAgent(spec)
        agent.coverage_score = 0.2  # Very low coverage
        
        manager.register_agent(agent)
        
        # Record many failures
        for _ in range(15):
            manager.update_agent_performance(spec.agent_id, success=False, confidence=0.1)
        
        retirement_candidates = manager.check_retirement_candidates([agent])
        
        # Low coverage agent should be marked for retirement
        assert spec.agent_id in retirement_candidates
    
    def test_retire_agent(self):
        """Test agent retirement"""
        manager = AgentLifecycleManager()
        
        spec = create_sample_agent_spec()
        agent = EthicalAgent(spec)
        manager.register_agent(agent)
        
        initial_count = manager.active_agent_count
        manager.retire_agent(spec.agent_id)
        
        assert spec.agent_id in manager.retired_agents
        assert manager.active_agent_count < initial_count
    
    def test_high_success_rate_prevents_retirement(self):
        """Test high-performing agents are not retired"""
        manager = AgentLifecycleManager()
        
        spec = create_sample_agent_spec()
        agent = EthicalAgent(spec)
        agent.coverage_score = 0.85  # Good coverage
        
        manager.register_agent(agent)
        
        # Record successes
        for _ in range(10):
            manager.update_agent_performance(spec.agent_id, success=True, confidence=0.9)
        
        retirement_candidates = manager.check_retirement_candidates([agent])
        
        # Good performer should not be marked for retirement
        assert spec.agent_id not in retirement_candidates
    
    def test_get_agent_status(self):
        """Test retrieving agent status information"""
        manager = AgentLifecycleManager()
        
        spec = create_sample_agent_spec()
        agent = EthicalAgent(spec)
        manager.register_agent(agent)
        
        # Add some performance data
        manager.update_agent_performance(spec.agent_id, success=True, confidence=0.85)
        manager.update_agent_performance(spec.agent_id, success=False, confidence=0.3)
        
        status = manager.get_agent_status(spec.agent_id)
        
        assert status is not None
        assert status['agent_id'] == spec.agent_id
        assert status['signals_processed'] == 2
        assert 'success_rate' in status
    
    def test_get_all_agent_statuses(self):
        """Test retrieving status of all agents"""
        manager = AgentLifecycleManager()
        
        agents = [
            EthicalAgent(create_fairness_agent_spec()),
            EthicalAgent(create_privacy_agent_spec()),
            EthicalAgent(create_transparency_agent_spec()),
        ]
        
        for agent in agents:
            manager.register_agent(agent)
        
        all_statuses = manager.get_all_agent_statuses()
        
        assert len(all_statuses) == 3
        assert all('agent_id' in s for s in all_statuses)
    
    def test_get_statistics(self):
        """Test lifecycle manager statistics"""
        manager = AgentLifecycleManager()
        
        spec = create_sample_agent_spec()
        agent = EthicalAgent(spec)
        manager.register_agent(agent)
        
        stats = manager.get_statistics()
        
        assert 'active_agents' in stats
        assert 'retired_agents' in stats
        assert 'max_agents' in stats
        assert stats['active_agents'] == 1


class TestMultiAgentLifecycle:
    """Test lifecycle management with multiple agents"""
    
    def test_managing_multiple_agents(self):
        """Test managing multiple agents simultaneously"""
        manager = AgentLifecycleManager()
        
        agents = [
            EthicalAgent(create_fairness_agent_spec()),
            EthicalAgent(create_privacy_agent_spec()),
            EthicalAgent(create_transparency_agent_spec()),
        ]
        
        for agent in agents:
            manager.register_agent(agent)
        
        assert manager.active_agent_count == 3
    
    def test_selective_retirement(self):
        """Test retiring some agents while keeping others"""
        manager = AgentLifecycleManager()
        
        # Create one good and one bad agent
        good_agent = EthicalAgent(create_fairness_agent_spec())
        good_agent.coverage_score = 0.85
        
        bad_agent = EthicalAgent(create_privacy_agent_spec())
        bad_agent.coverage_score = 0.2
        
        manager.register_agent(good_agent)
        manager.register_agent(bad_agent)
        
        # Record performance
        manager.update_agent_performance(
            good_agent.spec.agent_id, success=True, confidence=0.9
        )
        manager.update_agent_performance(
            bad_agent.spec.agent_id, success=False, confidence=0.1
        )
        
        # Check retirement candidates
        candidates = manager.check_retirement_candidates([good_agent, bad_agent])
        
        # Bad agent should be marked for retirement
        assert bad_agent.spec.agent_id in candidates
        # Good agent should not be marked
        assert good_agent.spec.agent_id not in candidates
    
    def test_prevent_domain_overlap(self):
        """Test system prevents agents with same domain from spawning"""
        manager = AgentLifecycleManager()
        
        # Register first bias agent
        bias_agent1 = EthicalAgent(create_fairness_agent_spec())
        existing = [bias_agent1]
        
        # Try to spawn second bias agent
        bias_spec2 = create_sample_agent_spec(
            agent_id="bias_agent_2",
            name="Bias Monitor 2",
            domain="bias"
        )
        
        can_spawn = manager.should_spawn_new_agent(bias_spec2, existing)
        assert can_spawn is False
        
        # But privacy agent should be allowed
        privacy_spec = create_sample_agent_spec(
            agent_id="privacy_agent",
            domain="privacy"
        )
        
        can_spawn = manager.should_spawn_new_agent(privacy_spec, existing)
        assert can_spawn is True
