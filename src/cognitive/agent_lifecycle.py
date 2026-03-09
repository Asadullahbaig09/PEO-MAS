"""
Agent lifecycle management
Handles agent retirement, performance tracking, and duplicate prevention
"""

import logging
from typing import List, Optional
from datetime import datetime

from src.models.agent import AgentSpecification, AgentPerformanceMetrics
from src.cognitive.anomaly_detector import EthicalAgent
from config.settings import settings

logger = logging.getLogger(__name__)


class AgentLifecycleManager:
    """
    Manages agent lifecycle: creation, performance tracking, retirement
    Prevents spawning duplicate agents
    """
    
    def __init__(self):
        self.agent_performance: dict = {}  # agent_id -> AgentPerformanceMetrics
        self.agent_history: dict = {}      # agent_id -> spec (for comparison)
        self.active_agent_count = 0
        self.retired_agents: List[str] = []
    
    def should_spawn_new_agent(
        self,
        new_spec: AgentSpecification,
        existing_agents: List[EthicalAgent]
    ) -> bool:
        """
        Check if new agent should be spawned
        Prevents duplicate agents covering same domain
        """
        # Check if agent with same domain already exists
        for agent in existing_agents:
            if agent.spec.domain == new_spec.domain:
                logger.warning(
                    f"Agent for domain '{new_spec.domain}' already exists "
                    f"({agent.spec.name}). Skipping duplicate spawn."
                )
                return False
        
        # Check if max agents exceeded
        if self.active_agent_count >= settings.MAX_AGENTS:
            logger.warning(
                f"Max agents ({settings.MAX_AGENTS}) reached. "
                f"Consider retiring low-performing agents first."
            )
            return False
        
        return True
    
    def register_agent(
        self,
        agent: EthicalAgent,
        spawned_from: str = None
    ):
        """Register new agent for performance tracking"""
        agent_id = agent.spec.agent_id
        
        self.agent_performance[agent_id] = AgentPerformanceMetrics(
            agent_id=agent_id
        )
        self.agent_history[agent_id] = agent.spec
        self.active_agent_count += 1
        
        logger.info(
            f"Agent registered for tracking: {agent.spec.name} "
            f"(Domain: {agent.spec.domain})"
        )
    
    def update_agent_performance(
        self,
        agent_id: str,
        success: bool,
        confidence: float
    ):
        """Update agent performance metrics"""
        if agent_id not in self.agent_performance:
            return
        
        metrics = self.agent_performance[agent_id]
        metrics.update_metrics(success, confidence)
    
    def check_retirement_candidates(
        self,
        agents: List[EthicalAgent]
    ) -> List[str]:
        """
        Identify agents for retirement
        Returns list of agent IDs to retire
        """
        retirement_candidates = []
        
        for agent in agents:
            agent_id = agent.spec.agent_id
            
            if agent_id not in self.agent_performance:
                continue
            
            metrics = self.agent_performance[agent_id]
            
            # Retirement criteria
            should_retire = False
            reason = None
            
            # Low success rate (< 30%)
            if metrics.success_rate < 0.3 and metrics.signals_processed >= 10:
                should_retire = True
                reason = f"Low success rate: {metrics.success_rate:.1%}"
            
            # Low coverage score
            elif agent.coverage_score < settings.AGENT_RETIREMENT_THRESHOLD:
                should_retire = True
                reason = f"Low coverage: {agent.coverage_score:.2f}"
            
            # High failure count
            elif metrics.failed_assessments >= 20:
                should_retire = True
                reason = f"Too many failures: {metrics.failed_assessments}"
            
            if should_retire:
                retirement_candidates.append(agent_id)
                logger.warning(
                    f"Agent {agent.spec.name} marked for retirement. "
                    f"Reason: {reason}"
                )
        
        return retirement_candidates
    
    def retire_agent(self, agent_id: str):
        """Retire an agent from the system"""
        if agent_id in self.agent_performance:
            metrics = self.agent_performance[agent_id]
            
            logger.info(
                f"Agent retired: {agent_id}. "
                f"Performance: {metrics.success_rate:.1%} success rate, "
                f"{metrics.signals_processed} signals processed"
            )
            
            self.retired_agents.append(agent_id)
            self.active_agent_count = max(0, self.active_agent_count - 1)
    
    def get_agent_status(self, agent_id: str) -> Optional[dict]:
        """Get detailed status of an agent"""
        if agent_id not in self.agent_performance:
            return None
        
        metrics = self.agent_performance[agent_id]
        spec = self.agent_history.get(agent_id)
        
        return {
            'agent_id': agent_id,
            'name': spec.name if spec else 'Unknown',
            'domain': spec.domain if spec else 'Unknown',
            'signals_processed': metrics.signals_processed,
            'success_rate': f"{metrics.success_rate:.1%}",
            'average_confidence': f"{metrics.average_confidence:.2f}",
            'coverage_score': f"{metrics.coverage_score:.2f}",
            'last_updated': metrics.last_updated.isoformat(),
            'retirement_status': (
                'RETIRED' if agent_id in self.retired_agents else 'ACTIVE'
            )
        }
    
    def get_all_agent_statuses(self) -> List[dict]:
        """Get status of all agents"""
        statuses = []
        for agent_id in self.agent_performance.keys():
            status = self.get_agent_status(agent_id)
            if status:
                statuses.append(status)
        return statuses
    
    def get_statistics(self) -> dict:
        """Get lifecycle management statistics"""
        return {
            'active_agents': self.active_agent_count,
            'retired_agents': len(self.retired_agents),
            'max_agents': settings.MAX_AGENTS,
            'capacity_used': f"{(self.active_agent_count / settings.MAX_AGENTS * 100):.1f}%",
            'total_agents_tracked': len(self.agent_performance),
            'average_success_rate': (
                sum(m.success_rate for m in self.agent_performance.values()) /
                len(self.agent_performance)
                if self.agent_performance else 0
            )
        }
