from typing import List, Dict
from src.cognitive.anomaly_detector import EthicalAgent


class AgentPool:
    """Manages collection of ethical agents"""
    
    def __init__(self):
        self.agents: Dict[str, EthicalAgent] = {}
        
    def add_agent(self, agent: EthicalAgent):
        """Add agent to pool"""
        self.agents[agent.spec.agent_id] = agent
        
    def get_agent(self, agent_id: str) -> EthicalAgent:
        """Get specific agent"""
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> List[EthicalAgent]:
        """Get all agents"""
        return list(self.agents.values())
    
    def get_agents_for_category(self, category: str) -> List[EthicalAgent]:
        """Get agents that can handle specific category"""
        return [
            agent for agent in self.agents.values()
            if category in agent.spec.capabilities
        ]
    
    def get_statistics(self) -> Dict:
        """Get pool statistics with performance metrics"""
        agents_list = list(self.agents.values())
        
        return {
            'total_agents': len(self.agents),
            'average_coverage': (
                sum(a.coverage_score for a in agents_list) / len(agents_list) 
                if agents_list else 0
            ),
            'agent_ids': list(self.agents.keys()),
            'performance_by_agent': {
                agent_id: agent.get_performance_stats()
                for agent_id, agent in self.agents.items()
            }
        }