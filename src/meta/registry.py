from typing import Dict, List, Optional

from src.models.agent import AgentSpecification
from src.cognitive.anomaly_detector import EthicalAgent


class AgentRegistry:
    """Registry managing all agents (AgentNet++ style) with RAG support"""
    
    def __init__(self, retriever=None, generator=None):
        """
        Initialize agent registry with optional RAG components
        
        Args:
            retriever: Optional DocumentRetriever for RAG
            generator: Optional EthicalAssessmentGenerator for RAG
        """
        self.agents: Dict[str, EthicalAgent] = {}
        self.message_bus = []
        
        # RAG components (shared across all agents)
        self.retriever = retriever
        self.generator = generator
        
        if retriever and generator:
            print("✓ Agent Registry initialized with RAG support")
        
    def register(self, spec: AgentSpecification) -> EthicalAgent:
        """Register new agent in the system with RAG capabilities"""
        # Create agent with RAG components if available
        agent = EthicalAgent(
            spec=spec,
            retriever=self.retriever,
            generator=self.generator
        )
        self.agents[spec.agent_id] = agent
        
        self._setup_communication(agent)
        
        rag_status = "with RAG" if agent.use_rag else "without RAG"
        print(f"✓ Registered new agent: {spec.name} (ID: {spec.agent_id}) {rag_status}")
        return agent
    
    def _setup_communication(self, agent: EthicalAgent):
        """Setup message-passing capabilities"""
        # In production: implement actual message queue (Redis, RabbitMQ)
        pass
    
    def get_all_agents(self) -> List[EthicalAgent]:
        """Get all registered agents"""
        return list(self.agents.values())
    
    def get_agent(self, agent_id: str) -> Optional[EthicalAgent]:
        """Get specific agent by ID"""
        return self.agents.get(agent_id)
    
    def deregister(self, agent_id: str) -> bool:
        """Remove agent from registry"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            print(f"✓ Deregistered agent: {agent_id}")
            return True
        return False
    
    def get_statistics(self) -> Dict:
        """Get registry statistics"""
        return {
            'total_agents': len(self.agents),
            'agent_ids': list(self.agents.keys()),
            'agent_names': [a.spec.name for a in self.agents.values()]
        }
