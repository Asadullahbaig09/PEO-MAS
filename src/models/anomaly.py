from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any
from src.models.signal import EthicalSignal


@dataclass
class AnomalyReport:
    """Report of detected ethical anomaly"""
    anomaly_id: str
    signal: EthicalSignal
    explanation_score: float
    unexplained_factors: List[str]
    context: Dict[str, Any]
    requires_new_agent: bool
    timestamp: datetime
    severity_level: str = "medium"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'anomaly_id': self.anomaly_id,
            'signal': self.signal.to_dict(),
            'explanation_score': float(self.explanation_score),
            'unexplained_factors': self.unexplained_factors,
            'context': self.context,
            'requires_new_agent': self.requires_new_agent,
            'severity_level': self.severity_level,
            'timestamp': self.timestamp.isoformat()
        }
    
    @property
    def is_critical(self) -> bool:
        """Check if anomaly is critical"""
        return (
            self.severity_level == "critical" or
            (self.signal.severity > 0.85 and self.explanation_score < 0.3)
        )
    
    def __repr__(self) -> str:
        return (f"AnomalyReport(id={self.anomaly_id[:8]}, "
                f"severity={self.severity_level}, "
                f"explanation={self.explanation_score:.2f})")


@dataclass
class EvolutionEvent:
    """Record of system evolution (new agent creation)"""
    event_id: str
    trigger_anomaly: AnomalyReport
    new_agent_spec: 'AgentSpecification'
    timestamp: datetime
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'event_id': self.event_id,
            'trigger_anomaly_id': self.trigger_anomaly.anomaly_id,
            'new_agent_id': self.new_agent_spec.agent_id,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'error_message': self.error_message
        }