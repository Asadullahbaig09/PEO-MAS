import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict


class TimeAwareKnowledgeGraph:
    """Knowledge graph with temporal decay for ethical norms"""
    
    def __init__(self, decay_rate: float = 0.95):
        self.nodes = {}  # node_id -> {state, last_update, history}
        self.edges = defaultdict(list)  # source -> [(target, weight)]
        self.decay_rate = decay_rate
        
    def add_node(self, node_id: str, initial_state: float = 0.0):
        """Add a node representing an ethical concept"""
        self.nodes[node_id] = {
            'state': initial_state,
            'last_update': datetime.now(),
            'history': [(datetime.now(), initial_state)]
        }
        
    def update_node(self, node_id: str, signal_intensity: float):
        """Update node state with temporal decay: S(t) = α^t * S(t-1) + E(t)"""
        if node_id not in self.nodes:
            self.add_node(node_id)
        
        node = self.nodes[node_id]
        time_delta = (datetime.now() - node['last_update']).total_seconds() / 3600
        
        # Apply decay
        decayed_state = (self.decay_rate ** time_delta) * node['state']
        new_state = decayed_state + signal_intensity
        
        node['state'] = new_state
        node['last_update'] = datetime.now()
        node['history'].append((datetime.now(), new_state))
        
    def add_edge(self, source: str, target: str, weight: float = 1.0):
        """Add relationship between ethical concepts"""
        self.edges[source].append((target, weight))
        
    def get_node_state(self, node_id: str, current_time: datetime = None) -> float:
        """Calculate node state at given time with decay"""
        if node_id not in self.nodes:
            return 0.0
        
        node = self.nodes[node_id]
        if current_time is None:
            current_time = datetime.now()
        
        time_delta = (current_time - node['last_update']).total_seconds() / 3600
        return (self.decay_rate ** time_delta) * node['state']
    
    def get_related_concepts(self, node_id: str) -> List[str]:
        """Get concepts related to given node"""
        return [target for target, _ in self.edges.get(node_id, [])]
    
    def export_state(self) -> Dict:
        """Export current graph state"""
        return {
            'nodes': {k: v['state'] for k, v in self.nodes.items()},
            'edges': dict(self.edges),
            'timestamp': datetime.now().isoformat()
        }
