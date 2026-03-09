"""Meta Layer - Self-evolution and agent generation"""

from src.meta.generator import MetaAgentGenerator
from src.meta.registry import AgentRegistry

# Import LLM interface with try-except to handle if file doesn't exist
try:
    from src.meta.llm_interface import LocalLLMInterface
    __all__ = [
        'MetaAgentGenerator',
        'AgentRegistry',
        'LocalLLMInterface'
    ]
except ImportError:
    # LLM interface not available, only export generator and registry
    __all__ = [
        'MetaAgentGenerator',
        'AgentRegistry'
    ]