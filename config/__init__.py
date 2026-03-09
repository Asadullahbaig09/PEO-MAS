"""Configuration module for Perpetual Ethical Oversight MAS"""

from config.settings import settings
from config.constants import (
    AGENT_CAPABILITIES,
    SIGNAL_CATEGORIES,
    SCRAPER_SOURCES,
    AGENT_TOOLS,
    SEVERITY_LEVELS
)

__all__ = [
    'settings',
    'AGENT_CAPABILITIES',
    'SIGNAL_CATEGORIES',
    'SCRAPER_SOURCES',
    'AGENT_TOOLS',
    'SEVERITY_LEVELS'
]