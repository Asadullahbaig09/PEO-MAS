"""System-wide constants"""

# Agent capabilities mapping
AGENT_CAPABILITIES = {
    'bias': ['fairness', 'discrimination', 'equity'],
    'privacy': ['data_protection', 'gdpr', 'encryption'],
    'transparency': ['explainability', 'interpretability', 'auditability'],
    'accountability': ['liability', 'responsibility', 'governance'],
    'safety': ['harm_prevention', 'risk_assessment'],
    'security': ['breach_detection', 'vulnerability']
}

# Signal categories
SIGNAL_CATEGORIES = [
    'bias',
    'privacy', 
    'transparency',
    'accountability',
    'safety',
    'security',
    'general'
]

# Scraper sources
SCRAPER_SOURCES = {
    'legal': ['PACER', 'EUR-Lex', 'US-Courts'],
    'academic': ['ArXiv', 'SSRN', 'PubMed'],
    'social': ['Twitter', 'Reddit', 'HackerNews'],
    'news': ['Reuters', 'Bloomberg', 'TechCrunch']
}

# Tool types available to agents
AGENT_TOOLS = [
    'search',
    'analyze',
    'report',
    'statistical_analysis',
    'demographic_breakdown',
    'data_flow_analysis',
    'encryption_check',
    'model_explanation',
    'audit_trail',
    'compliance_check',
    'impact_assessment'
]

# Severity levels
SEVERITY_LEVELS = {
    'low': (0.0, 0.3),
    'medium': (0.3, 0.6),
    'high': (0.6, 0.85),
    'critical': (0.85, 1.0)
}