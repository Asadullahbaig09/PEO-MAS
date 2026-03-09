import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings - 100% Free Version with Optional LLM"""
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    OUTPUT_DIR: Path = PROJECT_ROOT / "output"
    LOG_FILE: str = str(PROJECT_ROOT / "logs" / "system.log")
    
    # System Configuration
    LOG_LEVEL: str = "INFO"
    PROCESSING_INTERVAL_SECONDS: int = 60
    MAX_SIGNALS_PER_CYCLE: int = 50
    
    # Thresholds
    ETHICAL_COVERAGE_THRESHOLD: float = 0.65
    ANOMALY_SEVERITY_THRESHOLD: float = 0.75
    DECAY_RATE: float = 0.95
    
    # Agent Management
    MAX_AGENTS: int = 50
    AGENT_RETIREMENT_THRESHOLD: float = 0.3
    AGENT_PERFORMANCE_WINDOW: int = 10
    COLLABORATIVE_DECISION_THRESHOLD: int = 3
    
    # Local LLM Configuration - HuggingFace Transformers
    USE_LOCAL_LLM: bool = True
    USE_HUGGINGFACE: bool = True
    HUGGINGFACE_MODEL: str = "mistralai/Mistral-7B-v0.1"
    FINETUNED_MODEL_PATH: Path = PROJECT_ROOT / "models" / "mistral_law_generator" / "final"
    
    # Legacy Ollama config (deprecated)
    # OLLAMA_API_URL: str = "http://localhost:11434"
    # OLLAMA_MODEL: str = "mistral:7b"
    # OLLAMA_TIMEOUT: int = 60
    
    # Free Embeddings
    USE_REAL_EMBEDDINGS: bool = True
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_MODEL_PATH: Path = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"
    
    # Free API Settings
    ARXIV_MAX_RESULTS: int = 10
    
    # RSS Feeds for News (Free)
    NEWS_FEEDS: list = [
        "https://feeds.arstechnica.com/arstechnica/technology-lab",
        "https://www.wired.com/feed/category/business/latest/rss",
        "https://techcrunch.com/feed/",
    ]
    
    # Dashboard
    DASHBOARD_HOST: str = "127.0.0.1"
    DASHBOARD_PORT: int = 8050
    DASHBOARD_DEBUG: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
Path(settings.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)