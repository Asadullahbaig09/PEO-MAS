"""Ingestion Layer - Data collection from external sources"""

from src.ingestion.base import EthicalSignalScraper
from src.ingestion.legal_scrapers import LegalAPIScraper
from src.ingestion.academic_scrapers import ArXivScraper
from src.ingestion.ingestion_layer import IngestionLayer

# Import social and news scrapers with try-except
try:
    from src.ingestion.social_scrapers import RedditScraper
except ImportError:
    from src.ingestion.social_scrapers import RedditScraper
    RedditScraper = None

try:
    from src.ingestion.news_scrapers import RSSNewsScraper
except ImportError:
    RSSNewsScraper = None

__all__ = [
    'EthicalSignalScraper',
    'LegalAPIScraper',
    'ArXivScraper',
    'IngestionLayer'
]

if RedditScraper:
    __all__.append('RedditScraper')

if RSSNewsScraper:
    __all__.append('RSSNewsScraper')