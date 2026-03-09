from typing import List
from datetime import datetime
import logging

from src.ingestion.base import EthicalSignalScraper
from src.ingestion.legal_scrapers import LegalAPIScraper
from src.ingestion.academic_scrapers import ArXivScraper
from src.ingestion.social_scrapers import RedditScraper  # Only Reddit!

try:
    from src.ingestion.news_scrapers import RSSNewsScraper
    NEWS_AVAILABLE = True
except ImportError:
    NEWS_AVAILABLE = False

from src.models.signal import EthicalSignal
from config.settings import settings

logger = logging.getLogger(__name__)


class IngestionLayer:
    """Orchestrates all signal scrapers — real-time APIs only"""
    
    def __init__(self):
        self.scrapers = [
            LegalAPIScraper("EFF/EU-AI-Act RSS"),       # Real-time legal RSS feeds
            ArXivScraper(),                             # ArXiv API (real)
            RedditScraper(),                            # Reddit JSON API (real)
        ]
        
        # Add RSS if available
        if NEWS_AVAILABLE:
            try:
                self.scrapers.append(RSSNewsScraper())
                logger.info("[OK] RSS News scraper enabled")
            except Exception as e:
                logger.warning(f"Could not load RSS News scraper: {e}")
        
        self.total_signals_collected = 0
        self.last_collection = None
        
        logger.info(f"[OK] Initialized with {len(self.scrapers)} real-time scrapers")
        logger.info("  - ArXiv: arxiv.org API")
        logger.info("  - Reddit: reddit.com JSON API")
        logger.info("  - Legal: EFF/EU-AI-Act RSS")
    
    def collect_signals(self) -> List[EthicalSignal]:
        """Collect signals from all sources"""
        all_signals = []
        
        for scraper in self.scrapers:
            try:
                signals = scraper.fetch_signals()
                all_signals.extend(signals)
                logger.info(f"[OK] Collected {len(signals)} from {scraper.source_name}")
            except Exception as e:
                logger.error(f"[ERR] Error scraping {scraper.source_name}: {e}")
        
        if len(all_signals) > settings.MAX_SIGNALS_PER_CYCLE:
            # Shuffle to ensure balanced representation from all sources
            import random
            random.shuffle(all_signals)
            logger.info(f"Limiting signals from {len(all_signals)} to {settings.MAX_SIGNALS_PER_CYCLE}")
            all_signals = all_signals[:settings.MAX_SIGNALS_PER_CYCLE]
        
        self.total_signals_collected += len(all_signals)
        self.last_collection = datetime.now()
        
        return all_signals
    
    def add_scraper(self, scraper: EthicalSignalScraper):
        """Add scraper dynamically"""
        self.scrapers.append(scraper)
        logger.info(f"Added scraper: {scraper.source_name}")
    
    def get_statistics(self) -> dict:
        """Get statistics"""
        return {
            'total_scrapers': len(self.scrapers),
            'total_signals_collected': self.total_signals_collected,
            'last_collection': self.last_collection.isoformat() if self.last_collection else None,
            'scraper_sources': [s.source_name for s in self.scrapers]
        }