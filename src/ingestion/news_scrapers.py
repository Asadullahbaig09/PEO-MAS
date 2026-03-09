from typing import List
import hashlib
from datetime import datetime
import logging
import random

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    feedparser = None

from src.ingestion.base import EthicalSignalScraper
from src.models.signal import EthicalSignal
from config.settings import settings

logger = logging.getLogger(__name__)


class RSSNewsScraper(EthicalSignalScraper):
    """Free RSS feed scraper for tech news"""
    
    def __init__(self):
        super().__init__("RSS News")
        
        if not FEEDPARSER_AVAILABLE:
            logger.warning("feedparser not installed — RSS scraping disabled")
            logger.warning("Install with: pip install feedparser")
        
        self.feeds = getattr(settings, 'NEWS_FEEDS', [
            "https://feeds.arstechnica.com/arstechnica/tech-policy",  # Ars Technica tech policy (20 entries)
            "https://www.technologyreview.com/feed/",               # MIT Technology Review (10 entries)
            "https://ainowinstitute.org/feed",                       # AI Now Institute (10 entries)
            "https://www.wired.com/feed/tag/ai/latest/rss",          # Wired AI (10 entries)
            "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",  # The Verge AI (10 entries)
            "https://spectrum.ieee.org/feeds/topic/artificial-intelligence.rss",  # IEEE Spectrum AI (30 entries)
        ])
    
    def fetch_signals(self) -> List[EthicalSignal]:
        """Fetch news from ALL RSS feeds"""
        
        if not FEEDPARSER_AVAILABLE:
            logger.warning("feedparser not installed — returning empty")
            return []
        
        signals = []
        seen_links: set = set()
        
        for feed_url in self.feeds:
            try:
                feed = feedparser.parse(feed_url)
                entries = feed.entries[:15]  # Up to 15 per feed
                
                for entry in entries:
                    link = entry.get('link', '')
                    if link in seen_links:
                        continue
                    
                    title = entry.get('title', '')
                    summary = entry.get('summary', entry.get('description', ''))
                    combined = f"{title} {summary}".lower()
                    
                    if self._is_relevant(combined):
                        seen_links.add(link)
                        signal = self._create_signal_from_entry(entry, feed_url)
                        signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error fetching RSS feed {feed_url}: {e}")
                continue
        
        logger.info(f"Fetched {len(signals)} relevant signals from {len(self.feeds)} RSS feeds")
        self.last_scrape = datetime.now()
        return signals
    
    def _is_relevant(self, text: str) -> bool:
        """Check if news is relevant to AI ethics — requires BOTH an AI term AND an ethics/policy term"""
        ai_keywords = [
            'ai', 'artificial intelligence', 'machine learning',
            'algorithm', 'automated', 'neural network', 'deep learning',
            'chatgpt', 'gpt', 'llm', 'language model', 'openai',
            'anthropic', 'deepfake', 'robot', 'generative',
        ]
        ethics_keywords = [
            'bias', 'privacy', 'ethics', 'fairness', 'transparency',
            'accountability', 'discrimination', 'data protection',
            'surveillance', 'regulation', 'governance', 'harm',
            'safety', 'rights', 'consent', 'gdpr', 'eu ai act',
            'risk', 'concern', 'policy', 'misuse', 'abuse',
            'copyright', 'security', 'law', 'legal', 'lawsuit',
            'ban', 'restrict', 'disinformation', 'misinformation',
            'danger', 'threat', 'control', 'oversight', 'compliance',
        ]
        has_ai = any(kw in text for kw in ai_keywords)
        has_ethics = any(kw in text for kw in ethics_keywords)
        return has_ai and has_ethics
    
    def _create_signal_from_entry(self, entry, feed_url: str) -> EthicalSignal:
        """Convert RSS entry to EthicalSignal"""
        
        title = entry.get('title', '')
        summary = entry.get('summary', entry.get('description', ''))[:500]
        link = entry.get('link', '')
        
        signal_id = hashlib.md5(
            f"{link}:{datetime.now().timestamp()}".encode()
        ).hexdigest()[:16]
        
        content = f"{title}: {summary}"
        
        severity = self.calculate_severity(content)
        category = self.categorize_content(content)
        
        return EthicalSignal(
            signal_id=signal_id,
            source=self.source_name,
            content=content,
            vector_embedding=self.vectorize_content(content),
            severity=severity,
            timestamp=datetime.now(),
            category=category,
            metadata={
                'type': 'news',
                'title': title,
                'link': link,
                'feed_source': feed_url,
                'published': entry.get('published', '')
            }
        )
    
    # No mock data fallback — returns empty list on error (see fetch_signals)