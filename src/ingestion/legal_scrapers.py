from typing import List
import hashlib
from datetime import datetime
import logging
import requests
import feedparser
from urllib.parse import quote_plus

from src.ingestion.base import EthicalSignalScraper
from src.models.signal import EthicalSignal
from config.settings import settings

logger = logging.getLogger(__name__)


class LegalAPIScraper(EthicalSignalScraper):
    """Real-time legal case scraper using RSS feeds from legal/regulatory news sources"""
    
    def __init__(self, source_name: str = "Legal News"):
        super().__init__(source_name)
        # Real-time legal/policy RSS feeds — all verified to return entries
        self.rss_feeds = [
            "https://www.eff.org/rss/updates.xml",           # Electronic Frontier Foundation (50 entries)
            "https://artificialintelligenceact.eu/feed/",    # EU AI Act official news (22 entries)
        ]
        
    def fetch_signals(self) -> List[EthicalSignal]:
        """
        Fetch real-time legal cases from RSS feeds of authoritative legal/tech policy sources.
        
        Sources include:
        - Electronic Frontier Foundation (digital rights litigation)
        - International Association of Privacy Professionals (GDPR/privacy cases)
        - Tech Policy Press (AI regulation news)
        - Lawfare (technology law analysis)
        """
        signals = []
        
        # Two-gate filter: content must mention BOTH AI AND an ethics/legal concern
        ai_keywords = [
            'ai', 'artificial intelligence', 'algorithm', 'machine learning',
            'automated', 'facial recognition', 'chatbot', 'deepfake',
            'neural network', 'large language model', 'llm', 'generative',
            'predictive', 'autonomous', 'gpt', 'chatgpt'
        ]
        ethics_keywords = [
            'privacy', 'gdpr', 'data protection', 'surveillance', 'bias',
            'discrimination', 'transparency', 'accountability',
            'regulation', 'lawsuit', 'court', 'enforcement',
            'fine', 'violation', 'ruling', 'decision', 'eu ai act',
            'ethics', 'rights', 'consent', 'harm', 'safety'
        ]
        
        try:
            for feed_url in self.rss_feeds:
                try:
                    # Fetch RSS feed (feedparser doesn't support timeout parameter directly)
                    feed = feedparser.parse(feed_url)
                    
                    if feed.bozo:  # Parse error
                        logger.warning(f"RSS parse error for {feed_url}")
                        continue
                    
                    # Process entries
                    for entry in feed.entries[:25]:  # Check last 25 entries
                        title = self.strip_html(entry.get('title', ''))
                        raw_summary = entry.get('summary', entry.get('description', ''))
                        summary = self.strip_html(raw_summary)
                        link = entry.get('link', '')
                        published = entry.get('published_parsed', entry.get('updated_parsed'))
                        
                        # Combine title and summary for keyword matching
                        content_text = f"{title} {summary}".lower()
                        
                        # Check if content is related to AI ethics/legal issues (dual-gate)
                        has_ai = any(keyword in content_text for keyword in ai_keywords)
                        has_ethics = any(keyword in content_text for keyword in ethics_keywords)
                        
                        if has_ai and has_ethics:
                            # Create clean signal content (no HTML)
                            content = f"{title}: {summary[:300]}" if summary else title
                            
                            # Calculate severity based on keywords
                            severity = self._calculate_severity(content_text)
                            
                            # Parse date
                            if published:
                                timestamp = datetime(*published[:6])
                            else:
                                timestamp = datetime.now()
                            
                            # Create unique ID
                            signal_id = hashlib.md5(
                                f"{feed_url}:{title}:{timestamp.date()}".encode()
                            ).hexdigest()[:16]
                            
                            signal = EthicalSignal(
                                signal_id=signal_id,
                                source=self.source_name,
                                content=content,
                                vector_embedding=self.vectorize_content(content),
                                severity=severity,
                                timestamp=timestamp,
                                category=self.categorize_content(content),
                                metadata={
                                    'type': 'legal',
                                    'source': 'rss_feed',
                                    'url': link,
                                    'feed': feed_url
                                }
                            )
                            signals.append(signal)
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch from {feed_url}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in legal RSS scraper: {e}")
        
        # If we got signals, return them
        if signals:
            self.last_scrape = datetime.now()
            logger.info(f"[OK] Collected {len(signals)} real-time legal cases")
            return signals
        
        # Fallback: No RSS results - log and return empty or use backup
        logger.warning("No legal cases found in RSS feeds")
        return []
    
    def _calculate_severity(self, content: str) -> float:
        """Calculate severity based on legal keywords"""
        content_lower = content.lower()
        
        # Critical severity indicators
        if any(word in content_lower for word in ['violation', 'fine', 'lawsuit', 'banned', 'illegal', 'unconstitutional']):
            return 0.75 + (hash(content) % 20) / 100  # 0.75-0.94
        
        # High severity indicators  
        if any(word in content_lower for word in ['enforcement', 'ruling', 'court', 'investigation', 'breach']):
            return 0.65 + (hash(content) % 15) / 100  # 0.65-0.79
        
        # Medium severity indicators
        if any(word in content_lower for word in ['regulation', 'privacy', 'gdpr', 'transparency', 'accountability']):
            return 0.55 + (hash(content) % 15) / 100  # 0.55-0.69
        
        # Default
        return 0.50 + (hash(content) % 10) / 100  # 0.50-0.59
