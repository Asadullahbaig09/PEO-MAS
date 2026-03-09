from typing import List
import hashlib
from datetime import datetime
import requests
import logging
import time

from src.ingestion.base import EthicalSignalScraper
from src.models.signal import EthicalSignal

logger = logging.getLogger(__name__)


class RedditScraper(EthicalSignalScraper):
    """Real Reddit scraper using FREE public JSON API"""
    
    def __init__(self):
        super().__init__("Reddit")
        
        # FIXED: Add platform attribute
        self.platform = "Reddit"
        
        # AI Ethics-focused subreddits (all verified to return 200)
        self.subreddits = [
            'artificial',
            'MachineLearning',
            'privacy',
            'AIethics',
            'ResponsibleAI',
            'ChatGPT',
            'OpenAI',
            'dataprivacy',
            'deeplearning',
        ]
        
        self.headers = {
            'User-Agent': 'EthicalOversightMAS/1.0 (Research Project)'
        }
        
        self.last_request_time = 0
        self.rate_limit_delay = 2
    
    def fetch_signals(self) -> List[EthicalSignal]:
        """Fetch real posts from ALL subreddits"""
        
        signals = []
        seen_ids: set = set()
        
        for subreddit in self.subreddits:
            try:
                posts = self._fetch_subreddit_posts(subreddit, sort='hot', limit=15)
                relevant_posts = [p for p in posts if self._is_ai_ethics_relevant(p)]
                
                for post in relevant_posts:
                    post_id = post.get('id', '')
                    if post_id in seen_ids:
                        continue
                    seen_ids.add(post_id)
                    signal = self._create_signal_from_post(post)
                    if signal:
                        signals.append(signal)
                
            except Exception as e:
                logger.error(f"[ERR] Error fetching from r/{subreddit}: {e}")
                continue
        
        logger.info(f"[OK] Collected {len(signals)} signals from {len(self.subreddits)} subreddits")
        self.last_scrape = datetime.now()
        return signals
    
    def _fetch_subreddit_posts(self, subreddit: str, sort: str = 'hot', limit: int = 10) -> List[dict]:
        """Fetch posts from subreddit"""
        
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={limit}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                posts = data['data']['children']
                return [post['data'] for post in posts]
            else:
                logger.warning(f"Reddit returned status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching from r/{subreddit}: {e}")
            return []
    
    def _is_ai_ethics_relevant(self, post: dict) -> bool:
        """Check if post is relevant"""
        
        title = post.get('title', '').lower()
        selftext = post.get('selftext', '').lower()
        combined = f"{title} {selftext}"
        
        ai_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'ml',
            'deep learning', 'neural network', 'algorithm', 'automated',
            'chatgpt', 'gpt', 'llm', 'language model', 'openai',
            'anthropic', 'deepfake', 'robot', 'generative',
        ]
        
        has_ai = any(keyword in combined for keyword in ai_keywords)
        
        if not has_ai:
            return False
        
        ethics_keywords = [
            'bias', 'ethical', 'privacy', 'fairness', 'discrimination',
            'transparency', 'accountability', 'harm', 'safety', 'risk',
            'concern', 'problem', 'issue', 'danger', 'threat', 'abuse',
            'misuse', 'regulation', 'governance', 'copyright', 'security',
            'law', 'legal', 'lawsuit', 'ban', 'restrict', 'disinformation',
            'misinformation', 'control', 'oversight', 'compliance', 'policy',
        ]
        
        has_ethics = any(keyword in combined for keyword in ethics_keywords)
        return has_ethics
    
    def _create_signal_from_post(self, post: dict) -> EthicalSignal:
        """Convert Reddit post to signal"""
        
        title = post.get('title', '')
        selftext = post.get('selftext', '')[:500]
        url = f"https://www.reddit.com{post.get('permalink', '')}"
        score = post.get('score', 0)
        num_comments = post.get('num_comments', 0)
        
        content = f"{title}"
        if selftext:
            content += f": {selftext}"
        
        signal_id = hashlib.md5(
            f"reddit:{post.get('id', '')}".encode()
        ).hexdigest()[:16]
        
        base_severity = self.calculate_severity(content)
        
        if score > 100:
            base_severity += 0.1
        if num_comments > 50:
            base_severity += 0.1
        
        severity = min(1.0, base_severity)
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
                'type': 'social',
                'platform': 'Reddit',
                'url': url,
                'subreddit': post.get('subreddit', ''),
                'score': score,
                'num_comments': num_comments,
                'created_utc': post.get('created_utc', 0),
                'author': post.get('author', '[deleted]'),
                'real_data': True
            }
        )

