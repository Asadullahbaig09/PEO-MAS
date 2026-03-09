"""
ArXiv Academic Paper Scraper — Real API

Fetches AI ethics-relevant papers from the ArXiv API (http://export.arxiv.org/api).
No mock data — returns empty list if the API is unreachable.
"""

from typing import List
import hashlib
import logging
import random
from datetime import datetime
from xml.etree import ElementTree

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from src.ingestion.base import EthicalSignalScraper
from src.models.signal import EthicalSignal

logger = logging.getLogger(__name__)

# ArXiv Atom namespace
ATOM_NS = '{http://www.w3.org/2005/Atom}'
ARXIV_NS = '{http://arxiv.org/schemas/atom}'


class ArXivScraper(EthicalSignalScraper):
    """Fetches AI-ethics papers from the real ArXiv API"""

    def __init__(self):
        super().__init__("ArXiv")
        self.base_url = "http://export.arxiv.org/api/query"

        # ArXiv category-based queries (AND queries return 0 on current API)
        # cs.CY = Computers & Society, cs.AI = AI, cs.LG = Machine Learning,
        # cs.CL = Computation & Language (NLP/LLMs), cs.CR = Crypto & Security
        self.queries = [
            "cat:cs.CY",   # Computers & Society — ethics, policy, fairness
            "cat:cs.AI",   # Artificial Intelligence
            "cat:cs.LG",   # Machine Learning
            "cat:cs.CL",   # Computation & Language (LLMs, NLP bias)
            "cat:cs.CR",   # Cryptography & Security
        ]

    def fetch_signals(self) -> List[EthicalSignal]:
        """Fetch papers from ArXiv API — iterates ALL category queries."""

        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not installed — cannot call ArXiv API")
            return []

        signals: List[EthicalSignal] = []
        seen_ids: set = set()

        for query in self.queries:
            params = {
                "search_query": query,
                "start": 0,
                "max_results": 10,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }

            try:
                logger.info(f"Fetching ArXiv papers: {query}")
                resp = requests.get(self.base_url, params=params, timeout=20)
                resp.raise_for_status()
            except Exception as e:
                logger.error(f"ArXiv API request failed for {query}: {e}")
                continue

            try:
                root = ElementTree.fromstring(resp.text)
            except ElementTree.ParseError as e:
                logger.error(f"ArXiv XML parse error: {e}")
                continue

            entries = root.findall(f"{ATOM_NS}entry")
            for entry in entries:
                # Deduplicate by ArXiv ID
                id_el = entry.find(f"{ATOM_NS}id")
                arxiv_id = (id_el.text or "").strip() if id_el is not None else ""
                if arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)

                signal = self._entry_to_signal(entry)
                if signal is not None:
                    signals.append(signal)

            # Small delay between API calls to be polite
            import time
            time.sleep(1)

        logger.info(f"ArXiv: fetched {len(signals)} relevant papers from {len(self.queries)} categories")
        self.last_scrape = datetime.now()
        return signals

    def _entry_to_signal(self, entry):
        """Convert one Atom <entry> to an EthicalSignal."""
        title_el = entry.find(f"{ATOM_NS}title")
        summary_el = entry.find(f"{ATOM_NS}summary")
        link_el = entry.find(f"{ATOM_NS}id")
        published_el = entry.find(f"{ATOM_NS}published")

        title = (title_el.text or "").strip().replace("\n", " ") if title_el is not None else ""
        abstract = (summary_el.text or "").strip().replace("\n", " ")[:500] if summary_el is not None else ""
        link = (link_el.text or "").strip() if link_el is not None else ""
        published = (published_el.text or "").strip() if published_el is not None else ""

        if not title:
            return None

        content = f"{title}: {abstract}" if abstract else title

        # Only keep papers relevant to AI ethics
        if not self._is_relevant(content):
            return None

        signal_id = hashlib.md5(
            f"arxiv:{link}:{datetime.now().timestamp()}".encode()
        ).hexdigest()[:16]

        # Get arXiv category
        primary_cat_el = entry.find(f"{ARXIV_NS}primary_category")
        arxiv_category = primary_cat_el.attrib.get("term", "") if primary_cat_el is not None else ""

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
                "type": "research",
                "title": title,
                "link": link,
                "arxiv_category": arxiv_category,
                "published": published,
            },
        )

    @staticmethod
    def _is_relevant(text: str) -> bool:
        """Quick keyword filter — keep only AI-ethics papers."""
        text_lower = text.lower()
        ethics_keywords = [
            "ethic", "bias", "fairness", "privacy", "transparency",
            "accountability", "discrimination", "safety", "harm",
            "regulation", "governance", "explainab", "interpretab",
            "audit", "compliance", "rights", "surveillance",
            "security", "adversarial", "robustness",
        ]
        return any(kw in text_lower for kw in ethics_keywords)

