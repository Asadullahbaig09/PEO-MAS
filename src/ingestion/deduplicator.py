"""
Signal deduplication logic
Prevents processing of duplicate signals across cycles
"""

import hashlib
import logging
from typing import List, Set
from collections import deque
from datetime import datetime, timedelta

from src.models.signal import EthicalSignal

logger = logging.getLogger(__name__)


class SignalDeduplicator:
    """
    Deduplicates signals based on content similarity and exact matches
    Uses rolling window to remember recent signals
    """
    
    def __init__(self, window_size: int = 1000, ttl_hours: int = 24):
        """
        Args:
            window_size: Number of signals to keep in memory
            ttl_hours: How long to remember a signal (hours)
        """
        self.window_size = window_size
        self.ttl_hours = ttl_hours
        
        # Store signal hashes with timestamp
        self.signal_hashes: dict = {}  # hash -> timestamp
        self.content_signatures: Set[str] = set()  # For quick lookup
        
        # Recent signals for similarity check
        self.recent_signals: deque = deque(maxlen=window_size)
    
    def _generate_hash(self, signal: EthicalSignal) -> str:
        """Generate hash for signal content"""
        # Normalize content (lowercase, strip whitespace)
        normalized = signal.content.lower().strip()
        
        # Create hash from content + category
        content_to_hash = f"{normalized}|{signal.category}|{signal.source}"
        return hashlib.sha256(content_to_hash.encode()).hexdigest()
    
    def _get_signature(self, signal: EthicalSignal) -> str:
        """Get quick signature for fast matching (first 100 chars)"""
        return f"{signal.content[:100]}_{signal.category}"
    
    def is_duplicate(self, signal: EthicalSignal) -> bool:
        """
        Check if signal is a duplicate
        Returns True if exact or near-duplicate found
        """
        signal_hash = self._generate_hash(signal)
        signature = self._get_signature(signal)
        
        # Exact match check
        if signal_hash in self.signal_hashes:
            logger.debug(
                f"Duplicate signal detected (exact match): {signal.content[:50]}..."
            )
            return True
        
        # Signature quick check
        if signature in self.content_signatures:
            logger.debug(
                f"Duplicate signal detected (signature match): {signal.content[:50]}..."
            )
            return True
        
        # Check against recent signals for similarity
        for recent_signal in self.recent_signals:
            if self._are_similar(signal, recent_signal):
                logger.debug(
                    f"Similar signal found: {signal.content[:50]}... "
                    f"(previously seen from {recent_signal.source})"
                )
                return True
        
        return False
    
    def _are_similar(self, signal1: EthicalSignal, signal2: EthicalSignal) -> bool:
        """
        Check if two signals are similar
        Same category + very similar content = duplicate
        """
        # Must be same category
        if signal1.category != signal2.category:
            return False
        
        # Calculate simple text similarity (Jaccard)
        words1 = set(signal1.content.lower().split())
        words2 = set(signal2.content.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = intersection / union if union > 0 else 0
        
        # 70% similarity threshold = likely duplicate
        return similarity > 0.7
    
    def add_signal(self, signal: EthicalSignal):
        """Register signal as seen"""
        signal_hash = self._generate_hash(signal)
        signature = self._get_signature(signal)
        
        self.signal_hashes[signal_hash] = datetime.now()
        self.content_signatures.add(signature)
        self.recent_signals.append(signal)
        
        logger.debug(f"Signal registered: {signal.content[:50]}...")
    
    def deduplicate_batch(self, signals: List[EthicalSignal]) -> List[EthicalSignal]:
        """
        Filter out duplicates from a batch of signals
        
        Usage:
            unique_signals = deduplicator.deduplicate_batch(signals)
        """
        unique_signals = []
        duplicates_removed = 0
        
        for signal in signals:
            if not self.is_duplicate(signal):
                unique_signals.append(signal)
                self.add_signal(signal)
            else:
                duplicates_removed += 1
        
        if duplicates_removed > 0:
            logger.info(
                f"Removed {duplicates_removed} duplicate signals "
                f"from batch of {len(signals)}"
            )
        
        return unique_signals
    
    def cleanup_expired(self):
        """Remove expired signals from history"""
        cutoff_time = datetime.now() - timedelta(hours=self.ttl_hours)
        
        expired_hashes = [
            h for h, ts in self.signal_hashes.items()
            if ts < cutoff_time
        ]
        
        for h in expired_hashes:
            del self.signal_hashes[h]
        
        if expired_hashes:
            logger.debug(
                f"Cleaned up {len(expired_hashes)} expired signal hashes"
            )
    
    def get_statistics(self) -> dict:
        """Get deduplicator statistics"""
        return {
            'signals_in_window': len(self.recent_signals),
            'hashes_tracked': len(self.signal_hashes),
            'signatures_tracked': len(self.content_signatures),
            'window_capacity': self.window_size,
            'ttl_hours': self.ttl_hours
        }
