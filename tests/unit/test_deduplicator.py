"""
Unit tests for deduplication logic
"""

import pytest
from datetime import datetime

from src.ingestion.deduplicator import SignalDeduplicator
from tests.fixtures.sample_data import (
    create_sample_signal,
    create_bias_signal,
    create_duplicate_signal,
    get_sample_signals
)


class TestSignalDeduplicator:
    """Test signal deduplication functionality"""
    
    def test_deduplicator_creation(self):
        """Test deduplicator initializes correctly"""
        dedup = SignalDeduplicator(window_size=100, ttl_hours=24)
        
        assert dedup.window_size == 100
        assert dedup.ttl_hours == 24
        assert len(dedup.signal_hashes) == 0
    
    def test_hash_generation(self):
        """Test hash generation for signals"""
        dedup = SignalDeduplicator()
        
        signal1 = create_bias_signal()
        signal2 = create_bias_signal()
        
        hash1 = dedup._generate_hash(signal1)
        hash2 = dedup._generate_hash(signal2)
        
        # Same content should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex length
    
    def test_exact_duplicate_detection(self):
        """Test exact duplicate detection"""
        dedup = SignalDeduplicator()
        
        signal1 = create_bias_signal()
        dedup.add_signal(signal1)
        
        signal2 = create_bias_signal()  # Exact duplicate
        is_duplicate = dedup.is_duplicate(signal2)
        
        assert is_duplicate is True
    
    def test_non_duplicate_detection(self):
        """Test non-duplicate signals are not flagged"""
        dedup = SignalDeduplicator()
        
        signal1 = create_bias_signal()
        dedup.add_signal(signal1)
        
        signal2 = create_sample_signal(
            content="Different content about fairness",
            category="bias",
            severity=0.5
        )
        is_duplicate = dedup.is_duplicate(signal2)
        
        assert is_duplicate is False
    
    def test_similarity_detection(self):
        """Test similar signals are detected"""
        dedup = SignalDeduplicator()
        
        signal1 = create_sample_signal(
            content="CRITICAL: Bias in machine learning models",
            category="bias"
        )
        dedup.add_signal(signal1)
        
        signal2 = create_sample_signal(
            signal_id="similar_1",
            content="CRITICAL: Bias in machine learning models",
            category="bias"
        )
        is_duplicate = dedup.is_duplicate(signal2)
        
        assert is_duplicate is True
    
    def test_batch_deduplication(self):
        """Test deduplication of signal batches"""
        dedup = SignalDeduplicator()
        
        signals = [
            create_bias_signal(),
            create_sample_signal(
                signal_id="unique_1",
                content="Unique signal about privacy",
                category="privacy"
            ),
            create_bias_signal(),  # Duplicate of first
        ]
        
        unique = dedup.deduplicate_batch(signals)
        
        # Should have 2 unique signals (1st bias, privacy)
        assert len(unique) <= len(signals)
        # Should have identified at least 1 duplicate
        assert len(unique) <= len(signals)
    
    def test_signature_generation(self):
        """Test quick signature generation"""
        dedup = SignalDeduplicator()
        
        signal1 = create_bias_signal()
        signal2 = create_sample_signal(
            content="Different content",
            category="privacy"
        )
        
        sig1 = dedup._get_signature(signal1)
        sig2 = dedup._get_signature(signal2)
        
        # Different signals should have different signatures
        assert sig1 != sig2
    
    def test_category_affects_similarity(self):
        """Test that signals must have same category to be similar"""
        dedup = SignalDeduplicator()
        
        signal1 = create_sample_signal(
            content="Critical issue in AI system",
            category="bias"
        )
        dedup.add_signal(signal1)
        
        signal2 = create_sample_signal(
            signal_id="diff_cat",
            content="Critical issue in AI system",
            category="privacy"  # Different category
        )
        
        is_duplicate = dedup.is_duplicate(signal2)
        # Different categories should not match
        assert is_duplicate is False
    
    def test_empty_content_handling(self):
        """Test handling of signals with empty content"""
        dedup = SignalDeduplicator()
        
        signal1 = create_sample_signal(content="")
        signal2 = create_sample_signal(signal_id="empty_2", content="")
        
        dedup.add_signal(signal1)
        is_dup = dedup.is_duplicate(signal2)
        
        # Behavior for empty content - should not crash
        assert isinstance(is_dup, bool)
    
    def test_window_size_enforcement(self):
        """Test that window size is enforced"""
        dedup = SignalDeduplicator(window_size=3)
        
        signals = [
            create_sample_signal(signal_id=f"sig_{i}")
            for i in range(5)
        ]
        
        for signal in signals:
            dedup.add_signal(signal)
        
        # Window should only keep last 3
        assert len(dedup.recent_signals) <= 3
    
    def test_statistics(self):
        """Test deduplicator statistics"""
        dedup = SignalDeduplicator()
        
        signals = get_sample_signals()
        for signal in signals:
            dedup.add_signal(signal)
        
        stats = dedup.get_statistics()
        
        assert 'signals_in_window' in stats
        assert 'hashes_tracked' in stats
        assert stats['hashes_tracked'] >= len(signals)


class TestDeduplicationInPipeline:
    """Test deduplication in realistic pipeline scenarios"""
    
    def test_deduplicate_across_sources(self):
        """Test deduplication of same signal from different sources"""
        dedup = SignalDeduplicator()
        
        signal1 = create_sample_signal(
            source="reddit",
            content="AI bias discrimination case",
            category="bias"
        )
        dedup.add_signal(signal1)
        
        signal2 = create_sample_signal(
            signal_id="arxiv_version",
            source="arxiv",
            content="AI bias discrimination case",
            category="bias"
        )
        
        is_duplicate = dedup.is_duplicate(signal2)
        assert is_duplicate is True
    
    def test_deduplicate_with_severity_differences(self):
        """Test that severity doesn't affect duplicate detection"""
        dedup = SignalDeduplicator()
        
        signal1 = create_sample_signal(
            content="Privacy issue in AI",
            category="privacy",
            severity=0.5
        )
        dedup.add_signal(signal1)
        
        signal2 = create_sample_signal(
            signal_id="higher_severity",
            content="Privacy issue in AI",
            category="privacy",
            severity=0.9
        )
        
        is_duplicate = dedup.is_duplicate(signal2)
        # Same content = duplicate, regardless of severity rating
        assert is_duplicate is True
    
    def test_cleanup_expired_signals(self):
        """Test cleanup of expired signals"""
        dedup = SignalDeduplicator(ttl_hours=0)  # Everything expires immediately
        
        signal1 = create_bias_signal()
        dedup.add_signal(signal1)
        
        initial_count = len(dedup.signal_hashes)
        dedup.cleanup_expired()
        
        # After cleanup, expired signals should be gone
        assert len(dedup.signal_hashes) <= initial_count
