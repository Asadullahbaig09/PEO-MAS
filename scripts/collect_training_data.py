"""
Run system with training data collection enabled

Usage:
    python scripts/collect_training_data.py --cycles 5
    python scripts/collect_training_data.py --cycles 10 --session "session_001"
    python scripts/collect_training_data.py --cycles 3 --interval 2h
    
Note: --interval specifies the TOTAL time window. It will be divided by the number
of cycles to determine wait time between cycles. For example, --cycles 3 --interval 2h
will run 3 cycles over 2 hours (one cycle every 40 minutes).
"""

import sys
import argparse
import time
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.system import PerpetualEthicalOversightMAS
from src.training.data_collector import TrainingDataCollector
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_interval(interval_str: str) -> int:
    """
    Parse interval string to seconds
    
    Args:
        interval_str: Interval like '2h', '30m', '90s'
    
    Returns:
        Interval in seconds
    """
    if not interval_str:
        return 0
    
    # Match pattern like '2h', '30m', '90s'
    match = re.match(r'^(\d+)([hms])$', interval_str.lower())
    if not match:
        raise ValueError(f"Invalid interval format: {interval_str}. Use format like '2h', '30m', '90s'")
    
    value = int(match.group(1))
    unit = match.group(2)
    
    if unit == 'h':
        return value * 3600
    elif unit == 'm':
        return value * 60
    elif unit == 's':
        return value
    
    return 0


def run_with_data_collection(cycles: int = 5, session_name: str = None, interval_seconds: int = 0):
    """
    Run system and collect training data
    
    Args:
        cycles: Number of cycles to run
        session_name: Optional session identifier
        interval_seconds: Total time window in seconds (divided by cycles for per-cycle wait)
    """
    # Calculate per-cycle wait time
    wait_per_cycle = 0
    if interval_seconds > 0 and cycles > 1:
        wait_per_cycle = interval_seconds // (cycles - 1)  # Divide by (cycles - 1) since no wait after last cycle
    
    print("=" * 70)
    print("  TRAINING DATA COLLECTION MODE")
    print("=" * 70)
    print(f"Cycles: {cycles}")
    print(f"Session: {session_name or 'auto-generated'}")
    if interval_seconds > 0:
        # Show total time window
        total_hours = interval_seconds // 3600
        total_minutes = (interval_seconds % 3600) // 60
        total_seconds = interval_seconds % 60
        total_str = ""
        if total_hours > 0:
            total_str += f"{total_hours}h "
        if total_minutes > 0:
            total_str += f"{total_minutes}m "
        if total_seconds > 0 or not total_str:
            total_str += f"{total_seconds}s"
        print(f"Total Time Window: {total_str.strip()}")
        
        # Show per-cycle wait time
        if wait_per_cycle > 0:
            wait_hours = wait_per_cycle // 3600
            wait_minutes = (wait_per_cycle % 3600) // 60
            wait_seconds = wait_per_cycle % 60
            wait_str = ""
            if wait_hours > 0:
                wait_str += f"{wait_hours}h "
            if wait_minutes > 0:
                wait_str += f"{wait_minutes}m "
            if wait_seconds > 0 or not wait_str:
                wait_str += f"{wait_seconds}s"
            print(f"Wait Between Cycles: {wait_str.strip()}")
    print("=" * 70)
    print()
    
    # Initialize data collector
    collector = TrainingDataCollector()
    
    # Initialize system
    system = PerpetualEthicalOversightMAS()
    
    # Run cycles with data collection
    for cycle in range(cycles):
        print(f"\n{'=' * 70}")
        print(f"CYCLE {cycle + 1}/{cycles}")
        print(f"{'=' * 70}\n")
        
        # Run system cycle
        signals = system.ingestion.collect_signals()
        
        print(f"[COLLECTION] Collected {len(signals)} signals for training data")
        
        # Process signals and collect data
        for signal in signals:
            # Detect anomalies
            agent_pool = list(system.agent_registry.agents.values())
            anomaly = system.anomaly_detector.detect_anomaly(signal, agent_pool)
            is_anomaly = anomaly is not None
            
            # Determine domain
            predicted_domain = signal.category
            
            # Get agent assessment if available
            agent_assessment = None
            if anomaly and hasattr(anomaly, 'assessment'):
                agent_assessment = {
                    'risk_level': anomaly.assessment.risk_level,
                    'confidence': anomaly.assessment.confidence
                }
            
            # Convert signal to dict for recording
            signal_dict = {
                'id': signal.signal_id,
                'title': signal.metadata.get('title', signal.content[:100]),
                'content': signal.content,
                'source': signal.source,
                'category': signal.category,
                'severity': signal.severity
            }
            
            # Record signal data
            collector.record_signal(
                signal=signal_dict,
                predicted_domain=predicted_domain,
                is_anomaly=is_anomaly,
                agent_assessment=agent_assessment
            )
            
            # Record domain prediction
            text = f"{signal.metadata.get('title', '')} {signal.content}"
            collector.record_domain_prediction(
                text=text,
                predicted_domain=predicted_domain
            )
        
        # Check for legal recommendations
        if anomaly and hasattr(system, 'law_generator') and system.law_generator:
            if hasattr(anomaly, 'legal_recommendation') and anomaly.legal_recommendation:
                collector.record_legal_recommendation(
                    signal=signal_dict,
                    recommendation=anomaly.legal_recommendation
                )
        
        # Show progress
        stats = collector.get_labeling_stats()
        print(f"\n[PROGRESS] Collected: {stats['total_signals']} signals, "
              f"{stats['total_legal_recs']} legal recs")
        
        # Wait between cycles (except after last cycle)
        if wait_per_cycle > 0 and cycle < cycles - 1:
            print(f"\n[WAITING] Sleeping for {wait_per_cycle} seconds before next cycle...")
            time.sleep(wait_per_cycle)
    
    # Save collected data
    print(f"\n{'=' * 70}")
    print("SAVING TRAINING DATA")
    print(f"{'=' * 70}\n")
    
    session_dir = collector.save_session(session_name)
    
    # Export for training
    exported = collector.export_for_training('all')
    
    print(f"\n{'=' * 70}")
    print("COLLECTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nData saved to: {session_dir}")
    print("\nExported training datasets:")
    for task, path in exported.items():
        print(f"  - {task}: {path}")
    
    # Show labeling stats
    stats = collector.get_labeling_stats()
    print(f"\nLabeling Progress:")
    print(f"  - Signals: {stats['labeling_progress']['signals']}")
    print(f"  - Legal Recommendations: {stats['labeling_progress']['legal']}")
    print(f"  - RAG Retrievals: {stats['labeling_progress']['rag']}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Review and label data in CSV files:")
    print(f"   {session_dir}/signals.csv")
    print(f"   {session_dir}/legal_recommendations.csv")
    print("\n2. Add human labels:")
    print("   - true_domain: correct domain for each signal")
    print("   - true_anomaly: True/False for anomaly classification")
    print("   - quality_rating: 1-5 rating for legal recommendations")
    print("\n3. Use labeled data for training:")
    print("   - Fine-tune embedding model (domain classification)")
    print("   - Train anomaly detector (ML-based)")
    print("   - Fine-tune LLM (legal recommendations)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect training data from system runs")
    parser.add_argument(
        '--cycles',
        type=int,
        default=5,
        help='Number of cycles to run (default: 5)'
    )
    parser.add_argument(
        '--session',
        type=str,
        default=None,
        help='Session name (default: auto-generated timestamp)'
    )
    parser.add_argument(
        '--interval',
        type=str,
        default=None,
        help='Total time window for all cycles (e.g., "2h", "30m", "90s"). Will be divided by number of cycles.'
    )
    
    args = parser.parse_args()
    
    # Parse interval if provided
    interval_seconds = 0
    if args.interval:
        try:
            interval_seconds = parse_interval(args.interval)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    run_with_data_collection(
        cycles=args.cycles,
        session_name=args.session,
        interval_seconds=interval_seconds
    )
