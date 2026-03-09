"""
Main entry point for running the Perpetual Ethical Oversight MAS

Usage:
    python scripts/run_system.py --cycles 5
    python scripts/run_system.py --continuous --interval 60
"""

import argparse
import time
import json
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.system import PerpetualEthicalOversightMAS
from config.settings import settings
from src.utils.logging import setup_logging


def signal_handler(sig, frame):
    """Handle graceful shutdown"""
    print("\n\nShutting down gracefully...")
    sys.exit(0)


def run_single_cycle(system: PerpetualEthicalOversightMAS, cycle_num: int) -> bool:
    """Run a single processing cycle"""
    print(f"\n{'='*70}")
    print(f"CYCLE {cycle_num} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    try:
        system.process_cycle()
        return True
    except Exception as e:
        print(f"\n✗ Error in cycle {cycle_num}: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_continuous(system: PerpetualEthicalOversightMAS, interval: int):
    """Run system continuously with specified interval"""
    cycle = 1
    
    print(f"\n🔄 Running continuously (interval: {interval}s)")
    print("Press Ctrl+C to stop\n")
    
    while True:
        run_single_cycle(system, cycle)
        
        print(f"\n⏸  Waiting {interval} seconds until next cycle...")
        time.sleep(interval)
        cycle += 1


def export_results(system: PerpetualEthicalOversightMAS, output_file: str):
    """Export system state to file"""
    state = system.export_system_state()
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(state, indent=2, fp=f, default=str)
    
    print(f"\n✓ System state exported to: {output_file}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Run Perpetual Ethical Oversight MAS"
    )
    parser.add_argument(
        '--cycles',
        type=int,
        default=3,
        help='Number of cycles to run (default: 3)'
    )
    parser.add_argument(
        '--continuous',
        action='store_true',
        help='Run continuously'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=settings.PROCESSING_INTERVAL_SECONDS,
        help='Interval between cycles in seconds (default: from config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/system_state.json',
        help='Output file for system state export'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default=settings.LOG_LEVEL,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(level=args.log_level, log_file=settings.LOG_FILE)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Print banner
    print("\n" + "="*70)
    print("  PERPETUAL ETHICAL OVERSIGHT MULTI-AGENT SYSTEM")
    print("  Self-Evolving AI Governance Framework")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Coverage Threshold: {settings.ETHICAL_COVERAGE_THRESHOLD}")
    print(f"  Anomaly Threshold: {settings.ANOMALY_SEVERITY_THRESHOLD}")
    print(f"  Decay Rate: {settings.DECAY_RATE}")
    print(f"  Log Level: {args.log_level}")
    print()
    
    # Initialize system
    print("Initializing system...")
    try:
        system = PerpetualEthicalOversightMAS()
        print("✓ System initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run based on mode
    if args.continuous:
        run_continuous(system, args.interval)
    else:
        # Run specified number of cycles
        success_count = 0
        for i in range(1, args.cycles + 1):
            if run_single_cycle(system, i):
                success_count += 1
            
            # Wait between cycles (except last)
            if i < args.cycles:
                time.sleep(2)
        
        # Export results
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"Completed {success_count}/{args.cycles} cycles successfully")
        
        export_results(system, args.output)
        
        # Export legal recommendations to separate file
        if system.anomaly_detector and hasattr(system.anomaly_detector, 'legal_recommendations'):
            legal_recs = system.anomaly_detector.legal_recommendations
            if legal_recs:
                legal_output = Path('output/legal_recommendations.json')
                legal_output.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert to serializable format
                from dataclasses import asdict
                recs_data = [asdict(rec) for rec in legal_recs]
                
                with open(legal_output, 'w') as f:
                    json.dump(recs_data, f, indent=2, default=str)
                
                print(f"\n✓ Legal recommendations exported to: {legal_output}")
        
        # Print final metrics
        metrics = system.metrics
        print(f"\nFinal Metrics:")
        print(f"  Total Signals Processed: {metrics['signals_processed']}")
        print(f"  Anomalies Detected: {metrics['anomalies_detected']}")
        if system.law_checker:
            print(f"  Legal Recommendations Generated: {metrics.get('legal_recommendations_generated', 0)}")
            print(f"  Laws Checked: {metrics.get('laws_checked', 0)}")
        
        print("\n✓ Execution completed\n")


if __name__ == "__main__":
    main()

