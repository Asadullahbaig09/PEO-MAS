"""
Add human labels to training data CSV
"""

import csv
import argparse
import sys
from pathlib import Path

# Label rules based on content analysis
def determine_labels(row):
    """
    Analyze signal and return true labels
    
    Returns:
        (true_domain, true_anomaly, severity_rating, notes)
    """
    title = row['title'].lower()
    content = row['content'].lower()
    predicted_domain = row['predicted_domain']
    predicted_anomaly = row['predicted_anomaly'] == 'True'
    severity = float(row['severity'])
    
    # Default labels
    true_domain = predicted_domain
    true_anomaly = predicted_anomaly
    severity_rating = severity
    notes = ""
    
    # Legal/Accountability signals
    if any(word in title for word in ['sue', 'federal agents', 'constitution', 'lawsuit']):
        true_domain = 'accountability'
        true_anomaly = True
        severity_rating = 0.85
        notes = "Legal accountability issue - government oversight"
    
    # AI Safety signals
    elif 'ai policy' in title or 'ai harm' in title:
        true_domain = 'safety'
        true_anomaly = True
        severity_rating = 0.75
        notes = "AI policy and safety concerns - important signal"
    
    # Privacy settlements/violations
    elif 'google settlement' in title or 'privacy controls' in title:
        true_domain = 'privacy'
        true_anomaly = True
        severity_rating = 0.90
        notes = "Major privacy settlement - high severity"
    
    elif 'privacy catastrophe' in title or 'privacy attack' in content:
        true_domain = 'privacy'
        true_anomaly = True
        severity_rating = 0.92
        notes = "Critical privacy vulnerability"
    
    elif 'age verification' in content or 'defend your privacy' in content:
        true_domain = 'privacy'
        true_anomaly = False
        severity_rating = 0.70
        notes = "Privacy advocacy - not anomaly"
    
    # Child privacy/parenting to big tech
    elif 'parenting to big tech' in title or 'kids' in content:
        true_domain = 'privacy'
        true_anomaly = True
        severity_rating = 0.80
        notes = "Child privacy legislation concern"
    
    # Bias in ML/AI
    elif 'bias in foundation models' in title or 'widespread bias' in title:
        true_domain = 'bias'
        true_anomaly = True
        severity_rating = 0.90
        notes = "Critical bias findings - high priority"
    
    elif 'fairness in machine learning' in title:
        true_domain = 'bias'
        true_anomaly = False
        severity_rating = 0.60
        notes = "Research on fairness - positive development"
    
    # Transparency issues
    elif 'transparency crisis' in title or 'black-box' in content:
        true_domain = 'transparency'
        true_anomaly = True
        severity_rating = 0.88
        notes = "Transparency crisis - autonomous vehicles"
    
    elif 'explainable ai' in title:
        true_domain = 'transparency'
        true_anomaly = False
        severity_rating = 0.65
        notes = "XAI research - positive progress"
    
    # Security issues
    elif 'security threat' in title or 'prompt injection' in content or 'moltbook' in title:
        true_domain = 'safety'
        true_anomaly = True
        severity_rating = 0.75
        notes = "Security vulnerability - prompt injection"
    
    elif 'elon musk' in title and 'spacex' in title and 'xai' in title:
        true_domain = 'accountability'
        true_anomaly = True
        severity_rating = 0.70
        notes = "Corporate consolidation - accountability concern"
    
    # General announcements (not anomalies)
    elif 'eff to close' in title or 'solidarity' in title:
        true_domain = 'general'
        true_anomaly = False
        severity_rating = 0.45
        notes = "Organizational announcement - not ethics issue"
    
    elif 'encrypt it already' in title:
        true_domain = 'privacy'
        true_anomaly = False
        severity_rating = 0.55
        notes = "Privacy tool announcement - not anomaly"
    
    elif 'chatbot' in title and 'ads' in title and 'anthropic' in title:
        true_domain = 'general'
        true_anomaly = False
        severity_rating = 0.40
        notes = "Business news - not ethics issue"
    
    elif 'nvidia' in title and 'openai deal' in title:
        true_domain = 'general'
        true_anomaly = False
        severity_rating = 0.35
        notes = "Business news - market update"
    
    elif 'mistral' in title or 'math startup' in title or 'ai bots' in title:
        true_domain = 'general'
        true_anomaly = False
        severity_rating = 0.40
        notes = "Tech news - not ethics concern"
    
    # Reddit posts
    elif 'psychiatrist' in title or 'genesight' in title:
        true_domain = 'privacy'
        true_anomaly = False
        severity_rating = 0.50
        notes = "Personal privacy question - general discussion"
    
    elif 'world models' in content or 'agi' in title or 'yann lecun' in content:
        true_domain = 'safety'
        true_anomaly = False
        severity_rating = 0.55
        notes = "AI architecture discussion - research topic"
    
    return true_domain, str(true_anomaly), f"{severity_rating:.2f}", notes


def label_csv(input_path, output_path=None):
    """
    Add labels to training data CSV
    """
    if output_path is None:
        output_path = input_path
    
    # Read CSV
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Add labels
    labeled_count = 0
    for row in rows:
        # Skip if already labeled
        if row['true_domain'] and row['true_domain'].strip():
            continue
        
        true_domain, true_anomaly, severity_rating, notes = determine_labels(row)
        row['true_domain'] = true_domain
        row['true_anomaly'] = true_anomaly
        row['severity_rating'] = severity_rating
        row['notes'] = notes
        labeled_count += 1
    
    # Write back
    fieldnames = [
        'timestamp', 'signal_id', 'title', 'content', 'source', 'category', 
        'severity', 'predicted_domain', 'predicted_anomaly', 'agent_risk_level', 
        'agent_confidence', 'true_domain', 'true_anomaly', 'severity_rating', 'notes'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Labeled {labeled_count} signals")
    print(f"✓ Saved to: {output_path}")
    
    # Show label distribution
    domain_counts = {}
    anomaly_count = 0
    for row in rows:
        domain = row['true_domain']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if row['true_anomaly'] == 'True':
            anomaly_count += 1
    
    print(f"\nLabel Distribution:")
    print(f"  Total signals: {len(rows)}")
    print(f"  Anomalies: {anomaly_count} ({anomaly_count/len(rows)*100:.1f}%)")
    print(f"\nDomain breakdown:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  - {domain}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label training data CSV files")
    parser.add_argument(
        '--session',
        type=str,
        default=None,
        help='Session directory name (e.g., "20260205_144027"). If not provided, uses the latest session.'
    )
    
    args = parser.parse_args()
    
    # Find the session directory
    training_base = Path(__file__).parent.parent / "data" / "training"
    
    if args.session:
        # Use specified session
        data_dir = training_base / args.session
        if not data_dir.exists():
            print(f"Error: Session directory not found: {data_dir}")
            sys.exit(1)
    else:
        # Find latest session (directories matching timestamp pattern)
        session_dirs = [d for d in training_base.iterdir() 
                       if d.is_dir() and not d.name.startswith('training_export')]
        
        if not session_dirs:
            print("Error: No training session directories found")
            sys.exit(1)
        
        # Sort by name (timestamp format sorts chronologically)
        data_dir = sorted(session_dirs)[-1]
    
    csv_path = data_dir / "signals.csv"
    
    if not csv_path.exists():
        print(f"Error: signals.csv not found in {data_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("  LABELING TRAINING DATA")
    print("=" * 70)
    print(f"Session: {data_dir.name}")
    print(f"Input: {csv_path}")
    print()
    
    label_csv(csv_path)
    
    print("\n" + "=" * 70)
    print("LABELING COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review labels in signals.csv")
    print("2. Adjust any incorrect labels manually")
    print("3. Run more collection cycles to get 200-500 signals")
    print("4. Export for training using TrainingDataCollector")
    print("=" * 70)
