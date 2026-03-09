"""
Pre-compute embeddings for all training data

This script:
1. Loads training data from latest export
2. Computes embeddings for all text samples
3. Saves embeddings alongside datasets for fast training

Usage:
    python scripts/prepare_embeddings.py
    python scripts/prepare_embeddings.py --session 20260209_185541
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge.embeddings import EmbeddingEngine


def find_latest_export(data_dir: Path) -> Path:
    """Find the latest training export directory"""
    export_dirs = list(data_dir.glob("training_export_*"))
    if not export_dirs:
        raise FileNotFoundError("No training export directories found")
    
    latest = sorted(export_dirs)[-1]
    print(f"✓ Found latest export: {latest.name}")
    return latest


def prepare_embeddings_for_dataset(
    dataset_path: Path,
    embedding_engine: EmbeddingEngine,
    output_path: Path
):
    """
    Compute embeddings for a dataset and save to disk
    
    Args:
        dataset_path: Path to dataset JSON file
        embedding_engine: Embedding model
        output_path: Where to save embeddings (.npy file)
    """
    if not dataset_path.exists():
        print(f"⚠️  Skipping {dataset_path.name} - file not found")
        return
    
    print(f"\n🔄 Processing {dataset_path.name}...")
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"   Loaded {len(data)} samples")
    
    # Extract text from all samples
    texts = [item['text'] for item in data]
    
    # Compute embeddings (batch processing for efficiency)
    print(f"   Computing embeddings...")
    embeddings = embedding_engine.encode(texts)
    
    # Save embeddings
    np.save(output_path, embeddings)
    print(f"   ✓ Saved embeddings to {output_path.name}")
    print(f"   Shape: {embeddings.shape}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings for training data")
    parser.add_argument(
        '--session',
        type=str,
        default=None,
        help='Specific training session (e.g., 20260209_185541). Uses ALL sessions if not specified.'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/training',
        help='Directory containing training exports'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  PRE-COMPUTING TRAINING EMBEDDINGS")
    print("=" * 70)
    
    # Find export directory(ies)
    data_dir = Path(args.data_dir)
    
    if args.session:
        # Process single session
        export_dir = data_dir / f"training_export_{args.session}"
        if not export_dir.exists():
            print(f"❌ Error: Export directory not found: {export_dir}")
            sys.exit(1)
        print(f"✓ Using specified export: {export_dir.name}")
        export_dirs = [export_dir]
    else:
        # Process ALL sessions
        export_dirs = sorted(list(data_dir.glob("training_export_*")))
        if not export_dirs:
            print(f"❌ Error: No training export directories found in {data_dir}")
            sys.exit(1)
        print(f"✓ Found {len(export_dirs)} training export directories:")
        for d in export_dirs:
            print(f"  - {d.name}")
    
    # Initialize embedding model
    print("\n🔄 Loading embedding model...")
    embedding_engine = EmbeddingEngine(cache_dir="./models")
    
    # Process each export directory
    total_samples = 0
    for export_dir in export_dirs:
        print(f"\n{'='*70}")
        print(f"Processing: {export_dir.name}")
        print('='*70)
        
        # Process each dataset
        datasets = [
            'domain_classification_dataset.json',
            'anomaly_detection_dataset.json',
            'legal_recommendation_dataset.json',
            'rag_retrieval_dataset.json'
        ]
        
        for dataset_name in datasets:
            dataset_path = export_dir / dataset_name
            # Save embeddings with same name but .npy extension
            embedding_path = export_dir / dataset_name.replace('.json', '_embeddings.npy')
            
            if dataset_path.exists():
                # Count samples before processing
                with open(dataset_path) as f:
                    sample_count = len(json.load(f))
                total_samples += sample_count
            
            prepare_embeddings_for_dataset(
                dataset_path,
                embedding_engine,
                embedding_path
            )
    
    print("\n" + "=" * 70)
    print("✅ EMBEDDINGS PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Total samples processed: {total_samples}")
    print("\nNext steps:")
    print("1. Train models with ALL combined data:")
    print("   python scripts/train_models.py --task all --epochs 30")
    print("\n2. Expect better performance with more training data!")
    print("=" * 70)


if __name__ == "__main__":
    main()
