"""
Main training script - runs training loops for all models

Usage:
    python scripts/train_models.py --task domain --epochs 20
    python scripts/train_models.py --task anomaly --epochs 25
    python scripts/train_models.py --task all
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.trainer import ModelTrainer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train AI ethics oversight models")
    parser.add_argument(
        '--task',
        type=str,
        choices=['domain', 'anomaly', 'legal', 'rag', 'all'],
        default='all',
        help='Which model to train'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (default: task-specific)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training (default: 16)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/training',
        help='Directory containing training data exports'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  TRAINING AI ETHICS OVERSIGHT MODELS")
    print("=" * 70)
    print(f"Task: {args.task}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print("=" * 70)
    print()
    
    # Initialize trainer
    trainer = ModelTrainer(data_dir=args.data_dir)
    
    # Load training data
    try:
        data_files = trainer.load_training_data()
        print(f"✓ Loaded training data from: {trainer.find_latest_export().name}\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure you have collected and exported training data:")
        print("  1. Run: python scripts/collect_training_data.py --cycles 5")
        print("  2. Label data: python scripts/label_training_data.py")
        print("  3. Training data will be exported automatically")
        sys.exit(1)
    
    # Train models based on task
    results = {}
    
    if args.task in ['domain', 'all']:
        if data_files['domain'].exists():
            print("\n" + "=" * 70)
            print("TRAINING DOMAIN CLASSIFIER")
            print("=" * 70)
            
            epochs = args.epochs or 20
            history = trainer.train_domain_classifier(
                data_path=data_files['domain'],
                epochs=epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            results['domain'] = history
            
            print(f"\n✓ Domain classifier training complete!")
            print(f"  Final val accuracy: {history['val_accuracy'][-1]:.3f}")
        else:
            print(f"\n⚠️  Skipping domain classification - no data file found")
    
    if args.task in ['anomaly', 'all']:
        if data_files['anomaly'].exists():
            print("\n" + "=" * 70)
            print("TRAINING ANOMALY DETECTOR")
            print("=" * 70)
            
            epochs = args.epochs or 25
            history = trainer.train_anomaly_detector(
                data_path=data_files['anomaly'],
                epochs=epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            results['anomaly'] = history
            
            print(f"\n✓ Anomaly detector training complete!")
            print(f"  Final val F1: {history['val_f1'][-1]:.3f}")
        else:
            print(f"\n⚠️  Skipping anomaly detection - no data file found")
    
    # TODO: Add legal and rag training when more data is collected
    if args.task in ['legal', 'all']:
        print(f"\n⚠️  Legal recommendation training not yet implemented")
        print(f"    (Waiting for more legal recommendation data)")
    
    if args.task in ['rag', 'all']:
        print(f"\n⚠️  RAG reranking training not yet implemented")
        print(f"    (Waiting for more RAG retrieval data)")
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    for task, history in results.items():
        print(f"\n{task.upper()}:")
        print(f"  Epochs trained: {len(history['train_loss'])}")
        print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
        
        if 'val_accuracy' in history:
            print(f"  Best val accuracy: {max(history['val_accuracy']):.3f}")
        if 'val_f1' in history:
            print(f"  Best val F1: {max(history['val_f1']):.3f}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Continue collecting data in parallel:")
    print("   python scripts/collect_training_data.py --cycles 5 --interval 2h")
    print("\n2. Label new data:")
    print("   python scripts/label_training_data.py")
    print("\n3. Retrain with more data for better performance:")
    print("   python scripts/train_models.py --task all --epochs 30")
    print("\n4. Evaluate models on test set (TODO: implement evaluation)")
    print("\n5. Deploy trained models to production system (TODO: implement)")
    print("=" * 70)


if __name__ == "__main__":
    main()
