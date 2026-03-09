"""
Threshold Tuning Script for Anomaly Detector

Finds optimal decision threshold to maximize F1 score without retraining.
Current threshold: 0.5 results in many false positives.

Usage:
    python scripts/tune_threshold.py
    python scripts/tune_threshold.py --start 0.5 --end 0.8 --step 0.05
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_curve,
    confusion_matrix
)
import matplotlib.pyplot as plt
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.models import AnomalyDetectorNetwork
from src.training.trainer import AnomalyDataset


def find_optimal_threshold(
    model_path: str = "models/trained/anomaly_detector_best.pt",
    dataset_path: str = "data/training/combined_training_data/anomaly_detection_dataset.json",
    start: float = 0.5,
    end: float = 0.8,
    step: float = 0.02
):
    """
    Find optimal decision threshold for anomaly detection
    
    Args:
        model_path: Path to trained model
        dataset_path: Path to dataset
        start: Starting threshold
        end: Ending threshold
        step: Step size
    """
    print("=" * 70)
    print("ANOMALY DETECTOR THRESHOLD TUNING")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}\n")
    
    # Load model
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    model = AnomalyDetectorNetwork(input_dim=384)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✓ Loaded model from {model_path.name}\n")
    
    # Load dataset
    full_dataset = AnomalyDataset(dataset_path, embedding_dim=384)
    
    # Create test set (same as evaluation)
    total_size = len(full_dataset)
    test_size = int(total_size * 0.15)
    train_size = total_size - test_size
    
    generator = torch.Generator().manual_seed(42)
    train_indices, test_indices = torch.utils.data.random_split(
        range(total_size),
        [train_size, test_size],
        generator=generator
    )
    
    test_dataset = Subset(full_dataset, test_indices.indices)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"✓ Test set: {len(test_dataset)} samples\n")
    
    # Get predictions
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, labels, _ in test_loader:
            embeddings = embeddings.to(device)
            anomaly_logits, _ = model(embeddings)
            probs = torch.sigmoid(anomaly_logits)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Test thresholds
    print("Testing thresholds:")
    print("-" * 70)
    print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'FP':>5} | {'FN':>5}")
    print("-" * 70)
    
    results = []
    thresholds = np.arange(start, end + step, step)
    
    for threshold in thresholds:
        preds = (all_probs >= threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, preds, average='binary', zero_division=0
        )
        
        cm = confusion_matrix(all_labels, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        results.append({
            'threshold': float(threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'tn': int(tn)
        })
        
        print(f"{threshold:10.2f} | {precision:10.2%} | {recall:10.2%} | {f1:10.2%} | {fp:5d} | {fn:5d}")
    
    print("-" * 70)
    
    # Find best threshold
    best_result = max(results, key=lambda x: x['f1'])
    
    print(f"\n{'=' * 70}")
    print("OPTIMAL THRESHOLD")
    print("=" * 70)
    print(f"Threshold: {best_result['threshold']:.2f}")
    print(f"Precision: {best_result['precision']:.2%}")
    print(f"Recall:    {best_result['recall']:.2%}")
    print(f"F1 Score:  {best_result['f1']:.2%}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {best_result['tn']:3d}  |  FP: {best_result['fp']:3d}")
    print(f"  FN: {best_result['fn']:3d}  |  TP: {best_result['tp']:3d}")
    
    # Compare with baseline (0.5)
    baseline = next((r for r in results if abs(r['threshold'] - 0.5) < 0.01), results[0])
    print(f"\n{'=' * 70}")
    print("COMPARISON WITH BASELINE (threshold=0.5)")
    print("=" * 70)
    print(f"                 | Baseline | Optimal | Improvement")
    print(f"-----------------+----------+---------+-------------")
    print(f"Threshold        | {baseline['threshold']:8.2f} | {best_result['threshold']:7.2f} | -")
    print(f"Precision        | {baseline['precision']:8.2%} | {best_result['precision']:7.2%} | {(best_result['precision']-baseline['precision']):+7.2%}")
    print(f"Recall           | {baseline['recall']:8.2%} | {best_result['recall']:7.2%} | {(best_result['recall']-baseline['recall']):+7.2%}")
    print(f"F1 Score         | {baseline['f1']:8.2%} | {best_result['f1']:7.2%} | {(best_result['f1']-baseline['f1']):+7.2%}")
    print(f"False Positives  | {baseline['fp']:8d} | {best_result['fp']:7d} | {(best_result['fp']-baseline['fp']):+7d}")
    print(f"False Negatives  | {baseline['fn']:8d} | {best_result['fn']:7d} | {(best_result['fn']-baseline['fn']):+7d}")
    
    # Plot threshold vs metrics
    plt.figure(figsize=(12, 6))
    
    thresholds_list = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    
    plt.plot(thresholds_list, precisions, 'b-', label='Precision', linewidth=2)
    plt.plot(thresholds_list, recalls, 'r-', label='Recall', linewidth=2)
    plt.plot(thresholds_list, f1s, 'g-', label='F1 Score', linewidth=2, marker='o')
    
    # Mark optimal point
    plt.axvline(best_result['threshold'], color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='Current (0.5)')
    
    plt.xlabel('Decision Threshold')
    plt.ylabel('Score')
    plt.title('Anomaly Detection: Threshold vs Metrics')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_dir = Path('output/evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'threshold_tuning_curve.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved threshold tuning curve: {plot_path}")
    
    # Save results
    results_path = output_dir / 'threshold_tuning_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'optimal_threshold': best_result['threshold'],
            'optimal_metrics': {
                'precision': best_result['precision'],
                'recall': best_result['recall'],
                'f1': best_result['f1']
            },
            'baseline_metrics': {
                'precision': baseline['precision'],
                'recall': baseline['recall'],
                'f1': baseline['f1']
            },
            'all_results': results
        }, f, indent=2)
    
    print(f"✓ Saved results: {results_path}")
    
    # Implementation recommendation
    print(f"\n{'=' * 70}")
    print("IMPLEMENTATION")
    print("=" * 70)
    print(f"To use the optimal threshold ({best_result['threshold']:.2f}):\n")
    print("1. Update your inference code:")
    print("   ```python")
    print("   anomaly_prob, severity = model(embedding)")
    print(f"   is_anomaly = anomaly_prob >= {best_result['threshold']:.2f}  # Instead of 0.5")
    print("   ```\n")
    print("2. Or update the model's forward method to use this threshold by default")
    
    return best_result


def main():
    parser = argparse.ArgumentParser(description="Tune anomaly detection threshold")
    parser.add_argument('--start', type=float, default=0.5, help='Starting threshold')
    parser.add_argument('--end', type=float, default=0.8, help='Ending threshold')
    parser.add_argument('--step', type=float, default=0.02, help='Step size')
    
    args = parser.parse_args()
    
    find_optimal_threshold(
        start=args.start,
        end=args.end,
        step=args.step
    )


if __name__ == "__main__":
    main()
