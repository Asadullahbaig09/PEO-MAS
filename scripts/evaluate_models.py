"""
Model Evaluation Script - Phase 4

Evaluates trained models on held-out test set and generates comprehensive metrics.
Compares neural models with rule-based baseline.

Usage:
    python scripts/evaluate_models.py --model domain  # Evaluate domain classifier
    python scripts/evaluate_models.py --model anomaly  # Evaluate anomaly detector
    python scripts/evaluate_models.py --model all      # Evaluate all models
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, roc_auc_score,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.models import DomainClassifierNetwork, AnomalyDetectorNetwork
from src.training.trainer import DomainDataset, AnomalyDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, output_dir: str = "output/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"✓ Evaluator initialized on device: {self.device}")
        
        self.domain_names = ['bias', 'privacy', 'transparency', 
                             'accountability', 'safety', 'general']
    
    def create_test_set(
        self, 
        dataset_path: str, 
        test_split: float = 0.15,
        seed: int = 42
    ) -> Tuple[Dataset, Dataset]:
        """
        Create train/test split from combined dataset
        
        Args:
            dataset_path: Path to training dataset
            test_split: Fraction for test set (default 15%)
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        # Determine dataset type
        if 'domain' in str(dataset_path):
            full_dataset = DomainDataset(dataset_path)
        elif 'anomaly' in str(dataset_path):
            full_dataset = AnomalyDataset(dataset_path)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_path}")
        
        # Split indices
        total_size = len(full_dataset)
        test_size = int(total_size * test_split)
        train_size = total_size - test_size
        
        generator = torch.Generator().manual_seed(seed)
        train_indices, test_indices = torch.utils.data.random_split(
            range(total_size), 
            [train_size, test_size],
            generator=generator
        )
        
        train_dataset = Subset(full_dataset, train_indices.indices)
        test_dataset = Subset(full_dataset, test_indices.indices)
        
        logger.info(f"✓ Created test set: {len(test_dataset)} samples ({test_split:.0%})")
        logger.info(f"  Training set: {len(train_dataset)} samples")
        
        return train_dataset, test_dataset
    
    def evaluate_domain_classifier(
        self, 
        model_path: str = "models/trained/domain_classifier_best.pt",
        dataset_path: str = "data/training/combined_training_data/domain_classification_dataset.json"
    ) -> Dict:
        """
        Evaluate domain classifier on test set
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("=" * 60)
        logger.info("EVALUATING DOMAIN CLASSIFIER")
        logger.info("=" * 60)
        
        # Load model
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return {}
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model = DomainClassifierNetwork(input_dim=384, num_classes=6)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"✓ Loaded model from {model_path.name}")
        if 'accuracy' in checkpoint:
            logger.info(f"  Training accuracy: {checkpoint['accuracy']:.3f}")
        if 'loss' in checkpoint:
            logger.info(f"  Training loss: {checkpoint['loss']:.4f}")
        
        # Create test set
        _, test_dataset = self.create_test_set(dataset_path)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Collect predictions
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for embeddings, labels in test_loader:
                embeddings = embeddings.to(self.device)
                outputs = model(embeddings)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = (all_preds == all_labels).mean()
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Per-class metrics (handle missing classes in test set)
        labels_present = sorted(np.unique(all_labels))
        target_names_present = [self.domain_names[i] for i in labels_present]
        
        class_report = classification_report(
            all_labels, all_preds, 
            labels=labels_present,
            target_names=target_names_present,
            output_dict=True
        )
        
        logger.info(f"\n✓ Test Set Performance:")
        logger.info(f"  Accuracy:  {accuracy:.3%}")
        logger.info(f"  Precision: {precision:.3%}")
        logger.info(f"  Recall:    {recall:.3%}")
        logger.info(f"  F1 Score:  {f1:.3%}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        self._plot_confusion_matrix(
            cm, 
            self.domain_names, 
            "domain_classifier_confusion_matrix.png"
        )
        
        # Per-class report
        logger.info(f"\n✓ Per-Domain Performance:")
        for domain in self.domain_names:
            if domain in class_report:
                metrics = class_report[domain]
                logger.info(
                    f"  {domain:15s}: "
                    f"P={metrics['precision']:.2%}  "
                    f"R={metrics['recall']:.2%}  "
                    f"F1={metrics['f1-score']:.2%}  "
                    f"(n={int(metrics['support'])})"
                )
            else:
                logger.info(f"  {domain:15s}: (not in test set)")
        
        # ROC curves (one-vs-rest)
        try:
            self._plot_roc_curves(all_labels, all_probs, self.domain_names, "domain_roc_curves.png")
        except Exception as e:
            logger.warning(f"Could not generate ROC curves: {e}")
        
        # Prepare results
        results = {
            'model': 'domain_classifier',
            'test_samples': len(test_dataset),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'per_class_metrics': class_report,
            'confusion_matrix': cm.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_path = self.output_dir / "domain_classifier_evaluation.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✓ Results saved to {results_path}")
        
        return results
    
    def evaluate_anomaly_detector(
        self,
        model_path: str = "models/trained/anomaly_detector_best.pt",
        dataset_path: str = "data/training/combined_training_data/anomaly_detection_dataset.json"
    ) -> Dict:
        """
        Evaluate anomaly detector on test set
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("=" * 60)
        logger.info("EVALUATING ANOMALY DETECTOR")
        logger.info("=" * 60)
        
        # Load model
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return {}
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model = AnomalyDetectorNetwork(input_dim=384)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"✓ Loaded model from {model_path.name}")
        if 'f1_score' in checkpoint:
            logger.info(f"  Training F1: {checkpoint['f1_score']:.3f}")
        if 'loss' in checkpoint:
            logger.info(f"  Training loss: {checkpoint['loss']:.4f}")
        
        # Create test set
        _, test_dataset = self.create_test_set(dataset_path)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Collect predictions
        all_anomaly_preds = []
        all_anomaly_labels = []
        all_anomaly_probs = []
        all_severity_preds = []
        all_severity_labels = []
        
        with torch.no_grad():
            for embeddings, anomaly_labels, severity_labels in test_loader:
                embeddings = embeddings.to(self.device)
                anomaly_logits, severity_logits = model(embeddings)
                
                # Anomaly predictions
                anomaly_probs = torch.sigmoid(anomaly_logits).squeeze()
                anomaly_preds = (anomaly_probs > 0.5).long()
                
                # Severity predictions
                severity_preds = torch.sigmoid(severity_logits).squeeze()
                
                # Convert to lists (handle both 0-d and 1-d arrays)
                anomaly_preds_np = anomaly_preds.cpu().numpy()
                anomaly_probs_np = anomaly_probs.cpu().numpy()
                severity_preds_np = severity_preds.cpu().numpy()
                
                if anomaly_preds_np.ndim == 0:
                    all_anomaly_preds.append(anomaly_preds_np.item())
                    all_anomaly_probs.append(anomaly_probs_np.item())
                    all_severity_preds.append(severity_preds_np.item())
                else:
                    all_anomaly_preds.extend(anomaly_preds_np)
                    all_anomaly_probs.extend(anomaly_probs_np)
                    all_severity_preds.extend(severity_preds_np)
                
                all_anomaly_labels.extend(anomaly_labels.numpy())
                all_severity_labels.extend(severity_labels.numpy())
        
        all_anomaly_preds = np.array(all_anomaly_preds)
        all_anomaly_labels = np.array(all_anomaly_labels)
        all_anomaly_probs = np.array(all_anomaly_probs)
        all_severity_preds = np.array(all_severity_preds)
        all_severity_labels = np.array(all_severity_labels)
        
        # Anomaly detection metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_anomaly_labels, all_anomaly_preds, average='binary'
        )
        accuracy = (all_anomaly_preds == all_anomaly_labels).mean()
        
        try:
            auc_score = roc_auc_score(all_anomaly_labels, all_anomaly_probs)
        except:
            auc_score = 0.0
        
        logger.info(f"\n✓ Anomaly Detection Performance:")
        logger.info(f"  Accuracy:  {accuracy:.3%}")
        logger.info(f"  Precision: {precision:.3%}")
        logger.info(f"  Recall:    {recall:.3%}")
        logger.info(f"  F1 Score:  {f1:.3%}")
        logger.info(f"  AUC:       {auc_score:.3f}")
        
        # Severity regression metrics (only for anomalies)
        anomaly_mask = all_anomaly_labels == 1
        if anomaly_mask.sum() > 0:
            severity_mae = np.abs(
                all_severity_preds[anomaly_mask] - all_severity_labels[anomaly_mask]
            ).mean()
            severity_rmse = np.sqrt(
                ((all_severity_preds[anomaly_mask] - all_severity_labels[anomaly_mask]) ** 2).mean()
            )
            
            logger.info(f"\n✓ Severity Prediction (for anomalies):")
            logger.info(f"  MAE:  {severity_mae:.4f}")
            logger.info(f"  RMSE: {severity_rmse:.4f}")
        else:
            severity_mae = 0.0
            severity_rmse = 0.0
            logger.warning("No anomalies in test set for severity evaluation")
        
        # Confusion matrix
        cm = confusion_matrix(all_anomaly_labels, all_anomaly_preds)
        self._plot_confusion_matrix(
            cm,
            ['Normal', 'Anomaly'],
            "anomaly_detector_confusion_matrix.png"
        )
        
        # ROC and PR curves
        try:
            self._plot_anomaly_curves(
                all_anomaly_labels, all_anomaly_probs,
                "anomaly_detector_curves.png"
            )
        except Exception as e:
            logger.warning(f"Could not generate curves: {e}")
        
        # Prepare results
        results = {
            'model': 'anomaly_detector',
            'test_samples': len(test_dataset),
            'anomaly_detection': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc': float(auc_score)
            },
            'severity_prediction': {
                'mae': float(severity_mae),
                'rmse': float(severity_rmse)
            },
            'confusion_matrix': cm.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_path = self.output_dir / "anomaly_detector_evaluation.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✓ Results saved to {results_path}")
        
        return results
    
    def compare_with_baseline(self):
        """
        Compare neural models with rule-based baseline
        
        TODO: Implement baseline comparison
        - Load rule-based system results
        - Run same test set through both systems
        - Calculate improvement metrics
        """
        logger.info("=" * 60)
        logger.info("BASELINE COMPARISON")
        logger.info("=" * 60)
        logger.warning("Baseline comparison not yet implemented")
        logger.info("  Next steps:")
        logger.info("  1. Export rule-based system predictions")
        logger.info("  2. Run test set through both systems")
        logger.info("  3. Calculate performance delta")
    
    def _plot_confusion_matrix(self, cm, labels, filename):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Confusion matrix saved: {filename}")
    
    def _plot_roc_curves(self, y_true, y_probs, class_names, filename):
        """Plot ROC curves for multi-class classification"""
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(class_names):
            # One-vs-rest
            y_true_binary = (y_true == i).astype(int)
            y_prob_class = y_probs[:, i]
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob_class)
            auc = roc_auc_score(y_true_binary, y_prob_class)
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC={auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  ROC curves saved: {filename}")
    
    def _plot_anomaly_curves(self, y_true, y_probs, filename):
        """Plot ROC and Precision-Recall curves for anomaly detection"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc = roc_auc_score(y_true, y_probs)
        ax1.plot(fpr, tpr, label=f'AUC={auc:.3f}', linewidth=2)
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        ax2.plot(recall, precision, linewidth=2)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Detection curves saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        '--model',
        choices=['domain', 'anomaly', 'all'],
        default='all',
        help='Which model to evaluate'
    )
    parser.add_argument(
        '--output-dir',
        default='output/evaluation',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(output_dir=args.output_dir)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"MODEL EVALUATION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*60}\n")
    
    results = {}
    
    if args.model in ['domain', 'all']:
        try:
            results['domain'] = evaluator.evaluate_domain_classifier()
        except Exception as e:
            logger.error(f"Domain classifier evaluation failed: {e}", exc_info=True)
    
    if args.model in ['anomaly', 'all']:
        try:
            results['anomaly'] = evaluator.evaluate_anomaly_detector()
        except Exception as e:
            logger.error(f"Anomaly detector evaluation failed: {e}", exc_info=True)
    
    # Baseline comparison
    if args.model == 'all':
        evaluator.compare_with_baseline()
    
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {evaluator.output_dir}")


if __name__ == "__main__":
    main()
