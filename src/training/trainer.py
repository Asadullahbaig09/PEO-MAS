"""
Training orchestrator for all models in the system

Manages training loops for:
1. Domain Classification (embedding fine-tuning)
2. Anomaly Detection (neural network)
3. Legal Recommendation Quality (predictor)
4. RAG Retrieval Reranking (reranker)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np

from src.training.loss_functions import (
    DomainClassificationLoss,
    AnomalyDetectionLoss,
    LegalRecommendationLoss,
    RAGRetrievalLoss,
    MultiTaskLoss
)
from src.training.models import (
    DomainClassifierNetwork,
    AnomalyDetectorNetwork,
    LegalQualityPredictor,
    RAGRerankerNetwork
)
from src.training.data_collector import TrainingDataCollector

logger = logging.getLogger(__name__)


class DomainDataset(Dataset):
    """Dataset for domain classification training"""
    
    def __init__(self, data_path: str, embedding_dim: int = 384):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.embedding_dim = embedding_dim
        self.domain_to_idx = {
            'bias': 0, 'privacy': 1, 'transparency': 2,
            'accountability': 3, 'safety': 4, 'general': 5
        }
        
        # Load pre-computed embeddings if available
        embedding_path = Path(data_path).parent / (Path(data_path).stem + "_embeddings.npy")
        if embedding_path.exists():
            self.embeddings = np.load(embedding_path)
            print(f"✓ Loaded {len(self.embeddings)} pre-computed embeddings")
        else:
            print(f"⚠️  No pre-computed embeddings found at {embedding_path.name}")
            print(f"   Using random embeddings - run prepare_embeddings.py first!")
            self.embeddings = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Use pre-computed embedding or random fallback
        if self.embeddings is not None:
            embedding = torch.from_numpy(self.embeddings[idx]).float()
        else:
            embedding = torch.randn(self.embedding_dim)
        
        # Get domain label from exported data
        domain = item['label'].lower()
        label = self.domain_to_idx.get(domain, 5)  # Default to 'general'
        
        return embedding, label


class AnomalyDataset(Dataset):
    """Dataset for anomaly detection training"""
    
    def __init__(self, data_path: str, embedding_dim: int = 384):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.embedding_dim = embedding_dim
        
        # Load pre-computed embeddings if available
        embedding_path = Path(data_path).parent / (Path(data_path).stem + "_embeddings.npy")
        if embedding_path.exists():
            self.embeddings = np.load(embedding_path)
            print(f"✓ Loaded {len(self.embeddings)} pre-computed embeddings")
        else:
            print(f"⚠️  No pre-computed embeddings found at {embedding_path.name}")
            print(f"   Using random embeddings - run prepare_embeddings.py first!")
            self.embeddings = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Use pre-computed embedding or random fallback
        if self.embeddings is not None:
            embedding = torch.from_numpy(self.embeddings[idx]).float()
        else:
            embedding = torch.randn(self.embedding_dim)
        
        # Get labels from exported data
        is_anomaly = 1.0 if item['label'] else 0.0
        severity = float(item.get('severity', 0.5))
        
        # Return as scalars (model expects batch_size shape, not batch_size x 1)
        return embedding, torch.tensor(is_anomaly), torch.tensor(severity)


class ModelTrainer:
    """Orchestrates training for all models"""
    
    def __init__(
        self,
        data_dir: str = "./data/training",
        model_save_dir: str = "./models/trained",
        embedding_dim: int = 384,
        device: str = None
    ):
        self.data_dir = Path(data_dir)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_dim = embedding_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"✓ Trainer initialized on device: {self.device}")
        
        # Models
        self.domain_classifier = None
        self.anomaly_detector = None
        self.legal_predictor = None
        self.rag_reranker = None
        
        # Loss functions
        self.domain_loss = DomainClassificationLoss(num_classes=6)
        self.anomaly_loss = AnomalyDetectionLoss()
        self.legal_loss = LegalRecommendationLoss()
        self.rag_loss = RAGRetrievalLoss()
        
    def find_latest_export(self) -> Optional[Path]:
        """Find the latest training export directory"""
        export_dirs = list(self.data_dir.glob("training_export_*"))
        if not export_dirs:
            return None
        
        # Sort by timestamp in directory name
        latest = sorted(export_dirs)[-1]
        logger.info(f"Found latest training export: {latest.name}")
        return latest
    
    def find_all_exports(self) -> List[Path]:
        """Find all training export directories"""
        export_dirs = list(self.data_dir.glob("training_export_*"))
        if not export_dirs:
            return []
        
        # Sort by timestamp in directory name
        sorted_dirs = sorted(export_dirs)
        logger.info(f"Found {len(sorted_dirs)} training export directories")
        for d in sorted_dirs:
            logger.info(f"  - {d.name}")
        return sorted_dirs
    
    def load_training_data(self, export_dir: Path = None, combine_all: bool = True) -> Dict[str, Path]:
        """
        Load training data from export directory
        
        Args:
            export_dir: Specific export directory (if None, uses all exports)
            combine_all: If True, combines data from all export directories
        
        Returns:
            Dictionary mapping task names to combined data file paths
        """
        if not combine_all and export_dir is None:
            export_dir = self.find_latest_export()
        
        if not combine_all:
            # Use single export directory (old behavior)
            if export_dir is None:
                raise FileNotFoundError("No training export directories found")
            
            data_files = {
                'domain': export_dir / "domain_classification_dataset.json",
                'anomaly': export_dir / "anomaly_detection_dataset.json",
                'legal': export_dir / "legal_recommendation_dataset.json",
                'rag': export_dir / "rag_retrieval_dataset.json"
            }
        else:
            # Combine all export directories (new behavior)
            export_dirs = self.find_all_exports()
            if not export_dirs:
                raise FileNotFoundError("No training export directories found")
            
            # Combine datasets from all exports
            combined_dir = self.data_dir / "combined_training_data"
            combined_dir.mkdir(exist_ok=True)
            
            for task in ['domain', 'anomaly', 'legal', 'rag']:
                task_files = {
                    'domain': 'domain_classification_dataset.json',
                    'anomaly': 'anomaly_detection_dataset.json',
                    'legal': 'legal_recommendation_dataset.json',
                    'rag': 'rag_retrieval_dataset.json'
                }
                
                combined_data = []
                combined_embeddings = []
                
                for export_dir in export_dirs:
                    data_file = export_dir / task_files[task]
                    embedding_file = export_dir / task_files[task].replace('.json', '_embeddings.npy')
                    
                    if data_file.exists():
                        with open(data_file, 'r') as f:
                            data = json.load(f)
                            combined_data.extend(data)
                        
                        # Load embeddings if available
                        if embedding_file.exists():
                            embeddings = np.load(embedding_file)
                            if len(embeddings) > 0:
                                combined_embeddings.append(embeddings)
                
                # Save combined data
                combined_data_file = combined_dir / task_files[task]
                with open(combined_data_file, 'w') as f:
                    json.dump(combined_data, f, indent=2)
                
                # Save combined embeddings
                if combined_embeddings:
                    combined_embedding_array = np.vstack(combined_embeddings)
                    combined_embedding_file = combined_dir / task_files[task].replace('.json', '_embeddings.npy')
                    np.save(combined_embedding_file, combined_embedding_array)
                    logger.info(f"✓ Combined {len(combined_data)} {task} samples with {len(combined_embedding_array)} embeddings")
                elif combined_data:
                    logger.info(f"✓ Combined {len(combined_data)} {task} samples (no embeddings)")
            
            data_files = {
                'domain': combined_dir / "domain_classification_dataset.json",
                'anomaly': combined_dir / "anomaly_detection_dataset.json",
                'legal': combined_dir / "legal_recommendation_dataset.json",
                'rag': combined_dir / "rag_retrieval_dataset.json"
            }
        
        # Verify files exist
        for task, path in data_files.items():
            if not path.exists():
                logger.warning(f"Missing {task} dataset: {path}")
        
        return data_files
    
    def train_domain_classifier(
        self,
        data_path: Path,
        epochs: int = 20,
        batch_size: int = 16,
        learning_rate: float = 0.001
    ) -> Dict:
        """Train domain classification model"""
        logger.info("=" * 70)
        logger.info("TRAINING DOMAIN CLASSIFIER")
        logger.info("=" * 70)
        
        # Load dataset
        full_dataset = DomainDataset(str(data_path), self.embedding_dim)
        logger.info(f"Loaded {len(full_dataset)} domain samples")
        
        # Split train/val (80/20)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.domain_classifier = DomainClassifierNetwork(
            input_dim=self.embedding_dim,
            num_classes=6
        ).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(self.domain_classifier.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
        
        # Training loop
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch_domain(
                self.domain_classifier,
                train_loader,
                optimizer
            )
            
            # Validate
            val_loss, val_acc = self._validate_domain(
                self.domain_classifier,
                val_loader
            )
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Track history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.3f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model(
                    self.domain_classifier,
                    'domain_classifier_best.pt',
                    {'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc}
                )
                logger.info(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        logger.info(f"✓ Domain classifier training complete!")
        logger.info(f"  Best val loss: {best_val_loss:.4f}")
        
        return history
    
    def train_anomaly_detector(
        self,
        data_path: Path,
        epochs: int = 25,
        batch_size: int = 16,
        learning_rate: float = 0.001
    ) -> Dict:
        """Train anomaly detection model"""
        logger.info("=" * 70)
        logger.info("TRAINING ANOMALY DETECTOR")
        logger.info("=" * 70)
        
        # Load dataset
        full_dataset = AnomalyDataset(str(data_path), self.embedding_dim)
        logger.info(f"Loaded {len(full_dataset)} anomaly samples")
        
        # Split train/val
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.anomaly_detector = AnomalyDetectorNetwork(
            input_dim=self.embedding_dim
        ).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(self.anomaly_detector.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
        
        # Training loop
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        
        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch_anomaly(
                self.anomaly_detector,
                train_loader,
                optimizer
            )
            
            # Validate
            val_loss, val_f1 = self._validate_anomaly(
                self.anomaly_detector,
                val_loader
            )
            
            scheduler.step(val_loss)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val F1: {val_f1:.3f}"
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model(
                    self.anomaly_detector,
                    'anomaly_detector_best.pt',
                    {'epoch': epoch, 'val_loss': val_loss, 'val_f1': val_f1}
                )
                logger.info(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        logger.info(f"✓ Anomaly detector training complete!")
        logger.info(f"  Best val loss: {best_val_loss:.4f}")
        
        return history
    
    def _train_epoch_domain(self, model, loader, optimizer):
        """Train one epoch for domain classification"""
        model.train()
        total_loss = 0
        
        for embeddings, labels in loader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = self.domain_loss(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _validate_domain(self, model, loader):
        """Validate domain classification model"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for embeddings, labels in loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(embeddings)
                loss = self.domain_loss(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(loader), correct / total
    
    def _train_epoch_anomaly(self, model, loader, optimizer):
        """Train one epoch for anomaly detection"""
        model.train()
        total_loss = 0
        
        for embeddings, is_anomaly, severity in loader:
            embeddings = embeddings.to(self.device)
            is_anomaly = is_anomaly.to(self.device)
            severity = severity.to(self.device)
            
            optimizer.zero_grad()
            anomaly_pred, severity_pred = model(embeddings)
            
            loss = self.anomaly_loss(
                anomaly_pred,
                is_anomaly,
                severity_pred,
                severity
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def _validate_anomaly(self, model, loader):
        """Validate anomaly detection model"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for embeddings, is_anomaly, severity in loader:
                embeddings = embeddings.to(self.device)
                is_anomaly = is_anomaly.to(self.device)
                severity = severity.to(self.device)
                
                anomaly_pred, severity_pred = model(embeddings)
                
                loss = self.anomaly_loss(
                    anomaly_pred,
                    is_anomaly,
                    severity_pred,
                    severity
                )
                
                total_loss += loss.item()
                
                # Binary predictions
                binary_pred = (torch.sigmoid(anomaly_pred) > 0.5).float()
                all_preds.extend(binary_pred.cpu().numpy())
                all_labels.extend(is_anomaly.cpu().numpy())
        
        # Calculate F1 score
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return total_loss / len(loader), f1
    
    def _save_model(self, model, filename: str, metadata: Dict):
        """Save model checkpoint"""
        save_path = self.model_save_dir / filename
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }, save_path)
    
    def load_model(self, model, filename: str):
        """Load model checkpoint"""
        load_path = self.model_save_dir / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"✓ Loaded model from {filename}")
        logger.info(f"  Metadata: {checkpoint.get('metadata', {})}")
        
        return checkpoint.get('metadata', {})
