"""
Neural Network Models for Training

Defines trainable models for:
1. Anomaly Detector
2. Domain Classifier
3. Legal Quality Predictor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ==============================================================================
# 1. ANOMALY DETECTOR MODEL
# ==============================================================================

class AnomalyDetectorNetwork(nn.Module):
    """
    Neural network for anomaly detection
    
    Replaces threshold-based anomaly detection with learned classifier
    """
    
    def __init__(
        self,
        input_dim: int = 384,  # SentenceTransformer embedding dim
        hidden_dims: list = [256, 128, 64],
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Two heads: anomaly classification + severity regression
        self.anomaly_head = nn.Linear(prev_dim, 1)  # Binary classification
        self.severity_head = nn.Linear(prev_dim, 1)  # Severity prediction
        
    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            embeddings: Signal embeddings (batch_size, input_dim)
        
        Returns:
            anomaly_prob: Anomaly probability (batch_size,)
            severity_score: Predicted severity (batch_size,)
        """
        features = self.feature_extractor(embeddings)
        
        # Anomaly classification
        anomaly_logit = self.anomaly_head(features).squeeze(-1)
        anomaly_prob = torch.sigmoid(anomaly_logit)
        
        # Severity prediction (0-1 range)
        severity_logit = self.severity_head(features).squeeze(-1)
        severity_score = torch.sigmoid(severity_logit)
        
        return anomaly_prob, severity_score


# ==============================================================================
# 2. DOMAIN CLASSIFIER MODEL
# ==============================================================================

class DomainClassifierNetwork(nn.Module):
    """
    Neural network for domain classification
    
    Can be used as:
    1. Standalone classifier
    2. Fine-tuning head on top of embedding model
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        num_classes: int = 6,  # bias, privacy, transparency, accountability, safety, general
        hidden_dims: list = [256, 128],
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        
    def forward(
        self,
        embeddings: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            embeddings: Signal embeddings (batch_size, input_dim)
            return_features: If True, return features + logits
        
        Returns:
            logits: Class logits (batch_size, num_classes)
            features (optional): Extracted features
        """
        features = self.feature_extractor(embeddings)
        logits = self.classifier(features)
        
        if return_features:
            return logits, features
        return logits


# ==============================================================================
# 3. LEGAL QUALITY PREDICTOR MODEL
# ==============================================================================

class LegalQualityPredictor(nn.Module):
    """
    Neural network for predicting legal recommendation quality
    
    Takes signal + generated recommendation embeddings -> quality score
    """
    
    def __init__(
        self,
        signal_dim: int = 384,
        law_dim: int = 384,
        hidden_dims: list = [512, 256, 128],
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Combine signal and law embeddings
        input_dim = signal_dim + law_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Multiple quality aspects
        self.overall_quality = nn.Linear(prev_dim, 1)
        self.relevance = nn.Linear(prev_dim, 1)
        self.legal_soundness = nn.Linear(prev_dim, 1)
        self.clarity = nn.Linear(prev_dim, 1)
        
    def forward(
        self,
        signal_embeddings: torch.Tensor,
        law_embeddings: torch.Tensor
    ) -> dict:
        """
        Forward pass
        
        Args:
            signal_embeddings: Signal embeddings (batch_size, signal_dim)
            law_embeddings: Generated law embeddings (batch_size, law_dim)
        
        Returns:
            Dict with quality predictions (all in 1-5 range)
        """
        # Concatenate embeddings
        combined = torch.cat([signal_embeddings, law_embeddings], dim=1)
        
        # Extract features
        features = self.feature_extractor(combined)
        
        # Predict quality aspects (scale to 1-5 range)
        quality = torch.sigmoid(self.overall_quality(features)) * 4 + 1
        relevance = torch.sigmoid(self.relevance(features)) * 4 + 1
        soundness = torch.sigmoid(self.legal_soundness(features)) * 4 + 1
        clarity = torch.sigmoid(self.clarity(features)) * 4 + 1
        
        return {
            'quality': quality.squeeze(-1),
            'relevance': relevance.squeeze(-1),
            'legal_soundness': soundness.squeeze(-1),
            'clarity': clarity.squeeze(-1)
        }


# ==============================================================================
# 4. RAG RETRIEVAL RERANKER MODEL
# ==============================================================================

class RAGRerankerNetwork(nn.Module):
    """
    Neural reranker for improving RAG retrieval
    
    Takes query + document embeddings -> relevance score
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dims: list = [256, 128, 64],
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Interaction features between query and document
        input_dim = embedding_dim * 3  # query, doc, element-wise product
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.scorer = nn.Linear(prev_dim, 1)
        
    def forward(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            query_embeddings: Query embeddings (batch_size, embedding_dim)
            doc_embeddings: Document embeddings (batch_size, embedding_dim)
        
        Returns:
            relevance_scores: Relevance scores (batch_size,)
        """
        # Create interaction features
        element_wise = query_embeddings * doc_embeddings
        combined = torch.cat([query_embeddings, doc_embeddings, element_wise], dim=1)
        
        # Extract features and score
        features = self.feature_extractor(combined)
        scores = self.scorer(features).squeeze(-1)
        
        return scores


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def initialize_weights(model: nn.Module):
    """
    Initialize model weights using best practices
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_layers(model: nn.Module, freeze_until: Optional[str] = None):
    """
    Freeze layers for transfer learning
    
    Args:
        model: Model to freeze layers in
        freeze_until: Name of layer to freeze until (None = freeze all)
    """
    freeze = True
    
    for name, param in model.named_parameters():
        if freeze_until and freeze_until in name:
            freeze = False
        
        param.requires_grad = not freeze
