"""
Loss Functions for Multi-Agent RAG System Training

Defines loss functions for:
1. Anomaly Detection (Binary Classification)
2. Domain Classification (Multi-class)
3. Legal Recommendation Quality
4. RAG Retrieval Ranking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np


# ==============================================================================
# 1. ANOMALY DETECTION LOSS
# ==============================================================================

class AnomalyDetectionLoss(nn.Module):
    """
    Combined loss for anomaly detection with severity weighting
    
    Combines:
    - Binary Cross-Entropy for anomaly classification
    - MSE for severity prediction
    - Focal loss for handling class imbalance
    """
    
    def __init__(
        self,
        alpha: float = 0.7,  # Weight for BCE loss
        beta: float = 0.3,   # Weight for severity MSE
        gamma: float = 2.0,  # Focal loss focusing parameter
        pos_weight: Optional[float] = None  # Weight for positive class (anomalies)
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(
        self,
        predicted_anomaly: torch.Tensor,  # (batch_size,) - sigmoid outputs
        true_anomaly: torch.Tensor,       # (batch_size,) - 0/1 labels
        predicted_severity: torch.Tensor, # (batch_size,) - predicted severity
        true_severity: torch.Tensor       # (batch_size,) - true severity
    ) -> torch.Tensor:
        """
        Compute combined anomaly detection loss
        
        Args:
            predicted_anomaly: Predicted probability of anomaly (0-1)
            true_anomaly: True anomaly labels (0 or 1)
            predicted_severity: Predicted severity scores (0-1)
            true_severity: True severity scores (0-1)
        
        Returns:
            Combined loss value
        """
        # 1. Focal Binary Cross-Entropy Loss
        bce_loss = F.binary_cross_entropy(
            predicted_anomaly,
            true_anomaly,
            reduction='none'
        )
        
        # Apply focal loss weighting (focus on hard examples)
        pt = torch.where(true_anomaly == 1, predicted_anomaly, 1 - predicted_anomaly)
        focal_weight = (1 - pt) ** self.gamma
        focal_bce = (focal_weight * bce_loss).mean()
        
        # Apply positive class weighting if specified
        if self.pos_weight is not None:
            weights = torch.where(true_anomaly == 1, self.pos_weight, 1.0)
            focal_bce = (weights * focal_bce).mean()
        
        # 2. Severity Prediction Loss (only for true anomalies)
        severity_mask = true_anomaly > 0.5  # Only compute for anomalies
        if severity_mask.sum() > 0:
            severity_loss = F.mse_loss(
                predicted_severity[severity_mask],
                true_severity[severity_mask]
            )
        else:
            severity_loss = torch.tensor(0.0, device=predicted_anomaly.device)
        
        # Combined loss
        total_loss = self.alpha * focal_bce + self.beta * severity_loss
        
        return total_loss


# ==============================================================================
# 2. DOMAIN CLASSIFICATION LOSS
# ==============================================================================

class DomainClassificationLoss(nn.Module):
    """
    Loss for multi-class domain classification with label smoothing
    
    Supports:
    - Standard Cross-Entropy
    - Label Smoothing (helps with overconfidence)
    - Class weighting (handles imbalanced domains)
    """
    
    def __init__(
        self,
        num_classes: int = 6,  # bias, privacy, transparency, accountability, safety, general
        smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.class_weights = class_weights
        
    def forward(
        self,
        logits: torch.Tensor,      # (batch_size, num_classes)
        targets: torch.Tensor      # (batch_size,) - class indices
    ) -> torch.Tensor:
        """
        Compute domain classification loss with label smoothing
        
        Args:
            logits: Raw model outputs (before softmax)
            targets: True class indices (0 to num_classes-1)
        
        Returns:
            Loss value
        """
        # Apply label smoothing
        if self.smoothing > 0:
            # Create smoothed target distribution
            confidence = 1.0 - self.smoothing
            smooth_value = self.smoothing / (self.num_classes - 1)
            
            # One-hot encode targets
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, targets.unsqueeze(1), 1)
            
            # Apply smoothing
            smooth_targets = one_hot * confidence + smooth_value * (1 - one_hot)
            
            # Compute KL divergence
            log_probs = F.log_softmax(logits, dim=1)
            loss = -(smooth_targets * log_probs).sum(dim=1)
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Apply class weights if provided
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            loss = loss * weights
        
        return loss.mean()


class ContrastiveDomainLoss(nn.Module):
    """
    Contrastive loss for learning better domain embeddings
    
    Brings embeddings of same domain closer, pushes different domains apart
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        embeddings: torch.Tensor,  # (batch_size, embedding_dim)
        labels: torch.Tensor       # (batch_size,) - domain labels
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss
        
        Args:
            embeddings: Signal embeddings
            labels: Domain labels
        
        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs (same domain)
        batch_size = embeddings.size(0)
        labels = labels.view(-1, 1)
        mask = (labels == labels.T).float()
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=embeddings.device)
        
        # Compute log probabilities
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Mean of positive pairs
        mean_log_prob = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        
        # Loss is negative mean log probability
        loss = -mean_log_prob.mean()
        
        return loss


# ==============================================================================
# 3. LEGAL RECOMMENDATION QUALITY LOSS
# ==============================================================================

class LegalRecommendationLoss(nn.Module):
    """
    Multi-task loss for legal recommendation quality
    
    Combines:
    - Quality score regression (overall quality)
    - Pairwise ranking loss (prefer high-quality over low-quality)
    - Diversity penalty (avoid repetitive recommendations)
    """
    
    def __init__(
        self,
        alpha: float = 0.5,  # Quality regression weight
        beta: float = 0.3,   # Ranking loss weight
        gamma: float = 0.2   # Diversity weight
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(
        self,
        predicted_quality: torch.Tensor,    # (batch_size,) - predicted quality 1-5
        true_quality: torch.Tensor,         # (batch_size,) - human ratings 1-5
        embeddings: Optional[torch.Tensor] = None  # (batch_size, dim) - text embeddings
    ) -> torch.Tensor:
        """
        Compute legal recommendation quality loss
        
        Args:
            predicted_quality: Model's quality predictions
            true_quality: Human quality ratings
            embeddings: Optional text embeddings for diversity
        
        Returns:
            Combined loss
        """
        # 1. Quality Regression Loss (MSE)
        quality_loss = F.mse_loss(predicted_quality, true_quality)
        
        # 2. Pairwise Ranking Loss
        # For each pair, if quality_i > quality_j, then pred_i should be > pred_j
        batch_size = predicted_quality.size(0)
        if batch_size > 1:
            # Create all pairs
            quality_diff = true_quality.unsqueeze(1) - true_quality.unsqueeze(0)
            pred_diff = predicted_quality.unsqueeze(1) - predicted_quality.unsqueeze(0)
            
            # Only consider pairs with significant quality difference
            valid_pairs = torch.abs(quality_diff) > 0.5
            
            # Ranking loss: pred_diff should match sign of quality_diff
            ranking_loss = F.relu(1.0 - quality_diff.sign() * pred_diff)
            ranking_loss = (ranking_loss * valid_pairs.float()).sum() / (valid_pairs.sum() + 1e-6)
        else:
            ranking_loss = torch.tensor(0.0, device=predicted_quality.device)
        
        # 3. Diversity Loss (encourage different recommendations)
        if embeddings is not None and batch_size > 1:
            # Normalize embeddings
            norm_emb = F.normalize(embeddings, p=2, dim=1)
            
            # Compute pairwise similarity
            similarity = torch.matmul(norm_emb, norm_emb.T)
            
            # Remove diagonal
            mask = 1 - torch.eye(batch_size, device=embeddings.device)
            
            # Penalize high similarity (want diverse recommendations)
            diversity_loss = (similarity * mask).sum() / (mask.sum() + 1e-6)
        else:
            diversity_loss = torch.tensor(0.0, device=predicted_quality.device)
        
        # Combined loss
        total_loss = (
            self.alpha * quality_loss +
            self.beta * ranking_loss +
            self.gamma * diversity_loss
        )
        
        return total_loss


# ==============================================================================
# 4. RAG RETRIEVAL RANKING LOSS
# ==============================================================================

class RAGRetrievalLoss(nn.Module):
    """
    Loss for improving RAG document retrieval ranking
    
    Uses:
    - Pairwise ranking loss (relevant docs should rank higher)
    - Listwise loss (consider full ranking)
    """
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        scores: torch.Tensor,        # (batch_size, num_docs) - retrieval scores
        relevance: torch.Tensor      # (batch_size, num_docs) - binary relevance labels
    ) -> torch.Tensor:
        """
        Compute retrieval ranking loss
        
        Args:
            scores: Predicted relevance scores for documents
            relevance: True binary relevance (1=relevant, 0=not relevant)
        
        Returns:
            Ranking loss
        """
        batch_size, num_docs = scores.size()
        
        # Pairwise ranking loss
        # For each relevant doc, it should score higher than irrelevant docs
        loss = 0.0
        count = 0
        
        for i in range(batch_size):
            relevant_mask = relevance[i] == 1
            irrelevant_mask = relevance[i] == 0
            
            if relevant_mask.sum() > 0 and irrelevant_mask.sum() > 0:
                relevant_scores = scores[i][relevant_mask]
                irrelevant_scores = scores[i][irrelevant_mask]
                
                # All relevant docs should score higher than all irrelevant docs
                for rel_score in relevant_scores:
                    for irrel_score in irrelevant_scores:
                        # Margin-based ranking loss
                        loss += F.relu(self.margin - (rel_score - irrel_score))
                        count += 1
        
        return loss / (count + 1e-6)


class ListwiseRankingLoss(nn.Module):
    """
    Listwise ranking loss using ListNet approach
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        scores: torch.Tensor,     # (batch_size, num_docs)
        relevance: torch.Tensor   # (batch_size, num_docs)
    ) -> torch.Tensor:
        """
        Compute listwise ranking loss using cross-entropy
        
        Args:
            scores: Predicted scores
            relevance: True relevance scores (can be continuous)
        
        Returns:
            Listwise loss
        """
        # Convert to probability distributions
        pred_probs = F.softmax(scores, dim=1)
        true_probs = F.softmax(relevance, dim=1)
        
        # KL divergence between distributions
        loss = -(true_probs * torch.log(pred_probs + 1e-10)).sum(dim=1).mean()
        
        return loss


# ==============================================================================
# 5. MULTI-TASK COMBINED LOSS
# ==============================================================================

class MultiTaskLoss(nn.Module):
    """
    Combined loss for training all components jointly
    
    Allows end-to-end training with automatic task weighting
    """
    
    def __init__(
        self,
        task_weights: Optional[dict] = None,
        learnable_weights: bool = False
    ):
        super().__init__()
        
        # Default weights
        self.task_weights = task_weights or {
            'anomaly': 1.0,
            'domain': 1.0,
            'legal': 0.5,
            'rag': 0.5
        }
        
        # Learnable task weights (uncertainty weighting)
        if learnable_weights:
            self.log_vars = nn.ParameterDict({
                task: nn.Parameter(torch.zeros(1))
                for task in self.task_weights.keys()
            })
        else:
            self.log_vars = None
    
    def forward(self, losses: dict) -> Tuple[torch.Tensor, dict]:
        """
        Combine multiple task losses
        
        Args:
            losses: Dict of task_name -> loss_value
        
        Returns:
            total_loss: Combined loss
            weighted_losses: Dict of weighted individual losses
        """
        total_loss = 0.0
        weighted_losses = {}
        
        for task, loss in losses.items():
            if task in self.task_weights:
                if self.log_vars is not None:
                    # Uncertainty-based weighting
                    precision = torch.exp(-self.log_vars[task])
                    weighted_loss = precision * loss + self.log_vars[task]
                else:
                    # Fixed weighting
                    weighted_loss = self.task_weights[task] * loss
                
                total_loss += weighted_loss
                weighted_losses[task] = weighted_loss
        
        return total_loss, weighted_losses


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute balanced class weights for handling imbalanced datasets
    
    Args:
        labels: Array of class labels
        num_classes: Total number of classes
    
    Returns:
        Tensor of class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.arange(num_classes)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    
    return torch.FloatTensor(weights)


def compute_positive_weight(labels: np.ndarray) -> float:
    """
    Compute weight for positive class in binary classification
    
    Args:
        labels: Binary labels (0/1)
    
    Returns:
        Weight for positive class
    """
    num_pos = (labels == 1).sum()
    num_neg = (labels == 0).sum()
    
    if num_pos == 0:
        return 1.0
    
    return num_neg / num_pos
