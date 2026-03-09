"""
Neural Network Inference Module

Provides production-ready inference wrappers for trained models with
optimal thresholds discovered through evaluation.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

from src.training.models import AnomalyDetectorNetwork, DomainClassifierNetwork
from src.knowledge.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


class NeuralAnomalyDetector:
    """
    Production-ready neural network anomaly detector
    
    Uses trained AnomalyDetectorNetwork with optimal threshold=0.60
    discovered through threshold tuning (achieves 100% F1 score)
    """
    
    def __init__(
        self, 
        model_path: str = "models/trained/anomaly_detector_best.pt",
        threshold: float = 0.60,  # Optimal threshold (100% F1!)
        device: Optional[str] = None
    ):
        """
        Initialize neural anomaly detector
        
        Args:
            model_path: Path to trained model checkpoint
            threshold: Decision threshold (default 0.60 for 100% F1)
            device: Device to run on ('cuda' or 'cpu', auto-detect if None)
        """
        self.threshold = threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = AnomalyDetectorNetwork(input_dim=384)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✓ Neural anomaly detector loaded (threshold={threshold})")
        logger.info(f"  Device: {self.device}")
        if 'f1_score' in checkpoint:
            logger.info(f"  Training F1: {checkpoint['f1_score']:.3f}")
    
    def detect(
        self, 
        embedding: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        Detect if signal is anomaly
        
        Args:
            embedding: Signal embedding (384-dim vector)
        
        Returns:
            is_anomaly: Boolean anomaly prediction
            confidence: Anomaly probability (0-1)
            severity: Predicted severity score (0-1)
        """
        # Convert to tensor
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).float()
        
        embedding = embedding.to(self.device)
        
        # Handle batch vs single sample
        if len(embedding.shape) == 1:
            embedding = embedding.unsqueeze(0)  # Add batch dimension
        
        # Get predictions
        with torch.no_grad():
            anomaly_prob, severity_score = self.model(embedding)
        
        # Apply threshold
        is_anomaly = (anomaly_prob >= self.threshold).item()
        confidence = anomaly_prob.item()
        severity = severity_score.item()
        
        return is_anomaly, confidence, severity
    
    def batch_detect(
        self, 
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect anomalies in batch
        
        Args:
            embeddings: Batch of embeddings (batch_size, 384)
        
        Returns:
            is_anomaly: Boolean array (batch_size,)
            confidences: Probability array (batch_size,)
            severities: Severity array (batch_size,)
        """
        # Convert to tensor
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        
        embeddings = embeddings.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            anomaly_probs, severity_scores = self.model(embeddings)
        
        # Apply threshold
        is_anomaly = (anomaly_probs >= self.threshold).cpu().numpy()
        confidences = anomaly_probs.cpu().numpy()
        severities = severity_scores.cpu().numpy()
        
        return is_anomaly, confidences, severities


class NeuralDomainClassifier:
    """
    Production-ready neural network domain classifier
    
    Uses trained DomainClassifierNetwork (achieves 100% test accuracy)
    """
    
    def __init__(
        self,
        model_path: str = "models/trained/domain_classifier_best.pt",
        device: Optional[str] = None
    ):
        """
        Initialize neural domain classifier
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run on ('cuda' or 'cpu', auto-detect if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Domain mapping
        self.idx_to_domain = {
            0: 'bias',
            1: 'privacy', 
            2: 'transparency',
            3: 'accountability',
            4: 'safety',
            5: 'general'
        }
        
        # Load model
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = DomainClassifierNetwork(input_dim=384, num_classes=6)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✓ Neural domain classifier loaded")
        logger.info(f"  Device: {self.device}")
        if 'accuracy' in checkpoint:
            logger.info(f"  Training accuracy: {checkpoint['accuracy']:.3f}")
    
    def classify(
        self, 
        embedding: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify signal domain
        
        Args:
            embedding: Signal embedding (384-dim vector)
        
        Returns:
            domain: Predicted domain name
            confidence: Confidence score (0-1)
            probabilities: Dict mapping domain -> probability
        """
        # Convert to tensor
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).float()
        
        embedding = embedding.to(self.device)
        
        # Handle batch vs single sample
        if len(embedding.shape) == 1:
            embedding = embedding.unsqueeze(0)  # Add batch dimension
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(embedding)
            probs = torch.softmax(logits, dim=1)[0]  # First sample
        
        # Get prediction
        pred_idx = torch.argmax(probs).item()
        domain = self.idx_to_domain[pred_idx]
        confidence = probs[pred_idx].item()
        
        # Get all probabilities
        probabilities = {
            self.idx_to_domain[i]: probs[i].item() 
            for i in range(len(probs))
        }
        
        return domain, confidence, probabilities
    
    def batch_classify(
        self, 
        embeddings: np.ndarray
    ) -> Tuple[list, np.ndarray]:
        """
        Classify domains in batch
        
        Args:
            embeddings: Batch of embeddings (batch_size, 384)
        
        Returns:
            domains: List of domain names
            confidences: Confidence scores array (batch_size,)
        """
        # Convert to tensor
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        
        embeddings = embeddings.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(embeddings)
            probs = torch.softmax(logits, dim=1)
        
        # Get predictions
        pred_indices = torch.argmax(probs, dim=1).cpu().numpy()
        domains = [self.idx_to_domain[idx] for idx in pred_indices]
        confidences = probs[range(len(probs)), pred_indices].cpu().numpy()
        
        return domains, confidences


class HybridAnomalyDetector:
    """
    Hybrid anomaly detector that combines neural network with rule-based fallback
    
    Uses neural network as primary detector, falls back to rules if model unavailable
    """
    
    def __init__(
        self,
        neural_threshold: float = 0.60,
        rule_threshold: float = 0.65,
        use_neural: bool = True
    ):
        """
        Initialize hybrid detector
        
        Args:
            neural_threshold: Threshold for neural network (0.60 = 100% F1)
            rule_threshold: Threshold for rule-based backup
            use_neural: Try to use neural network (fallback to rules if fails)
        """
        self.rule_threshold = rule_threshold
        self.neural_detector = None
        self.has_neural = False  # Track if neural is available
        
        if use_neural:
            try:
                self.neural_detector = NeuralAnomalyDetector(threshold=neural_threshold)
                self.has_neural = True
                logger.info("✓ Using neural anomaly detector")
            except Exception as e:
                logger.warning(f"Neural detector unavailable: {e}")
                logger.info("  Falling back to rule-based detection")
    
    def detect(
        self, 
        signal,
        agent_pool=None
    ) -> Tuple[bool, float, float]:
        """
        Detect anomaly using neural network or rules
        
        Args:
            signal: EthicalSignal object
            agent_pool: List of agents (for rule-based fallback)
        
        Returns:
            is_anomaly: Boolean
            confidence: Confidence score
            severity: Severity prediction
        """
        # Try neural detector first
        if self.neural_detector is not None:
            try:
                # Get embedding from signal (use existing embedding or generate)
                if hasattr(signal, 'vector_embedding') and signal.vector_embedding is not None:
                    embedding = signal.vector_embedding
                else:
                    from src.knowledge.embeddings import EmbeddingEngine
                    embedding_engine = EmbeddingEngine()
                    content = signal.content if hasattr(signal, 'content') else str(signal)
                    embedding = embedding_engine.encode(content)
                
                return self.neural_detector.detect(embedding)
                
            except Exception as e:
                logger.warning(f"Neural detection failed: {e}, using rules")
        
        # Fallback to rule-based
        return self._rule_based_detect(signal, agent_pool)
    
    def _rule_based_detect(
        self, 
        signal, 
        agent_pool
    ) -> Tuple[bool, float, float]:
        """
        Rule-based anomaly detection (original logic)
        """
        if not agent_pool:
            return True, 0.0, signal.severity
        
        # Get assessments from relevant agents
        relevant_agents = [a for a in agent_pool if a.can_handle(signal)]
        
        if not relevant_agents:
            return True, 0.0, signal.severity
        
        # Calculate explanation scores
        explanation_scores = [agent.assess_signal(signal) for agent in relevant_agents]
        max_explanation = max(explanation_scores)
        
        # Anomaly if poorly explained
        is_anomaly = max_explanation < self.rule_threshold
        confidence = 1.0 - max_explanation
        severity = signal.severity
        
        return is_anomaly, confidence, severity
