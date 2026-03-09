# Phase 2: Loss Functions Guide

## Overview

This guide explains the loss functions designed for training your Multi-Agent RAG System components. Each loss function is optimized for a specific task with considerations for class imbalance, quality metrics, and ranking.

---

## 1. Anomaly Detection Loss

### **AnomalyDetectionLoss**

**Purpose:** Train neural network to replace threshold-based anomaly detection

**Components:**
```python
Total Loss = α × Focal_BCE + β × Severity_MSE

Where:
- Focal_BCE: Handles class imbalance (few anomalies vs many normal signals)
- Severity_MSE: Predicts how severe the anomaly is (0.0-1.0)
- α = 0.7, β = 0.3 (default weights)
```

**Why Focal Loss?**
- Your dataset likely has **imbalanced classes** (90% normal, 10% anomalies)
- Focal loss focuses on **hard-to-classify examples**
- Reduces loss for easy examples, increases for hard ones

**Mathematical Formulation:**
```
Focal_BCE = -(1 - p_t)^γ × log(p_t)

Where:
- p_t = predicted probability for true class
- γ = 2.0 (focusing parameter)
```

**Usage Example:**
```python
from src.training.loss_functions import AnomalyDetectionLoss

# Initialize loss
criterion = AnomalyDetectionLoss(
    alpha=0.7,      # Weight for classification
    beta=0.3,       # Weight for severity
    gamma=2.0,      # Focal loss parameter
    pos_weight=3.0  # Anomalies are 3x more important
)

# Compute loss
loss = criterion(
    predicted_anomaly=model_output_probs,  # (batch_size,)
    true_anomaly=labels,                   # (batch_size,)
    predicted_severity=severity_preds,     # (batch_size,)
    true_severity=severity_labels          # (batch_size,)
)
```

**When to Use:**
- Training anomaly detector from scratch
- Fine-tuning existing detector with labeled data
- Handling **class imbalance** (few anomalies)

**Expected Metrics:**
- Precision: 0.75-0.85 (avoid false alarms)
- Recall: 0.80-0.90 (catch real anomalies)
- F1-Score: 0.77-0.87

---

## 2. Domain Classification Loss

### **DomainClassificationLoss**

**Purpose:** Train domain classifier (bias, privacy, transparency, safety, accountability, general)

**Components:**
```python
Loss = CrossEntropy with Label Smoothing + Class Weighting

Label Smoothing:
- Prevents overconfident predictions
- Improves generalization
- Target distribution: 0.9 for true class, 0.02 for others
```

**Why Label Smoothing?**
- Some signals belong to **multiple domains** (privacy + transparency)
- Prevents model from being 100% certain
- Reduces overfitting on ambiguous examples

**Mathematical Formulation:**
```
Smoothed_Target[true_class] = 1 - ε
Smoothed_Target[other_classes] = ε / (num_classes - 1)

Where ε = 0.1 (smoothing factor)
```

**Usage Example:**
```python
from src.training.loss_functions import DomainClassificationLoss, compute_class_weights
import numpy as np

# Compute class weights from training data
train_labels = np.array([0, 1, 1, 2, 2, 2, 3, 4, 5, 5])  # Your domain labels
class_weights = compute_class_weights(train_labels, num_classes=6)

# Initialize loss
criterion = DomainClassificationLoss(
    num_classes=6,
    smoothing=0.1,           # Label smoothing factor
    class_weights=class_weights  # Handle imbalanced domains
)

# Compute loss
loss = criterion(
    logits=model_outputs,     # (batch_size, 6)
    targets=domain_labels     # (batch_size,)
)
```

**Class Mapping:**
```python
DOMAIN_CLASSES = {
    0: 'bias',
    1: 'privacy',
    2: 'transparency',
    3: 'accountability',
    4: 'safety',
    5: 'general'
}
```

**Expected Metrics:**
- Accuracy: 0.75-0.85
- Macro F1: 0.72-0.82 (balanced across all domains)
- Per-class recall: >0.70 for all domains

---

### **ContrastiveDomainLoss**

**Purpose:** Learn better domain embeddings (alternative/additional loss)

**How It Works:**
- Pulls embeddings of **same domain** closer together
- Pushes embeddings of **different domains** apart
- Uses **supervised contrastive learning**

**Usage Example:**
```python
from src.training.loss_functions import ContrastiveDomainLoss

criterion = ContrastiveDomainLoss(
    margin=1.0,
    temperature=0.07
)

# Use with embedding model
loss = criterion(
    embeddings=signal_embeddings,  # (batch_size, 384)
    labels=domain_labels            # (batch_size,)
)
```

**When to Use:**
- Fine-tuning SentenceTransformer embeddings
- Learning domain-specific representations
- Improving domain separation in embedding space

---

## 3. Legal Recommendation Quality Loss

### **LegalRecommendationLoss**

**Purpose:** Predict quality of generated legal recommendations (1-5 rating)

**Components:**
```python
Total Loss = α × Quality_MSE + β × Ranking_Loss + γ × Diversity_Loss

Where:
- Quality_MSE: Regression for quality score
- Ranking_Loss: High-quality recs should score higher than low-quality
- Diversity_Loss: Avoid repetitive recommendations
```

**Why Multi-Component?**
1. **MSE alone** doesn't capture ranking preference
2. **Ranking loss** ensures relative ordering is correct
3. **Diversity** prevents model from always generating same recommendation

**Usage Example:**
```python
from src.training.loss_functions import LegalRecommendationLoss

criterion = LegalRecommendationLoss(
    alpha=0.5,   # Quality regression
    beta=0.3,    # Ranking
    gamma=0.2    # Diversity
)

loss = criterion(
    predicted_quality=model_preds,     # (batch_size,) - 1 to 5
    true_quality=human_ratings,        # (batch_size,) - 1 to 5
    embeddings=law_text_embeddings     # (batch_size, 384)
)
```

**Quality Metrics:**
```python
Quality Aspects:
- Overall Quality: 1-5 (general usefulness)
- Relevance: 1-5 (addresses the issue)
- Legal Soundness: 1-5 (legally accurate)
- Clarity: 1-5 (clear writing)
```

**Expected Performance:**
- Mean Absolute Error (MAE): <0.5 (within 0.5 points)
- Ranking Accuracy: >0.80 (correct relative order)
- Diversity Score: >0.60

---

## 4. RAG Retrieval Ranking Loss

### **RAGRetrievalLoss**

**Purpose:** Improve document retrieval quality (relevant docs should rank higher)

**Components:**
```python
Loss = Pairwise_Ranking_Loss

For each relevant doc, it should score > all irrelevant docs
```

**Mathematical Formulation:**
```
Loss = Σ max(0, margin - (score_relevant - score_irrelevant))

Penalizes when:
- Relevant doc scores lower than irrelevant doc
- Margin not met (even if ordering is correct)
```

**Usage Example:**
```python
from src.training.loss_functions import RAGRetrievalLoss

criterion = RAGRetrievalLoss(margin=0.5)

loss = criterion(
    scores=retrieval_scores,    # (batch_size, num_docs) - predicted
    relevance=relevance_labels  # (batch_size, num_docs) - 0/1 labels
)
```

**Relevance Labeling:**
```python
# After retrieval, manually label each document
relevance_labels = [
    [1, 1, 0, 0, 0],  # First 2 docs relevant
    [1, 0, 1, 0, 0],  # Docs 0 and 2 relevant
    ...
]
```

**Expected Metrics:**
- Precision@3: >0.75 (75% of top-3 docs are relevant)
- Recall@5: >0.85 (85% of relevant docs in top-5)
- NDCG@5: >0.80 (normalized ranking quality)

---

### **ListwiseRankingLoss**

**Purpose:** Alternative to pairwise, considers full ranking

**Usage Example:**
```python
from src.training.loss_functions import ListwiseRankingLoss

criterion = ListwiseRankingLoss()

loss = criterion(
    scores=predicted_scores,
    relevance=relevance_scores  # Can be continuous (0-1)
)
```

**When to Use:**
- When you have **graded relevance** (0.0, 0.3, 0.7, 1.0)
- Want to optimize for ranking metrics (NDCG)
- Have many documents per query

---

## 5. Multi-Task Combined Loss

### **MultiTaskLoss**

**Purpose:** Train all components jointly in end-to-end fashion

**Components:**
```python
Total = w_anomaly × L_anomaly + w_domain × L_domain + 
        w_legal × L_legal + w_rag × L_rag

Weights can be:
1. Fixed (manual tuning)
2. Learnable (uncertainty weighting)
```

**Uncertainty Weighting:**
- Model learns task weights automatically
- Based on **task uncertainty**
- High uncertainty tasks get lower weight

**Usage Example:**
```python
from src.training.loss_functions import MultiTaskLoss

# Option 1: Fixed weights
criterion = MultiTaskLoss(
    task_weights={
        'anomaly': 1.0,
        'domain': 1.0,
        'legal': 0.5,
        'rag': 0.5
    }
)

# Option 2: Learnable weights
criterion = MultiTaskLoss(learnable_weights=True)

# Compute losses
losses = {
    'anomaly': anomaly_loss,
    'domain': domain_loss,
    'legal': legal_loss,
    'rag': rag_loss
}

total_loss, weighted_losses = criterion(losses)
```

**When to Use:**
- Training embeddings that serve multiple tasks
- Want to balance multiple objectives
- Have sufficient data for all tasks

---

## Loss Function Selection Guide

### For Your System:

**Start with (Priority Order):**

1. **DomainClassificationLoss**
   - Easiest to train
   - Most labeled data available
   - Directly improves agent assignment
   - **Start here!**

2. **AnomalyDetectionLoss**
   - Replace threshold-based detection
   - Need ~200-500 labeled signals
   - Medium difficulty

3. **LegalRecommendationLoss**
   - Requires human ratings (time-consuming)
   - Need ~100-200 rated recommendations
   - High impact on quality

4. **RAGRetrievalLoss**
   - Most complex labeling (per-document relevance)
   - Need ~100+ queries with labeled docs
   - Optimize last

### Recommended Training Order:

```
Week 1-2: Collect data (200+ signals)
Week 3: Label domain classifications → Train DomainClassificationLoss
Week 4: Label anomalies → Train AnomalyDetectionLoss
Week 5-6: Rate legal recommendations → Train LegalRecommendationLoss
Week 7+: Label RAG relevance → Train RAGRetrievalLoss
```

---

## Hyperparameter Guidelines

### Anomaly Detection:
```python
alpha = 0.7       # BCE weight
beta = 0.3        # Severity weight
gamma = 2.0       # Focal loss (2-3 for imbalanced data)
pos_weight = num_negative / num_positive  # Auto-compute
```

### Domain Classification:
```python
smoothing = 0.1   # Label smoothing (0.05-0.15)
num_classes = 6   # Fixed
class_weights = auto  # Compute from data
```

### Legal Quality:
```python
alpha = 0.5       # Quality regression
beta = 0.3        # Ranking
gamma = 0.2       # Diversity
```

### RAG Retrieval:
```python
margin = 0.5      # Pairwise margin (0.3-0.7)
```

---

## Next Steps

After understanding loss functions:

1. **Prepare Training Data** (Phase 1)
   - Collect 200+ labeled signals
   - See: `docs/TRAINING_DATA_GUIDE.md`

2. **Implement Training Loop** (Phase 3)
   - Use loss functions in training
   - See: `docs/TRAINING_LOOP_GUIDE.md`

3. **Evaluate Models** (Phase 4)
   - Compute metrics
   - Compare with baselines

4. **Deploy Fine-tuned Models** (Phase 5)
   - Replace components in system
   - Monitor performance

---

## Mathematical Reference

### Focal Loss Formula:
```
FL(p_t) = -(1 - p_t)^γ × log(p_t)

Where:
- p_t = p if y=1, else (1-p)
- γ controls focusing (γ=0 → standard CE, γ=2 → strong focusing)
```

### Label Smoothing:
```
y_smooth = y × (1 - ε) + ε / K

Where:
- y = one-hot label
- ε = smoothing factor (0.1)
- K = num_classes
```

### Contrastive Loss:
```
L = -log(exp(sim(x_i, x_j) / τ) / Σ_k exp(sim(x_i, x_k) / τ))

Where:
- sim = cosine similarity
- τ = temperature (0.07)
- j = positive sample (same class)
- k = all samples
```

---

## Troubleshooting

### Loss Not Decreasing:
- **Check learning rate** (try 1e-4, 1e-5)
- **Verify data quality** (correct labels?)
- **Check for NaN values** (gradient explosion)
- **Try simpler model** (fewer layers)

### Overfitting:
- **Increase dropout** (0.3 → 0.5)
- **Add regularization** (weight decay = 1e-4)
- **Reduce model size**
- **Get more data**

### Class Imbalance Issues:
- **Use Focal Loss** (γ=2-3)
- **Compute class weights**
- **Oversample minority class**
- **Adjust pos_weight**

### Poor Ranking Performance:
- **Increase margin** (0.5 → 1.0)
- **Use listwise loss** instead of pairwise
- **Check relevance labels** (quality issue?)
- **Try different temperature** (contrastive)

---

## Code Organization

```
src/training/
├── loss_functions.py      # All loss functions (THIS FILE)
├── models.py              # Neural network models
├── data_collector.py      # Data collection
├── trainer.py             # Training loops (Phase 3)
└── evaluator.py           # Evaluation metrics (Phase 4)
```
