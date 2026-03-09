# Training Data Collection Guide

## Overview

This guide explains how to collect and label training data from your Multi-Agent RAG System for future model fine-tuning.

## Phase 1: Collect Training Data

### Step 1: Run Data Collection

```bash
# Collect data from 5 system cycles
python scripts/collect_training_data.py --cycles 5

# Or with custom session name
python scripts/collect_training_data.py --cycles 10 --session "production_week1"
```

### Step 2: Review Generated Files

Data is saved to `data/training/[session_name]/`:

```
data/training/20260205_091630/
├── summary.json                    # Collection summary
├── signals.json                    # Raw signal data
├── signals.csv                     # ← Label this!
├── legal_recommendations.json      # Raw legal recommendations
├── legal_recommendations.csv       # ← Rate this!
├── domains.json                    # Domain predictions
└── rag_retrievals.json            # RAG retrieval results
```

## Step 3: Label the Data

### A. Signal Labeling (signals.csv)

Open `signals.csv` in Excel/Google Sheets and add labels:

| Column | Description | How to Label |
|--------|-------------|--------------|
| `true_domain` | Correct domain | privacy, bias, transparency, safety, accountability, general |
| `true_anomaly` | Is it really an anomaly? | TRUE or FALSE |
| `severity_rating` | How severe is it? | 0.0 to 1.0 (0.0 = low, 1.0 = critical) |
| `notes` | Optional feedback | "Actually a privacy violation, not bias" |

**Example:**

```csv
signal_id,title,predicted_domain,predicted_anomaly,true_domain,true_anomaly,severity_rating,notes
abc123,Google Privacy Settlement,general,TRUE,privacy,TRUE,0.85,Major privacy case
def456,AI Chatbot Ads,general,FALSE,general,FALSE,0.30,Not really an issue
```

### B. Legal Recommendation Rating (legal_recommendations.csv)

Rate each generated legal recommendation:

| Column | Description | Rating Scale |
|--------|-------------|--------------|
| `quality_rating` | Overall quality | 1-5 (1=poor, 5=excellent) |
| `relevance_rating` | Relevance to issue | 1-5 |
| `legal_soundness` | Legal accuracy | 1-5 |
| `clarity_rating` | Clarity of writing | 1-5 |
| `human_feedback` | Specific feedback | "Missing enforcement mechanism" |

**Example:**

```csv
rec_id,rec_title,quality_rating,relevance_rating,legal_soundness,clarity_rating,human_feedback
xyz789,Data Privacy Act,4.5,5.0,4.0,4.5,Good but needs more specific penalties
```

## Step 4: Export Training Datasets

After labeling, export for training:

```python
from src.training.data_collector import TrainingDataCollector

collector = TrainingDataCollector()
# Load your session
collector.load_human_labels('data/training/20260205_091630/signals.csv')

# Export training-ready datasets
datasets = collector.export_for_training('all')
```

This creates:
- `anomaly_detection_dataset.json` - For training anomaly detector
- `domain_classification_dataset.json` - For fine-tuning embeddings
- `legal_recommendation_dataset.json` - For LLM fine-tuning
- `rag_retrieval_dataset.json` - For improving RAG

## What You Can Train

### 1. Anomaly Detector (Binary Classification)

**Current:** Threshold-based (coverage < 0.65)  
**After Training:** Neural network classifier

```python
# Dataset format
{
    "text": "CRITICAL FINDINGS: Widespread bias in foundation models...",
    "severity": 0.89,
    "label": True  # Is anomaly
}
```

### 2. Domain Classifier (Multi-class Classification)

**Current:** Rule-based matching  
**After Training:** Fine-tuned embedding model

```python
# Dataset format
{
    "text": "Privacy catastrophe in federated learning systems...",
    "label": "privacy"  # One of: privacy, bias, transparency, safety, accountability, general
}
```

### 3. Legal Recommendation Generator (LLM Fine-tuning)

**Current:** Generic LLM prompting  
**After Training:** Domain-specific legal expert model

```python
# Dataset format
{
    "input": "Signal: Google Privacy Settlement\nDomain: privacy\nSeverity: 0.85",
    "output": "Title: Data Privacy Protection Act\nSection 1: ...",
    "quality_score": 4.5
}
```

## Data Collection Best Practices

### How Much Data Do You Need?

| Task | Minimum | Recommended | Optimal |
|------|---------|-------------|---------|
| Anomaly Detection | 100 samples | 500 samples | 1000+ samples |
| Domain Classification | 50 per class | 200 per class | 500+ per class |
| LLM Fine-tuning | 100 examples | 500 examples | 1000+ examples |
| RAG Improvement | 50 queries | 200 queries | 500+ queries |

### Collection Strategy

**Week 1-2: Baseline Collection**
```bash
# Run 5 cycles per day
python scripts/collect_training_data.py --cycles 5 --session "week1_day1"
python scripts/collect_training_data.py --cycles 5 --session "week1_day2"
# ... collect ~200-300 signals
```

**Week 3-4: Labeling**
- Label 50-100 signals per day
- Focus on diverse examples (all domains)
- Get multiple raters for quality checks

**Week 5+: Training**
- Start with domain classification (easiest)
- Then anomaly detection
- Finally LLM fine-tuning (most complex)

## Quality Control

### Signal Labeling Checklist

- [ ] Each signal has `true_domain` label
- [ ] `true_anomaly` matches your judgment
- [ ] `severity_rating` is consistent (0.0-1.0)
- [ ] Difficult cases have `notes`
- [ ] All domains represented (no class imbalance)

### Legal Recommendation Checklist

- [ ] All ratings on 1-5 scale
- [ ] `quality_rating` reflects overall usefulness
- [ ] `legal_soundness` checked against regulations
- [ ] Specific `human_feedback` for low-rated recommendations
- [ ] High-quality examples marked for training emphasis

## Troubleshooting

### "I have too much unlabeled data"

**Solution:** Start with high-value samples
1. Label high-severity signals first (severity > 0.7)
2. Label diverse domains (avoid privacy-only)
3. Label edge cases (hard to classify)

### "My labels are inconsistent"

**Solution:** Create labeling guidelines
1. Define each domain clearly
2. Set anomaly criteria (what makes it anomalous?)
3. Use severity examples (0.3 = minor, 0.7 = major, 0.9 = critical)
4. Have multiple people label same samples

### "I don't know which domain to assign"

**Solution:** Use multi-label approach
- Primary domain: `privacy`
- Secondary domains: `['transparency', 'accountability']`
- Update data collector to support multi-label

## Next Steps

After collecting 200+ labeled samples:

1. **Train Domain Classifier**  
   See: `docs/TRAINING_EMBEDDING_MODEL.md`

2. **Train Anomaly Detector**  
   See: `docs/TRAINING_ANOMALY_DETECTOR.md`

3. **Fine-tune LLM**  
   See: `docs/TRAINING_LEGAL_LLM.md`

## Example Workflow

```bash
# Day 1: Collect data
python scripts/collect_training_data.py --cycles 10 --session "production_001"

# Day 2-3: Label data
# Open data/training/production_001/signals.csv
# Add true_domain, true_anomaly, severity_rating columns
# Save as signals_labeled.csv

# Day 4: Export training data
python -c "
from src.training.data_collector import TrainingDataCollector
c = TrainingDataCollector()
c.load_human_labels('data/training/production_001/signals_labeled.csv')
datasets = c.export_for_training('all')
print('Ready for training:', datasets)
"

# Day 5+: Train models
python scripts/train_domain_classifier.py --data data/training/training_export_*/domain_classification_dataset.json
```

## Support

For questions or issues:
1. Check `logs/` for error messages
2. Review `output/system_state.json` for metrics
3. Examine collected data for patterns
