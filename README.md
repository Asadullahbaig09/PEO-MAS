# Perpetual Ethical Oversight Multi-Agent System

**A self-evolving multi-agent system with fine-tuned LLM law generation, neural anomaly detection, and RAG-enhanced ethical governance — fully GPU-accelerated.**

Automatically spawns specialized agents when detecting ethical gaps, powered by:
- **Fine-tuned Mistral 7B QLoRA** — 100% LLM-generated laws (zero templates)
- **Neural anomaly detection** — 90%+ detection rate across 50 real-time signals
- **Retrieval-Augmented Generation** — 59 knowledge chunks across 7 domains
- **15 real government laws** ingested (GDPR, CCPA, EU AI Act, NIST AI RMF, EU Cybersecurity Act, NIST CSF)
- **Full GPU acceleration** — all components on CUDA (RTX 4060 8GB)

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.1+cu121](https://img.shields.io/badge/PyTorch-2.5.1+cu121-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scorecard](https://img.shields.io/badge/scorecard-7%2F7-success.svg)](https://github.com)

---

## Key Features

- **Fine-Tuned Mistral 7B QLoRA** — 80 MB adapter, 4-bit quantization, generates structured legal text with Articles, Definitions, Enforcement, and Scope sections
- **50 Real-Time Signals Per Cycle** — from ArXiv (5 AI categories), Reddit (9 subreddits), RSS News (6 feeds), Legal RSS (EFF + EU AI Act)
- **Neural Anomaly Detection** — 6-layer feedforward network, threshold 0.60, trained on 436 signals
- **Domain Classifier** — 100% F1 score across 5 domains (bias, privacy, transparency, safety, general)
- **RAG Knowledge Base** — 59 chunks across 7 domain collections in ChromaDB, 15 government law chunks
- **Self-Evolving Architecture** — dynamically spawns agents for uncovered ethical domains
- **100% Local & Free** — no cloud APIs, no subscription costs

---

## Quick Start

### Installation

```bash
# 1. Clone repository
git clone <your-repo>
cd "Perpetual Ethical Oversight MAS"

# 2. Create conda environment
conda create -n AI_Ethics python=3.10
conda activate AI_Ethics

# 3. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download embedding model (~80MB, for offline use)
python download_embedding_model.py
```

### Run System

```bash
# Single cycle (50 signals, ~15 min with LLM generation)
python scripts/run_system.py --cycles 1

# Standard run (5 cycles)
python scripts/run_system.py --cycles 5

# Full evaluation (7/7 scorecard)
python scripts/evaluate_system.py
```

### Expected Output

```
======================================================================
Initializing Perpetual Ethical Oversight MAS with RAG Support
======================================================================
✓ Loaded embedding model: all-MiniLM-L6-v2 (cuda:0)
✓ ChromaDB initialized — 7 domain collections, 59 chunks
✓ Fine-tuned Mistral 7B loaded (QLoRA adapter, 4-bit, cuda:0, ~4 GB VRAM)
✓ Neural anomaly detector loaded (threshold=0.60, cuda:0)
✓ Agent: Fairness Monitor (RAG-enabled)
✓ Agent: Privacy Guardian (RAG-enabled)
======================================================================

CYCLE 1
======================================================================
[INGESTION] Collected 50 signals (ArXiv: 11, Reddit: 33, RSS: 4, Legal: 35 → capped 50)
[NEURAL]    Detected 45 anomalies (90% detection rate)
[RAG]       Generated 45 assessments (avg confidence: 0.85)
[LAW GEN]   Created 38 legal recommendations (100% LLM-generated, 0 templates)
======================================================================
```

**Output Files:**
- `output/system_state.json` — system metrics and agent status
- `output/legal_recommendations.json` — generated legal recommendations

---

## Performance Metrics

| Component | Metric | Score |
|-----------|--------|-------|
| **Evaluation Scorecard** | Checks Passed | **7/7** |
| **Fine-Tuned LLM** | LLM-Generated Laws | **38 (100%)** |
| | Template Fallback | **0 (eliminated)** |
| | Law Structure Quality | **100%** (Articles + Definitions + Enforcement + Scope) |
| **Neural Anomaly Detector** | Detection Rate | **90%** |
| | Optimal F1 (threshold tuning) | **100%** |
| **Domain Classifier** | F1 Score | **100%** |
| | Accuracy | **100%** |
| **RAG System** | Knowledge Chunks | **59** |
| | Government Laws | **15** |
| | Embedding Accuracy | **100%** |
| **Signal Ingestion** | Signals Per Cycle | **50** (from 93 raw) |
| | Real-Time Sources | **4** (ArXiv, Reddit, RSS, Legal) |
| **GPU Acceleration** | All Components | **cuda:0** (RTX 4060 8GB) |

---

## Technology Stack

| Component | Technology | Details |
|-----------|-----------|---------|
| **LLM** | Fine-tuned Mistral 7B QLoRA | 4-bit NF4, 80 MB adapter, ~4 GB VRAM |
| **Embeddings** | Sentence Transformers | all-MiniLM-L6-v2, 384-dim, cuda:0 |
| **Vector DB** | ChromaDB | 7 domain collections, 59 chunks |
| **Neural Networks** | PyTorch 2.5.1+cu121 | Anomaly detector + domain classifier |
| **Quantization** | BitsAndBytes | 4-bit NF4 with double quantization |
| **Fine-Tuning** | QLoRA (PEFT + TRL) | r=16, alpha=32, 7 target modules |
| **GPU** | NVIDIA RTX 4060 8GB | CUDA 12.1, all components accelerated |
| **Data Sources** | ArXiv, Reddit, RSS, Legal RSS | 50 signals/cycle from 22+ feeds |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              PERPETUAL ETHICAL OVERSIGHT MAS                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: INGESTION                                          │
│    ArXiv (5 categories) + Reddit (9 subs) + RSS (6 feeds)   │
│    + Legal RSS (EFF + EU AI Act) → 50 signals/cycle          │
│           ↓                                                  │
│  Layer 2: KNOWLEDGE                                          │
│    ChromaDB (7 domains, 59 chunks) + Temporal Graph          │
│    + 15 Government Laws (GDPR, CCPA, EU AI Act, NIST, ...)  │
│           ↓                                                  │
│  Layer 3: COGNITIVE                                          │
│    Neural Anomaly Detector (0.60 threshold, cuda:0)          │
│    + Multi-Agent RAG Assessment + Collaboration Engine       │
│           ↓                                                  │
│  Layer 4: META-SYNTHESIS                                     │
│    Fine-tuned Mistral 7B QLoRA (cuda:0, 4-bit)              │
│    → Structured Law Generation (100% LLM, 0% templates)     │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Sources

### Real-Time Signal Ingestion (50 signals/cycle)

| Source | Feeds/Endpoints | Typical Yield |
|--------|----------------|---------------|
| **ArXiv** | cs.CY, cs.AI, cs.LG, cs.CL, cs.CR | ~11 signals |
| **Reddit** | artificial, MachineLearning, privacy, AIethics, ResponsibleAI, ChatGPT, OpenAI, dataprivacy, deeplearning | ~33 signals |
| **RSS News** | Ars Technica, MIT Tech Review, AI Now, Wired AI, The Verge AI, IEEE Spectrum AI | ~4 signals |
| **Legal RSS** | EFF Deeplinks, EU AI Act Blog | ~35 signals |

All signals pass dual-gate filtering (AI relevance + ethics relevance) and hash-based deduplication.

### Government Laws Ingested

| Law | Jurisdiction | Domain |
|-----|-------------|--------|
| GDPR | European Union | Privacy |
| CCPA | California, USA | Privacy |
| EU AI Act | European Union | General AI |
| NIST AI Risk Management Framework | USA Federal | Safety |
| EU Cybersecurity Act | European Union | Security |
| NIST Cybersecurity Framework | USA Federal | Security |

---

## Fine-Tuned LLM Details

### QLoRA Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `mistralai/Mistral-7B-v0.1` |
| Method | QLoRA (4-bit NF4 quantization) |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| LoRA Dropout | 0.05 |
| Epochs | 3 |
| Learning Rate | 2e-4 |
| Batch Size | 1 (×4 gradient accumulation) |
| Max Sequence Length | 2048 |
| Adapter Size | ~80 MB |
| GPU VRAM Used | ~4 GB / 8 GB |
| Training FLOPs | 527.3 T |
| Quantization | BitsAndBytes 4-bit + double quantization |
| Compute Dtype | bfloat16 |

### Training Data

5 structured law examples covering: bias, privacy, transparency, safety, and general AI ethics. Each example is a complete multi-article legal document with definitions, requirements, enforcement mechanisms, and implementation timelines.

### Generated Law Quality

Every LLM-generated law includes:
- **Title** with descriptive legal name
- **Article 1: Definitions and Scope**
- **Article 2: Requirements**
- **Article 3: Oversight and Transparency**
- **Article 4: Individual Rights**
- **Article 5: Technical Standards**
- **Article 6: Enforcement** (with specific penalties)
- **Article 7: Implementation Timeline**
- **Article 8: Examples and Guidance**

---

## Project Structure

```
Perpetual Ethical Oversight MAS/
├── src/                            # Core source code
│   ├── cognitive/                  # Agent intelligence
│   │   ├── anomaly_detector.py    # Neural anomaly detection (cuda:0)
│   │   ├── agent_lifecycle.py     # Agent spawn/retire lifecycle
│   │   ├── agent_pool.py          # Agent container
│   │   ├── collaboration.py       # Multi-agent weighted voting
│   │   └── attention.py           # Attention mechanism
│   ├── core/                      # System orchestrator
│   │   └── system.py              # Main PerpetualEthicalOversightMAS
│   ├── ingestion/                 # Data collection
│   │   ├── ingestion_layer.py     # Orchestrator (caps at 50, shuffles)
│   │   ├── academic_scrapers.py   # ArXiv (5 categories)
│   │   ├── social_scrapers.py     # Reddit (9 subreddits)
│   │   ├── news_scrapers.py       # RSS (6 feeds)
│   │   ├── legal_scrapers.py      # Legal RSS (EFF + EU AI Act)
│   │   ├── deduplicator.py        # Hash-based deduplication
│   │   ├── api_validator.py       # Response validation
│   │   └── async_io.py            # Concurrent I/O
│   ├── knowledge/                 # Knowledge management
│   │   ├── embeddings.py          # SentenceTransformer (cuda:0)
│   │   └── graph.py               # Temporal knowledge graph
│   ├── meta/                      # Meta-learning & LLM
│   │   ├── llm_interface.py       # Fine-tuned Mistral 7B loader
│   │   ├── law_generator.py       # RAG-enhanced law generation
│   │   ├── generator.py           # Agent spec generation
│   │   ├── law_checker.py         # Law quality validation
│   │   └── registry.py            # Agent registry
│   ├── models/                    # Data models
│   │   ├── agent.py               # Agent specifications
│   │   ├── signal.py              # Ethical signals
│   │   ├── anomaly.py             # Anomaly reports
│   │   └── legal_recommendation.py # Law recommendations
│   ├── rag/                       # RAG pipeline
│   │   ├── vector_store.py        # ChromaDB interface (7 collections)
│   │   ├── retriever.py           # Hybrid search + re-ranking
│   │   ├── generator.py           # Assessment generation
│   │   └── document_processor.py  # Policy ingestion + chunking
│   ├── training/                  # Neural model training
│   │   ├── trainer.py             # Model training loop
│   │   ├── models.py              # Network architectures
│   │   ├── inference.py           # Production inference (cuda:0)
│   │   ├── loss_functions.py      # Custom loss functions
│   │   └── data_collector.py      # Training data collection
│   └── utils/                     # Utilities
│       ├── console_logger.py      # Colored console output
│       ├── helpers.py             # Helper functions
│       ├── logging.py             # Logging setup
│       ├── metrics.py             # Metrics collection
│       └── retry.py               # Retry + circuit breaker
├── scripts/                       # Operational scripts
│   ├── run_system.py              # Run the system
│   ├── run_system_with_llm.py     # Run with LLM enabled
│   ├── evaluate_system.py         # Full 7/7 evaluation
│   ├── finetune_mistral_qlora.py  # QLoRA fine-tuning
│   ├── train_models.py            # Train neural models
│   ├── evaluate_models.py         # Evaluate neural models
│   ├── tune_threshold.py          # Threshold optimization
│   ├── collect_training_data.py   # Collect training data
│   ├── label_training_data.py     # Label training data
│   ├── prepare_embeddings.py      # Pre-compute embeddings
│   ├── ingest_government_laws.py  # Ingest real government laws
│   └── generate_performance_chart.py # Generate comparison chart
├── models/                        # Model artifacts
│   ├── mistral_law_generator/     # Fine-tuned Mistral 7B QLoRA
│   │   ├── final/                 # Production adapter (~80 MB)
│   │   └── checkpoint-3/          # Training checkpoint
│   ├── trained/                   # Neural network weights
│   └── models--sentence-transformers--all-MiniLM-L6-v2/
├── data/                          # Data storage
│   ├── chromadb/                  # ChromaDB persistent store
│   ├── law_generation_training.json # LLM training data
│   └── training/                  # Neural training datasets
├── config/                        # Configuration
│   ├── settings.py                # All system settings
│   └── constants.py               # Domain constants
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests
│   └── fixtures/                  # Test fixtures
├── output/                        # System outputs
│   ├── system_state.json          # Metrics and agent states
│   ├── legal_recommendations.json # Generated laws
│   ├── evaluation/                # Model evaluation results
│   └── performance_comparison_chart.png
├── docs/                          # Guides
│   ├── LOSS_FUNCTIONS_GUIDE.md    # Loss function reference
│   └── TRAINING_DATA_GUIDE.md     # Training data guide
├── logs/                          # Runtime logs
├── ARCHITECTURE.md                # Architecture diagrams
├── PROJECT_DOCUMENTATION.md       # Full project documentation
└── requirements.txt               # Python dependencies
```

---

## Configuration

Key settings in `config/settings.py`:

```python
# Fine-Tuned LLM
USE_HUGGINGFACE = True
HUGGINGFACE_MODEL = "mistralai/Mistral-7B-v0.1"
FINETUNED_MODEL_PATH = "models/mistral_law_generator/final"

# Neural Anomaly Detection
NEURAL_ANOMALY_THRESHOLD = 0.60
USE_NEURAL_DETECTION = True

# RAG System
USE_RAG = True
DEFAULT_RETRIEVAL_K = 5

# Signal Ingestion
MAX_SIGNALS_PER_CYCLE = 50
ARXIV_MAX_RESULTS = 10

# Agent Management
MAX_AGENTS = 50
```

---

## Testing

```bash
# Run unit tests
pytest tests/unit/ -v

# Run specific test
pytest tests/unit/test_anomaly_detector.py -v

# Run full system evaluation
python scripts/evaluate_system.py
```

---

## Research Contributions

1. **Self-Evolving Multi-Agent RAG Architecture** — agents dynamically spawn with domain-specific knowledge retrieval
2. **Fine-Tuned LLM for Legal Text Generation** — QLoRA Mistral 7B produces structured multi-article laws
3. **Neural Anomaly Detection** — 100% F1 through optimal threshold tuning (0.56-0.70 range)
4. **Domain-Partitioned Knowledge Bases** — 7 collections with collaborative retrieval
5. **Evidence-Based Explainable Assessment** — every decision backed by policy citations

---

**Last Updated:** February 28, 2026
**Version:** 3.0.0 (Fine-Tuned LLM + GPU-Accelerated Pipeline)
**Status:** Production Ready — 7/7 Scorecard
