# System Architecture Diagrams

**Last Updated:** February 28, 2026
**System Status:** Production Ready — 7/7 Scorecard
**LLM:** Fine-tuned Mistral 7B QLoRA (cuda:0, 4-bit, ~4 GB VRAM)
**Signals Per Cycle:** 50 (from 93 raw across 4 sources)
**RAG Knowledge Base:** 59 chunks, 7 domains, 15 government laws
**Active Agents:** 7 (dynamically evolving)

---

## High-Level Architecture

```mermaid
graph TB
    subgraph "Real-Time Data Sources"
        A1[ArXiv API<br/>5 categories]
        A2[Reddit API<br/>9 subreddits]
        A3[RSS News<br/>6 feeds]
        A4[Legal RSS<br/>EFF + EU AI Act]
    end

    subgraph "Layer 1: Ingestion (50 signals/cycle)"
        B1[Signal Scrapers]
        B2[Deduplicator<br/>Hash-based]
        B3[API Validator]
        B4[Dual-Gate Filter<br/>AI + Ethics keywords]
    end

    subgraph "Layer 2: Knowledge & RAG"
        C1[ChromaDB Vector Store<br/>7 domains, 59 chunks]
        C2[Temporal Knowledge Graph<br/>Decay: S(t)=α^t·S(t-1)+E(t)]
        C3[Document Processor<br/>Chunking + Embedding]
        C4[Embedding Engine<br/>all-MiniLM-L6-v2, cuda:0]
    end

    subgraph "Layer 3: Cognitive Multi-Agent System"
        D1[Agent Pool<br/>7 active agents]
        D2[Neural Anomaly Detector<br/>threshold=0.60, cuda:0]
        D3[Attention Mechanism]
        D4[Agent Lifecycle Manager]
        D5[Collaboration Engine<br/>Weighted voting]
    end

    subgraph "Layer 4: Meta-Synthesis"
        E1[Document Retriever<br/>Hybrid search + re-rank]
        E2[Assessment Generator<br/>Evidence-based]
        E3[Evidence Synthesizer]
    end

    subgraph "Layer 5: Law Generation"
        F1[Meta Agent Generator]
        F2[Agent Registry]
        F3[Fine-tuned Mistral 7B<br/>QLoRA, 4-bit, cuda:0]
        F4[Law Quality Checker<br/>Structure validation]
    end

    subgraph "Output"
        G1[38 Legal Recommendations<br/>100% LLM-generated]
        G2[Ethical Assessments<br/>with citations]
        G3[System Evolution Log]
        G4[Performance Chart]
    end

    A1 & A2 & A3 & A4 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> C1 & C2
    C3 --> C1
    C4 --> C1

    C1 & C2 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> D4
    D4 --> D5

    D1 --> E1
    E1 --> E2
    E2 --> E3

    D2 --> F1
    F1 --> F2
    F3 --> F1
    F3 --> F4
    F2 --> D1

    E3 --> G1 & G2
    F1 --> G3
    F4 --> G4

    style C1 fill:#90EE90
    style E1 fill:#87CEEB
    style E2 fill:#87CEEB
    style F3 fill:#FFB6C1
    style D2 fill:#DDA0DD
```

---

## RAG Workflow: Signal to Assessment

**Status:** Fully functional with fine-tuned Mistral 7B QLoRA
**Performance:** ~5 documents retrieved per assessment, 0.85 confidence scores

```mermaid
sequenceDiagram
    participant Signal as Ethical Signal
    participant Agent as Ethical Agent
    participant Retriever as Document Retriever
    participant VectorDB as ChromaDB (7 domains)
    participant Generator as Assessment Generator
    participant LLM as Fine-tuned Mistral 7B<br/>(QLoRA, cuda:0)
    participant Output as Assessment Output

    Signal->>Agent: New signal arrives
    Agent->>Agent: Check domain match

    alt RAG Enabled (Active)
        Agent->>Retriever: retrieve_for_signal(domain, k=5)
        Retriever->>VectorDB: Search domain collection
        Note over VectorDB: 7 domain collections:<br/>bias, privacy, transparency,<br/>accountability, safety,<br/>security, general
        VectorDB-->>Retriever: Top-5 documents + scores
        Retriever->>Retriever: Re-rank (semantic + keyword + recency)
        Retriever-->>Agent: RetrievalResult (3-5 docs)

        Agent->>Generator: generate_assessment(signal, context)
        Generator->>Generator: Build context from retrieved docs

        Generator->>LLM: Send prompt + RAG context
        Note over LLM: Fine-tuned Mistral 7B<br/>QLoRA adapter loaded<br/>4-bit quantization<br/>~4 GB VRAM
        LLM-->>Generator: Structured legal text<br/>(Articles, Definitions,<br/>Enforcement, Scope)
        Generator->>Generator: Validate law structure

        Generator-->>Agent: EthicalAssessment<br/>(risk_level, confidence=0.85)
        Agent->>Output: Evidence-based assessment<br/>with policy citations
    end

    Note over Output: Final Assessment includes:<br/>- Structured law (8 articles)<br/>- Risk Level<br/>- Confidence Score<br/>- Policy Citations<br/>- Enforcement Mechanisms
```

---

## Multi-Agent Collaboration Flow

**Active Agents (Production):**
- Fairness Monitor (bias) — Base agent
- Privacy Guardian (privacy) — Base agent
- General AI Ethics Overseer (general) — Evolved
- Security Compliance Monitor (security) — Evolved
- Accountability Monitor (accountability) — Evolved
- Transparency Monitor (transparency) — Evolved
- Safety Compliance Agent (safety) — Evolved

```mermaid
graph LR
    subgraph "Signal Processing"
        S[Ethical Signal<br/>50 signals/cycle]
    end

    subgraph "Agent Pool (7 agents)"
        A1[Privacy Agent<br/>Expertise: 0.9]
        A2[Bias Agent<br/>Expertise: 0.8]
        A3[Security Agent<br/>Expertise: 0.7]
    end

    subgraph "RAG Retrieval (59 chunks)"
        R1[Privacy Policies<br/>+ GDPR, CCPA]
        R2[Fairness Guidelines<br/>+ Bias Prevention]
        R3[Security Standards<br/>+ NIST CSF, EU Cyber]
    end

    subgraph "Individual Assessments"
        AS1[Assessment 1<br/>Confidence: 0.87]
        AS2[Assessment 2<br/>Confidence: 0.82]
        AS3[Assessment 3<br/>Confidence: 0.75]
    end

    subgraph "Collaboration"
        COL[Weighted Voting<br/>Evidence Synthesis]
    end

    subgraph "Law Generation"
        LLM[Fine-tuned Mistral 7B<br/>QLoRA, cuda:0]
        LAW[Structured Law<br/>8 Articles + Enforcement]
    end

    S --> A1 & A2 & A3
    A1 --> R1 --> AS1
    A2 --> R2 --> AS2
    A3 --> R3 --> AS3
    AS1 & AS2 & AS3 --> COL
    COL --> LLM --> LAW

    style S fill:#FFE4B5
    style R1 fill:#90EE90
    style R2 fill:#90EE90
    style R3 fill:#90EE90
    style LLM fill:#FFB6C1
    style LAW fill:#87CEEB
```

---

## Self-Evolution Process

```mermaid
stateDiagram-v2
    [*] --> SignalIngestion
    SignalIngestion --> NeuralDetection

    NeuralDetection --> Normal: Score < 0.60
    NeuralDetection --> AnomalyDetected: Score >= 0.60

    Normal --> ProcessNextSignal

    AnomalyDetected --> GapAnalysis
    GapAnalysis --> GenerateAgentSpec

    GenerateAgentSpec --> LLMGeneration: Fine-tuned Mistral 7B (cuda:0)

    LLMGeneration --> CreateAgent
    CreateAgent --> InitializeRAG: Attach retriever + generator
    InitializeRAG --> RegisterAgent: use_rag=true, retrieval_k=5
    RegisterAgent --> AgentPool

    AgentPool --> ProcessNextSignal
    ProcessNextSignal --> SignalIngestion

    note right of AnomalyDetected
        Neural anomaly detector
        (threshold=0.60, cuda:0)
        90% detection rate
    end note

    note right of LLMGeneration
        QLoRA adapter loaded
        4-bit quantization
        ~4 GB VRAM
        100% LLM-generated laws
    end note
```

---

## End-to-End Data Flow

```mermaid
flowchart TD
    Start([Start System]) --> Init[Initialize Components]
    Init --> LoadModels[Load Models to GPU]
    LoadModels --> Models[Embedding: cuda:0<br/>Anomaly Detector: cuda:0<br/>Mistral 7B QLoRA: cuda:0<br/>Domain Classifier: cuda:0]
    Models --> LoadRAG[Load RAG Knowledge Base<br/>59 chunks, 7 domains,<br/>15 government laws]
    LoadRAG --> Ready[System Ready]

    Ready --> Cycle[Start Processing Cycle]
    Cycle --> Scrape[Scrape 4 Sources in Parallel]
    Scrape --> Sources[ArXiv: 11 signals<br/>Reddit: 33 signals<br/>RSS: 4 signals<br/>Legal: 35 signals]
    Sources --> Shuffle[Shuffle + Cap at 50]
    Shuffle --> Dedupe{Deduplicate}

    Dedupe -->|Duplicate| Skip[Skip]
    Dedupe -->|New| Embed[Generate Embedding<br/>384-dim, cuda:0]

    Embed --> Detect[Neural Anomaly Detection<br/>threshold=0.60, cuda:0]
    Detect --> Rate[90% Detection Rate<br/>45 of 50 flagged]

    Rate --> RAGRetrieve[RAG Retrieval<br/>5 docs per signal]
    RAGRetrieve --> Assess[Agent Assessment<br/>confidence ~0.85]
    Assess --> LawGen[Law Generation<br/>Fine-tuned Mistral 7B]
    LawGen --> Quality[Quality Check<br/>Articles + Definitions +<br/>Enforcement + Scope]
    Quality --> Output[38 Legal Recommendations<br/>100% LLM-generated]
    Output --> Export[Export Results<br/>system_state.json<br/>legal_recommendations.json]

    style Models fill:#FFB6C1
    style LoadRAG fill:#90EE90
    style LawGen fill:#DDA0DD
    style Output fill:#87CEEB
```

---

## Component Layer Architecture

```mermaid
graph TB
    subgraph "Core System"
        SYS[PerpetualEthicalOversightMAS<br/>src/core/system.py]
    end

    subgraph "Service Layer"
        ING[Ingestion Service<br/>4 scrapers, 50 signals/cycle]
        COG[Cognitive Service<br/>Neural detection, 7 agents]
        META[Meta-Learning Service<br/>Agent generation, law checking]
        RAG[RAG Service<br/>59 chunks, 7 domains]
    end

    subgraph "Data Access Layer"
        VDB[(ChromaDB<br/>7 collections, 59 chunks)]
        KG[(Temporal Knowledge Graph<br/>In-Memory)]
    end

    subgraph "GPU-Accelerated Models (cuda:0)"
        LLM[Fine-tuned Mistral 7B<br/>QLoRA, 4-bit, ~4 GB VRAM]
        EMB[Embedding Model<br/>all-MiniLM-L6-v2]
        ANOM[Anomaly Detector<br/>6-layer NN, threshold=0.60]
        DOM[Domain Classifier<br/>100% F1]
    end

    subgraph "External Sources"
        ARXIV[ArXiv: cs.CY, cs.AI,<br/>cs.LG, cs.CL, cs.CR]
        REDDIT[Reddit: 9 subreddits]
        RSS[RSS: 6 news feeds]
        LEGAL[Legal: EFF + EU AI Act]
    end

    SYS --> ING & COG & META & RAG

    ING --> VDB & KG
    COG --> VDB & KG
    META --> VDB
    RAG --> VDB

    ING --> ARXIV & REDDIT & RSS & LEGAL
    META --> LLM
    RAG --> LLM & EMB
    COG --> ANOM & DOM & EMB

    style SYS fill:#FFD700
    style LLM fill:#FFB6C1
    style VDB fill:#90EE90
    style EMB fill:#87CEEB
    style ANOM fill:#DDA0DD
```

---

## GPU Memory Layout

```
NVIDIA RTX 4060 (8 GB VRAM)
┌──────────────────────────────────────────┐
│  Fine-tuned Mistral 7B QLoRA   ~4.0 GB  │
│  (4-bit NF4 + LoRA adapter)             │
├──────────────────────────────────────────┤
│  Embedding Model (MiniLM-L6)   ~0.1 GB  │
├──────────────────────────────────────────┤
│  Anomaly Detector NN           ~0.01 GB  │
├──────────────────────────────────────────┤
│  Domain Classifier NN          ~0.01 GB  │
├──────────────────────────────────────────┤
│  PyTorch CUDA Overhead         ~1.5 GB   │
├──────────────────────────────────────────┤
│  Free                          ~2.4 GB   │
└──────────────────────────────────────────┘
Total Used: ~5.6 GB / 8 GB
```

---

## Security & Privacy Architecture

**100% Local Operation:**
- No cloud API calls for LLM (fine-tuned model runs locally on GPU)
- No vector DB cloud services (ChromaDB stored at `./data/chromadb`)
- No external authentication required
- All data processing happens on local machine

```mermaid
graph TB
    subgraph "External World (HTTPS only)"
        EXT[ArXiv, Reddit, RSS<br/>Read-only data scraping]
    end

    subgraph "Network Boundary"
        VAL[API Validator<br/>Schema validation]
        RETRY[Retry Handler<br/>Max 3 retries]
        CB[Circuit Breaker<br/>5 failures → disable]
    end

    subgraph "Application Layer — LOCAL ONLY"
        APP[System Orchestrator]
        AGENTS[Agent Pool — 7 agents]
        RAG_SYS[RAG System — 59 chunks]
    end

    subgraph "GPU Layer — LOCAL ONLY (cuda:0)"
        LLM_GPU[Mistral 7B QLoRA<br/>~4 GB VRAM]
        EMB_GPU[Embedding Model]
        NN_GPU[Neural Networks]
    end

    subgraph "Data Layer — LOCAL ONLY"
        VDB_LOCAL[(ChromaDB<br/>./data/chromadb)]
        LOGS_LOCAL[(Logs<br/>./logs)]
        OUT_LOCAL[(Output<br/>./output)]
    end

    EXT -->|HTTPS| VAL --> RETRY --> CB --> APP
    APP --> AGENTS & RAG_SYS
    AGENTS & RAG_SYS --> LLM_GPU & EMB_GPU & NN_GPU
    APP --> VDB_LOCAL & LOGS_LOCAL & OUT_LOCAL

    style VDB_LOCAL fill:#90EE90
    style LLM_GPU fill:#FFB6C1
```

---

## Project Structure

```
Perpetual Ethical Oversight MAS/
│
├── src/ingestion/                  — Signal Collection (50/cycle)
│   ├── ingestion_layer.py         ← Orchestrator (shuffle + cap at 50)
│   ├── academic_scrapers.py       ← ArXiv (5 categories: cs.CY/AI/LG/CL/CR)
│   ├── social_scrapers.py         ← Reddit (9 subreddits)
│   ├── news_scrapers.py           ← RSS (6 feeds)
│   ├── legal_scrapers.py          ← Legal RSS (EFF + EU AI Act)
│   ├── deduplicator.py            ← Hash-based deduplication
│   ├── api_validator.py           ← Response validation
│   └── async_io.py                ← Concurrent I/O
│
├── src/knowledge/                  — Knowledge Management
│   ├── graph.py                   ← Temporal knowledge graph
│   └── embeddings.py              ← SentenceTransformer (cuda:0)
│
├── src/rag/                        — RAG Pipeline (59 chunks, 7 domains)
│   ├── vector_store.py            ← ChromaDB interface
│   ├── retriever.py               ← Hybrid search + re-ranking
│   ├── generator.py               ← Assessment generation
│   └── document_processor.py      ← Policy ingestion + chunking
│
├── src/cognitive/                  — Multi-Agent Intelligence
│   ├── anomaly_detector.py        ← Neural detector (cuda:0, threshold=0.60)
│   ├── agent_lifecycle.py         ← Spawn/retire lifecycle
│   ├── agent_pool.py              ← Agent container
│   ├── collaboration.py           ← Weighted voting
│   └── attention.py               ← Attention mechanism
│
├── src/meta/                       — Law Generation & Meta-Learning
│   ├── llm_interface.py           ← Fine-tuned Mistral 7B loader (QLoRA, cuda:0)
│   ├── law_generator.py           ← RAG-enhanced law generation
│   ├── generator.py               ← Agent spec generation
│   ├── law_checker.py             ← Law quality validation
│   └── registry.py                ← Agent registry
│
├── src/training/                   — Neural Model Training
│   ├── trainer.py                 ← Training loop
│   ├── models.py                  ← Network architectures
│   ├── inference.py               ← Production inference (cuda:0)
│   ├── loss_functions.py          ← Custom losses
│   └── data_collector.py          ← Training data collection
│
├── models/                         — Model Artifacts
│   ├── mistral_law_generator/     ← QLoRA adapter (~80 MB)
│   └── trained/                   ← Neural weights (.pt)
│
├── data/                           — Data Storage
│   ├── chromadb/                  ← Vector DB (persistent)
│   └── training/                  ← Training datasets
│
├── scripts/                        — Operational Scripts
│   ├── run_system.py              ← Run system
│   ├── evaluate_system.py         ← Full evaluation (7/7)
│   ├── finetune_mistral_qlora.py  ← QLoRA fine-tuning
│   └── ...                        ← Training, ingestion, analysis
│
├── output/                         — Results
│   ├── legal_recommendations.json ← 38 LLM-generated laws
│   ├── system_state.json          ← Metrics + agent states
│   └── performance_comparison_chart.png
│
└── config/settings.py              — All configuration
```

---

## Key Design Patterns

```mermaid
mindmap
  root((Architecture Patterns))
    Multi-Agent System
      Dynamic Agent Pool (7 agents)
      Weighted Collaboration
      Lifecycle Management
      Domain Specialization
    RAG Pattern
      ChromaDB Vector Store
      Hybrid Retrieval
      Evidence Synthesis
      7 Domain Collections
    Fine-Tuned LLM
      QLoRA (4-bit NF4)
      Structured Law Output
      GPU-Accelerated (cuda:0)
      80 MB Adapter
    Neural Detection
      Anomaly Detector (0.60)
      Domain Classifier (100% F1)
      Threshold Optimization
    Resilience
      Circuit Breaker
      Retry with Backoff
      Graceful Degradation
    Data Pipeline
      4 Source Scrapers
      Dual-Gate Filtering
      Hash Deduplication
      Shuffle + Cap at 50
```

---

**Note:** Diagrams are in Mermaid format and render in GitHub, GitLab, VS Code (with Mermaid extension), or at https://mermaid.live/
