# Fine-Tuned Mistral 7B QLoRA — Law Generation Adapter

## Model Description

This is a **QLoRA fine-tuned adapter** for `mistralai/Mistral-7B-v0.1`, trained via Supervised Fine-Tuning (SFT) for structured AI ethics law generation.

The adapter enables the base Mistral 7B model to generate comprehensive legal documents with Articles, Definitions, Enforcement mechanisms, and Scope sections — eliminating the need for template-based law generation.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Base Model** | `mistralai/Mistral-7B-v0.1` |
| **Fine-Tuning Method** | QLoRA (Quantized Low-Rank Adaptation) |
| **Framework** | TRL 0.28.0 + PEFT + Transformers 5.1.0 |
| **LoRA Rank (r)** | 16 |
| **LoRA Alpha** | 32 |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **LoRA Dropout** | 0.05 |
| **Bias** | none |
| **Task Type** | CAUSAL_LM |

### Quantization

| Parameter | Value |
|-----------|-------|
| **Quantization** | BitsAndBytes 4-bit |
| **Quant Type** | NormalFloat4 (nf4) |
| **Double Quantization** | Enabled |
| **Compute Dtype** | bfloat16 |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Epochs** | 3 |
| **Learning Rate** | 2e-4 |
| **Batch Size** | 1 |
| **Gradient Accumulation** | 4 steps |
| **Max Sequence Length** | 2048 tokens |
| **Optimizer** | AdamW (paged, 8-bit) |
| **Seed** | 42 |
| **Total FLOPs** | 527.3 T |

### Hardware

| Parameter | Value |
|-----------|-------|
| **GPU** | NVIDIA RTX 4060 (8 GB VRAM) |
| **VRAM Used** | ~4 GB |
| **CUDA** | 12.1 |

## Training Data

5 comprehensive law examples covering the ethical domains:
- **Bias** — Fair AI Employment Practices Act (algorithmic bias in hiring)
- **Privacy** — AI Data Protection Standards (training data privacy and consent)
- **Transparency** — Algorithmic Accountability and Transparency Act (explainability)
- **Safety** — AI Safety in Healthcare Applications (autonomous systems standards)
- **General** — Universal AI Ethics Framework (foundational principles)

Each training example is a complete multi-article legal document (500-2000+ words) with:
- Definitions and Scope
- Requirements
- Oversight and Transparency provisions
- Individual Rights
- Technical Standards
- Enforcement mechanisms (specific penalties)
- Implementation timelines
- Examples and guidance

**Prompt Template:**
```
<s>[INST] You are an AI ethics legal expert. Generate a comprehensive law or regulation based on the following request:

{instruction}

Domain: {domain} [/INST]

{output}</s>
```

## Adapter Files

| File | Size | Purpose |
|------|------|---------|
| `adapter_model.safetensors` | ~80 MB | LoRA weight matrices |
| `adapter_config.json` | 1.1 KB | PEFT/LoRA configuration |
| `tokenizer.json` | 3.4 MB | Tokenizer vocabulary |
| `tokenizer_config.json` | 0.5 KB | Tokenizer settings |
| `training_args.bin` | 5.1 KB | Training arguments |

## Production Performance

| Metric | Value |
|--------|-------|
| **Laws Generated** | 38 per cycle (100% LLM) |
| **Template Fallback** | 0 (eliminated) |
| **Law Structure** | 100% include Articles + Definitions + Enforcement + Scope |
| **Evaluation Scorecard** | 7/7 |
| **GPU Inference** | cuda:0, ~4 GB VRAM |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Load base model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "models/mistral_law_generator/final")
tokenizer = AutoTokenizer.from_pretrained("models/mistral_law_generator/final")
```

## Citation

```bibtex
@software{perpetual_ethical_mas_qlora,
  title={Fine-Tuned Mistral 7B QLoRA for AI Ethics Law Generation},
  year={2026},
  method={QLoRA (r=16, alpha=32, 4-bit NF4)},
  base_model={mistralai/Mistral-7B-v0.1}
}
```
