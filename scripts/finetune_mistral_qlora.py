"""
Mistral 7B QLoRA Fine-Tuning Script

Fine-tunes Mistral 7B using QLoRA (Quantized Low-Rank Adaptation) for
AI ethics law generation on RTX 4060 8GB GPU.

QLoRA enables fine-tuning 7B models on consumer GPUs by:
1. Loading model in 4-bit quantization (~3.5GB instead of 14GB)
2. Training only small LoRA adapters (<<1% of parameters)
3. Using paged optimizers to offload to CPU RAM

Expected VRAM usage: 5-6 GB (fits on 8GB GPU!)

Usage:
    # Basic fine-tuning
    python scripts/finetune_mistral_qlora.py
    
    # Custom settings
    python scripts/finetune_mistral_qlora.py --epochs 5 --batch_size 2 --rank 32
    
    # Resume from checkpoint
    python scripts/finetune_mistral_qlora.py --resume models/mistral_law_generator/checkpoint-100
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_gpu():
    """Check GPU availability and VRAM"""
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available! QLoRA requires GPU.")
        logger.error("   Install CUDA: https://developer.nvidia.com/cuda-downloads")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    logger.info(f"✓ GPU detected: {gpu_name}")
    logger.info(f"  VRAM: {gpu_memory:.1f} GB")
    
    if gpu_memory < 6:
        logger.warning(f"⚠️  Low VRAM ({gpu_memory:.1f} GB). QLoRA needs 6+ GB.")
        logger.warning("   Consider reducing batch size or using gradient checkpointing")
    
    return True


def load_training_data(data_path: str) -> Dataset:
    """
    Load and format training data for law generation
    
    Expected format (JSONL):
    {"instruction": "...", "output": "...", "domain": "..."}
    
    Or JSON array:
    [{"instruction": "...", "output": "...", "domain": "..."}, ...]
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    logger.info(f"Loading training data from: {data_path}")
    
    # Load data
    if data_path.suffix == '.jsonl':
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:  # .json
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    logger.info(f"✓ Loaded {len(data)} training examples")
    
    # Format for SFT (Supervised Fine-Tuning)
    formatted_data = []
    for item in data:
        instruction = item.get('instruction', '')
        output = item.get('output', '')
        domain = item.get('domain', 'general')
        
        # Create prompt template
        text = f"""<s>[INST] You are an AI ethics legal expert. Generate a comprehensive law or regulation based on the following request:

{instruction}

Domain: {domain} [/INST]

{output}</s>"""
        
        formatted_data.append({'text': text})
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(formatted_data)
    
    logger.info(f"✓ Dataset formatted: {len(dataset)} samples")
    
    return dataset


def create_qlora_config(rank: int = 16, alpha: int = 32) -> LoraConfig:
    """
    Create QLoRA configuration
    
    Args:
        rank: LoRA rank (higher = more capacity, more VRAM)
        alpha: LoRA alpha (scaling factor)
    
    Returns:
        LoraConfig object
    """
    return LoraConfig(
        r=rank,  # Rank of LoRA matrices
        lora_alpha=alpha,  # Scaling factor
        target_modules=[
            "q_proj",   # Query projection
            "v_proj",   # Value projection  
            "k_proj",   # Key projection
            "o_proj",   # Output projection
            "gate_proj",  # Gate projection (Mistral-specific)
            "up_proj",    # Up projection (Mistral-specific)
            "down_proj"   # Down projection (Mistral-specific)
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )


def create_quantization_config() -> BitsAndBytesConfig:
    """
    Create 4-bit quantization configuration for QLoRA
    
    Uses NormalFloat4 (nf4) quantization optimized for neural networks
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,  # Load model in 4-bit
        bnb_4bit_use_double_quant=True,  # Double quantization for extra compression
        bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization
        bnb_4bit_compute_dtype=torch.bfloat16  # Compute in bfloat16 for stability
    )


def finetune_mistral(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    data_path: str = "data/law_generation_training.jsonl",
    output_dir: str = "models/mistral_law_generator",
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    max_seq_length: int = 2048,
    resume_from_checkpoint: str = None,
    seed: int = 42
):
    """
    Fine-tune Mistral 7B using QLoRA
    
    Args:
        model_name: HuggingFace model ID
        data_path: Path to training data (JSONL or JSON)
        output_dir: Directory to save fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        lora_rank: LoRA rank (16 = good balance)
        lora_alpha: LoRA alpha (32 = 2x rank)
        max_seq_length: Maximum sequence length
        resume_from_checkpoint: Resume from checkpoint path
        seed: Random seed
    """
    set_seed(seed)
    
    logger.info("=" * 70)
    logger.info("MISTRAL 7B QLORA FINE-TUNING")
    logger.info("=" * 70)
    
    # Check GPU
    if not check_gpu():
        logger.error("Exiting: GPU required for QLoRA")
        return
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    logger.info("\n📊 Loading training data...")
    dataset = load_training_data(data_path)
    
    # Split train/val (90/10)
    split = dataset.train_test_split(test_size=0.1, seed=seed)
    train_dataset = split['train']
    val_dataset = split['test']
    
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val:   {len(val_dataset)} samples")
    
    # Load tokenizer
    logger.info("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info(f"✓ Tokenizer loaded: {model_name}")
    
    # Load model with 4-bit quantization
    logger.info("\n🤖 Loading Mistral 7B in 4-bit...")
    bnb_config = create_quantization_config()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    logger.info("✓ Model loaded in 4-bit quantization")
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    logger.info("✓ Model prepared for k-bit training")
    
    # Create LoRA config (will be used by SFTTrainer)
    logger.info(f"\n🔧 Configuring LoRA adapters (rank={lora_rank}, alpha={lora_alpha})...")
    lora_config = create_qlora_config(rank=lora_rank, alpha=lora_alpha)
    logger.info("✓ LoRA config created")
    
    # Training arguments
    logger.info("\n⚙️  Configuring training...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,  # Saves VRAM
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=False,  # Don't use fp16 with bfloat16
        bf16=True,   # Use bfloat16
        logging_steps=5,
        logging_dir=str(output_dir / "logs"),
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        eval_strategy="steps",  # Updated from evaluation_strategy
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Disable wandb/tensorboard
        seed=seed
    )
    
    # Effective batch size
    effective_batch_size = batch_size * gradient_accumulation_steps
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {effective_batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Max sequence length: {max_seq_length}")
    
    # Create trainer
    logger.info("\n🚀 Creating SFT trainer...")
    
    # Formatting function for the new API
    def formatting_func(example):
        return example["text"]
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        formatting_func=formatting_func,
        args=training_args,
        processing_class=tokenizer,
    )
    
    logger.info("✓ Trainer created")
    
    # Start training
    logger.info("\n" + "=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    if resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    logger.info("=" * 70)
    logger.info(f"✓ TRAINING COMPLETED in {duration:.1f} minutes")
    logger.info("=" * 70)
    
    # Save final model
    logger.info("\n💾 Saving final model...")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    
    logger.info(f"✓ Model saved to: {output_dir / 'final'}")
    logger.info("\nTo use the fine-tuned model:")
    logger.info(f"  from peft import PeftModel")
    logger.info(f"  model = AutoModelForCausalLM.from_pretrained('{model_name}')")
    logger.info(f"  model = PeftModel.from_pretrained(model, '{output_dir / 'final'}')")
    
    logger.info("\n🎉 Fine-tuning complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Mistral 7B using QLoRA for AI ethics law generation"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="HuggingFace model ID (default: mistralai/Mistral-7B-v0.1)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/law_generation_training.jsonl",
        help="Path to training data (JSONL or JSON)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/mistral_law_generator",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device batch size (default: 1)"
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    finetune_mistral(
        model_name=args.model_name,
        data_path=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        max_seq_length=args.max_length,
        resume_from_checkpoint=args.resume,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
