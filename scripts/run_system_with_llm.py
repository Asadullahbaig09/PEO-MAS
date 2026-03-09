"""
Run system with proper GPU memory management for fine-tuned LLM

This script ensures the fine-tuned model loads successfully by:
1. Clearing any existing GPU memory before starting
2. Using a single Python process (not spawning subprocesses)
3. Properly releasing resources after completion
"""

import gc
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def clear_gpu_memory():
    """Clear GPU memory from any previous sessions"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print("✓ GPU memory cleared")

def main():
    # Clear GPU memory first
    print("\n" + "="*70)
    print("  PREPARING GPU MEMORY FOR FINE-TUNED MODEL")
    print("="*70)
    clear_gpu_memory()
    
    # Import after clearing memory
    from scripts.run_system import main as run_system_main
    
    # Run the system
    run_system_main()
    
    # Cleanup after execution
    print("\n" + "="*70)
    print("  CLEANING UP GPU MEMORY")
    print("="*70)
    clear_gpu_memory()

if __name__ == "__main__":
    main()
