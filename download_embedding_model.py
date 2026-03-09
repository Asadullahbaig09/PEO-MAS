"""
Download Embedding Model Script

This script downloads the sentence-transformers embedding model
to the local models directory for offline use.

Usage:
    python download_embedding_model.py
"""

from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings


def download_model():
    """Download embedding model to configured path"""
    
    model_path = settings.EMBEDDING_MODEL_PATH
    model_name = settings.EMBEDDING_MODEL
    
    print("="*70)
    print("  EMBEDDING MODEL DOWNLOAD")
    print("="*70)
    print(f"\nModel: {model_name}")
    print(f"Destination: {model_path}")
    
    # Check if already exists
    if model_path.exists():
        print(f"\n⚠️  Model already exists at: {model_path}")
        response = input("Download anyway? (y/n): ")
        if response.lower() != 'y':
            print("\nDownload cancelled.")
            return
    
    # Create directory
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download model
    print(f"\n📥 Downloading {model_name}...")
    print("This may take a few minutes depending on your connection...")
    
    try:
        model = SentenceTransformer(
            model_name,
            cache_folder=str(model_path.parent)
        )
        
        print(f"\n✓ Model downloaded successfully!")
        print(f"✓ Location: {model_path}")
        
        # Test the model
        print("\n🧪 Testing model...")
        test_text = "This is a test sentence for the embedding model."
        embedding = model.encode(test_text, convert_to_numpy=True)
        
        print(f"✓ Model test successful!")
        print(f"  - Embedding dimension: {embedding.shape[0]}")
        print(f"  - Model ready to use")
        
        print("\n" + "="*70)
        print("  DOWNLOAD COMPLETE")
        print("="*70)
        print("\nYou can now run the system with:")
        print("  python scripts/run_system.py")
        print("\nThe model will load instantly from the local cache.")
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Ensure you have write permissions to the models directory")
        print("  3. Try: pip install --upgrade sentence-transformers")
        sys.exit(1)


if __name__ == "__main__":
    download_model()
