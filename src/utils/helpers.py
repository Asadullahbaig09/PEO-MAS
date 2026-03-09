import hashlib
import json
from datetime import datetime
from typing import Any, Dict
from pathlib import Path


def generate_id(prefix: str, content: str) -> str:
    """Generate unique ID from content"""
    hash_obj = hashlib.md5(f"{prefix}:{content}".encode())
    return f"{prefix}_{hash_obj.hexdigest()[:12]}"


def save_json(data: Dict[str, Any], filepath: str):
    """Save data to JSON file"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def timestamp_str() -> str:
    """Get formatted timestamp string"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')
