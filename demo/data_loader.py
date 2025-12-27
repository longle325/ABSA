"""
Data Loader for Demo Application
Loads cleaned_data.jsonl and provides lookup functionality.
"""

import json
from pathlib import Path
from typing import Dict, Optional

# Path to cleaned data
DATA_PATH = Path(__file__).parent.parent / "cleaned_data" / "cleaned_data.jsonl"

# Global cache for data
_data_cache: Dict[str, dict] = {}
_loaded = False


def load_data():
    """Load cleaned_data.jsonl into memory."""
    global _data_cache, _loaded

    if _loaded:
        return

    if not DATA_PATH.exists():
        print(f"Warning: {DATA_PATH} not found")
        _loaded = True
        return

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                # Index by cleaned text for O(1) lookup
                _data_cache[sample['text']] = sample

    _loaded = True
    print(f"Loaded {len(_data_cache)} samples from dataset")


def lookup_text(cleaned_text: str) -> Optional[dict]:
    """
    Lookup a cleaned text in the dataset.

    Args:
        cleaned_text: Text after preprocessing

    Returns:
        Sample dict if found, None otherwise
    """
    if not _loaded:
        load_data()

    return _data_cache.get(cleaned_text)


def get_sample_texts(n: int = 5) -> list:
    """Get n sample texts for demo/examples."""
    if not _loaded:
        load_data()

    samples = list(_data_cache.values())[:n]
    return [s['original_text'] for s in samples]


# Load data on module import
load_data()
