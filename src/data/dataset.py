"""
Dataset Classes for Vietnamese Aspect-Based Sentiment Analysis

Simple implementation with pickle caching for fast loading.
"""

import json
import pickle
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from tqdm import tqdm
import random

from .bio_converter import char_span_to_bio_tags, build_label_vocab
from .tokenizer import tokenize_with_offsets, simple_tokenize_with_offsets


class ABSADataset:
    """
    Dataset class for Vietnamese ABSA sequence labeling.

    Features:
    - Pickle caching for fast subsequent loads
    - Progress bar during preprocessing
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: Optional[str] = None,
        use_simple_tokenizer: bool = False,
        use_cache: bool = True
    ):
        """
        Initialize ABSA dataset.

        Args:
            data_path: Path to JSONL data file
            split: Filter by split ('train', 'dev', 'test')
            use_simple_tokenizer: Use whitespace tokenizer instead of underthesea
            use_cache: Use pickle cache for faster loading
        """
        self.samples: List[Dict] = []
        self.label2id, self.id2label = build_label_vocab()
        self.split = split
        self.use_cache = use_cache

        # Choose tokenizer
        self.use_simple = use_simple_tokenizer
        if use_simple_tokenizer:
            self.tokenizer = simple_tokenize_with_offsets
        else:
            self.tokenizer = tokenize_with_offsets

        if data_path:
            self.load_data(data_path, split)

    def load_data(self, data_path: str, split: Optional[str] = None) -> None:
        """
        Load data from JSONL file with pickle caching.

        Args:
            data_path: Path to JSONL file
            split: Optional split filter
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Cache path based on tokenizer type
        cache_suffix = '.simple.cache.pkl' if self.use_simple else '.cache.pkl'
        cache_path = Path(str(data_path) + cache_suffix)

        # Try loading from cache
        if self.use_cache and cache_path.exists():
            if cache_path.stat().st_mtime > data_path.stat().st_mtime:
                print(f"Loading from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    all_samples = pickle.load(f)

                # Filter by split if needed
                if split:
                    self.samples = [s for s in all_samples if s.get('split', '') == split]
                else:
                    self.samples = all_samples

                print(f"Loaded {len(self.samples)} samples from cache")
                return

        # Process from scratch
        print(f"Loading and preprocessing: {data_path}")

        # Read all lines first to get count for progress bar
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        all_samples = []
        for idx, line in enumerate(tqdm(lines, desc="Tokenizing")):
            data = json.loads(line)

            sample_id = data.get('id', f'sample_{idx:05d}')
            text = data.get('text', '')
            labels = data.get('labels', [])
            sample_split = data.get('split', '')

            if not text:
                continue

            # Convert labels to tuples
            labels = [(int(l[0]), int(l[1]), l[2]) for l in labels]

            # Convert to BIO tags
            bio_tags, tokens_with_offsets = char_span_to_bio_tags(
                text, labels, self.tokenizer
            )

            # Extract tokens and offsets
            tokens = [t[0] for t in tokens_with_offsets]
            offsets = [(t[1], t[2]) for t in tokens_with_offsets]

            sample = {
                'id': sample_id,
                'text': text,
                'tokens': tokens,
                'token_offsets': offsets,
                'bio_tags': bio_tags,
                'original_labels': labels,
                'split': sample_split
            }
            all_samples.append(sample)

        # Save to cache
        if self.use_cache:
            print(f"Saving cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(all_samples, f)

        # Filter by split if needed
        if split:
            self.samples = [s for s in all_samples if s.get('split', '') == split]
        else:
            self.samples = all_samples

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    def get_all_tokens(self) -> List[List[str]]:
        """Get all token sequences."""
        return [s['tokens'] for s in self.samples]

    def get_all_tags(self) -> List[List[str]]:
        """Get all BIO tag sequences."""
        return [s['bio_tags'] for s in self.samples]

    @property
    def num_labels(self) -> int:
        return len(self.label2id)


def load_dataset_with_split(
    data_path: str,
    train_size: float = 0.8,
    test_size: float = 0.2,
    dev_size: float = 0.0,
    random_seed: int = 42,
    shuffle: bool = True,
    use_simple_tokenizer: bool = False
) -> Tuple[ABSADataset, Optional[ABSADataset], ABSADataset]:
    """
    Load dataset and split into train/dev/test.

    Args:
        data_path: Path to JSONL file
        train_size: Fraction for training
        test_size: Fraction for testing
        dev_size: Fraction for development (0 = no dev set)
        random_seed: Random seed for reproducibility
        shuffle: Shuffle before splitting
        use_simple_tokenizer: Use whitespace tokenizer

    Returns:
        Tuple of (train_data, dev_data, test_data)
    """
    # Load full dataset
    full_dataset = ABSADataset(
        data_path,
        split=None,
        use_simple_tokenizer=use_simple_tokenizer
    )

    samples = full_dataset.samples.copy()

    # Shuffle if required
    if shuffle:
        random.seed(random_seed)
        random.shuffle(samples)

    total = len(samples)
    train_end = int(total * train_size)

    if dev_size > 0:
        dev_end = train_end + int(total * dev_size)
        train_samples = samples[:train_end]
        dev_samples = samples[train_end:dev_end]
        test_samples = samples[dev_end:]
    else:
        train_samples = samples[:train_end]
        dev_samples = []
        test_samples = samples[train_end:]

    # Create dataset objects
    train_data = ABSADataset(use_simple_tokenizer=use_simple_tokenizer)
    train_data.samples = train_samples
    train_data.label2id = full_dataset.label2id
    train_data.id2label = full_dataset.id2label

    dev_data = None
    if dev_samples:
        dev_data = ABSADataset(use_simple_tokenizer=use_simple_tokenizer)
        dev_data.samples = dev_samples
        dev_data.label2id = full_dataset.label2id
        dev_data.id2label = full_dataset.id2label

    test_data = ABSADataset(use_simple_tokenizer=use_simple_tokenizer)
    test_data.samples = test_samples
    test_data.label2id = full_dataset.label2id
    test_data.id2label = full_dataset.id2label

    return train_data, dev_data, test_data


def load_dataset_splits(
    data_path: str,
    train_split: str = 'train',
    dev_split: str = 'dev',
    test_split: str = 'test'
) -> Tuple[ABSADataset, ABSADataset, ABSADataset]:
    """
    Load train/dev/test datasets from a single file with split field.

    Args:
        data_path: Path to JSONL file with split field

    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset)
    """
    train_data = ABSADataset(data_path, split=train_split)
    dev_data = ABSADataset(data_path, split=dev_split)
    test_data = ABSADataset(data_path, split=test_split)

    return train_data, dev_data, test_data


if __name__ == "__main__":
    import sys

    print("Testing ABSA Dataset")
    print("=" * 60)

    data_path = sys.argv[1] if len(sys.argv) > 1 else "cleaned_data/cleaned_data.jsonl"

    try:
        train, dev, test = load_dataset_with_split(
            data_path,
            train_size=0.8,
            test_size=0.2,
            random_seed=42
        )

        print(f"\nTrain: {len(train)} samples")
        print(f"Dev: {len(dev) if dev else 0} samples")
        print(f"Test: {len(test)} samples")

        if len(train) > 0:
            sample = train[0]
            print(f"\nFirst sample:")
            print(f"  Tokens: {sample['tokens'][:10]}...")
            print(f"  BIO tags: {sample['bio_tags'][:10]}...")

    except FileNotFoundError as e:
        print(f"Error: {e}")
