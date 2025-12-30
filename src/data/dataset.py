"""
Dataset Classes for Vietnamese Aspect-Based Sentiment Analysis

Provides unified dataset loading, preprocessing, and iteration
for all models (CRF, LogReg, SVM, PhoBERT).
"""

import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Iterator
from pathlib import Path

from .bio_converter import (
    char_span_to_bio_tags,
    build_label_vocab,
    get_all_labels
)
from .tokenizer import tokenize_with_offsets, simple_tokenize_with_offsets


@dataclass
class ABSASample:
    """Single sample for ABSA sequence labeling."""

    id: str
    text: str
    tokens: List[str]
    token_offsets: List[Tuple[int, int]]
    bio_tags: List[str]
    original_labels: List[Tuple[int, int, str]]
    split: str = ""

    def __len__(self) -> int:
        return len(self.tokens)

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'text': self.text,
            'tokens': self.tokens,
            'token_offsets': self.token_offsets,
            'bio_tags': self.bio_tags,
            'original_labels': self.original_labels,
            'split': self.split
        }


class ABSADataset:
    """
    Dataset class for Vietnamese ABSA sequence labeling.

    Handles data loading, BIO conversion, and provides iteration
    compatible with all model types.
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        split: Optional[str] = None,
        use_simple_tokenizer: bool = False
    ):
        """
        Initialize ABSA dataset.

        Args:
            data_path: Path to JSONL data file
            split: Filter by split ('train', 'dev', 'test') if data has split field
            use_simple_tokenizer: If True, use simple whitespace tokenizer instead of underthesea
        """
        self.samples: List[ABSASample] = []
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.split = split

        # Choose tokenizer
        if use_simple_tokenizer:
            self.tokenizer = simple_tokenize_with_offsets
        else:
            self.tokenizer = tokenize_with_offsets

        # Build label vocabulary
        self.label2id, self.id2label = build_label_vocab()

        # Load data if path provided
        if data_path:
            self.load_data(data_path, split)

    def load_data(self, data_path: str, split: Optional[str] = None) -> None:
        """
        Load data from JSONL file.

        Supports two formats:
        1. Original format: {"text": "...", "labels": [[start, end, "LABEL"], ...]}
        2. Cleaned format: {"id": "...", "text": "...", "labels": [...], "split": "..."}

        Args:
            data_path: Path to JSONL file
            split: Optional split filter
        """
        data_path = Path(data_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.samples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                # Skip if filtering by split
                sample_split = data.get('split', '')
                if split and sample_split and sample_split != split:
                    continue

                # Get sample ID
                sample_id = data.get('id', f'sample_{idx:05d}')

                # Get text and labels
                text = data.get('text', '')
                labels = data.get('labels', [])

                # Skip samples without text
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

                # Create sample
                sample = ABSASample(
                    id=sample_id,
                    text=text,
                    tokens=tokens,
                    token_offsets=offsets,
                    bio_tags=bio_tags,
                    original_labels=labels,
                    split=sample_split
                )

                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ABSASample:
        return self.samples[idx]

    def __iter__(self) -> Iterator[ABSASample]:
        return iter(self.samples)

    def get_all_tokens(self) -> List[List[str]]:
        """Get all token sequences."""
        return [s.tokens for s in self.samples]

    def get_all_tags(self) -> List[List[str]]:
        """Get all BIO tag sequences."""
        return [s.bio_tags for s in self.samples]

    def get_label_ids(self, bio_tags: List[str]) -> List[int]:
        """Convert BIO tags to label IDs."""
        return [self.label2id.get(tag, self.label2id['O']) for tag in bio_tags]

    def get_label_names(self, label_ids: List[int]) -> List[str]:
        """Convert label IDs to BIO tags."""
        return [self.id2label.get(i, 'O') for i in label_ids]

    @property
    def num_labels(self) -> int:
        """Get number of unique labels."""
        return len(self.label2id)

    def get_statistics(self) -> Dict:
        """
        Compute dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'num_samples': len(self.samples),
            'num_labels': self.num_labels,
            'aspect_counts': {},
            'sentiment_counts': {},
            'label_counts': {}
        }

        for sample in self.samples:
            for tag in sample.bio_tags:
                if tag != 'O':
                    # Count full label
                    label = tag[2:]  # Remove B- or I-
                    stats['label_counts'][label] = stats['label_counts'].get(label, 0) + 1

                    # Parse aspect and sentiment
                    if '#' in label:
                        aspect, sentiment = label.split('#')
                        stats['aspect_counts'][aspect] = stats['aspect_counts'].get(aspect, 0) + 1
                        stats['sentiment_counts'][sentiment] = stats['sentiment_counts'].get(sentiment, 0) + 1

        return stats

    def print_statistics(self) -> None:
        """Print dataset statistics."""
        stats = self.get_statistics()

        print(f"\nDataset Statistics")
        print("=" * 40)
        print(f"Number of samples: {stats['num_samples']}")
        print(f"Number of unique labels: {stats['num_labels']}")

        print(f"\nAspect distribution:")
        for aspect, count in sorted(stats['aspect_counts'].items(), key=lambda x: -x[1]):
            print(f"  {aspect}: {count}")

        print(f"\nSentiment distribution:")
        for sentiment, count in sorted(stats['sentiment_counts'].items(), key=lambda x: -x[1]):
            print(f"  {sentiment}: {count}")


def load_dataset(
    data_path: str,
    train_split: str = 'train',
    dev_split: str = 'dev',
    test_split: str = 'test'
) -> Tuple[ABSADataset, ABSADataset, ABSADataset]:
    """
    Load train/dev/test datasets from a single file with split field.

    Args:
        data_path: Path to JSONL file with split field
        train_split: Name of train split
        dev_split: Name of dev split
        test_split: Name of test split

    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset)
    """
    train_data = ABSADataset(data_path, split=train_split)
    dev_data = ABSADataset(data_path, split=dev_split)
    test_data = ABSADataset(data_path, split=test_split)

    return train_data, dev_data, test_data


def load_separate_datasets(
    train_path: str,
    dev_path: str,
    test_path: str
) -> Tuple[ABSADataset, ABSADataset, ABSADataset]:
    """
    Load train/dev/test datasets from separate files.

    Args:
        train_path: Path to training data
        dev_path: Path to development data
        test_path: Path to test data

    Returns:
        Tuple of (train_dataset, dev_dataset, test_dataset)
    """
    train_data = ABSADataset(train_path)
    dev_data = ABSADataset(dev_path)
    test_data = ABSADataset(test_path)

    return train_data, dev_data, test_data


if __name__ == "__main__":
    import sys

    # Test with sample data
    print("Testing ABSA Dataset")
    print("=" * 60)

    # Check if data path provided
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # Default path
        data_path = "cleaned_data/cleaned_data.jsonl"

    try:
        # Try loading with split
        train_data = ABSADataset(data_path, split='train', use_simple_tokenizer=True)

        print(f"\nLoaded {len(train_data)} training samples")

        if len(train_data) > 0:
            # Show first sample
            sample = train_data[0]
            print(f"\nFirst sample:")
            print(f"  ID: {sample.id}")
            print(f"  Text: {sample.text[:80]}...")
            print(f"  Tokens: {sample.tokens[:10]}...")
            print(f"  BIO tags: {sample.bio_tags[:10]}...")

            # Print statistics
            train_data.print_statistics()

    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        print("Using sample data for testing...")

        # Create sample data
        sample_text = "Pin cực trâu, camera chụp đẹp"
        sample_labels = [(0, 12, "BATTERY#POSITIVE"), (14, 28, "CAMERA#POSITIVE")]

        bio_tags, tokens = char_span_to_bio_tags(sample_text, sample_labels)

        print(f"\nSample text: {sample_text}")
        print(f"Tokens: {[t[0] for t in tokens]}")
        print(f"BIO tags: {bio_tags}")
