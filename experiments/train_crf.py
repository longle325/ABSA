#!/usr/bin/env python
"""
Training script for CRF model on Vietnamese ABSA dataset.

Usage:
    python experiments/train_crf.py --data cleaned_data/cleaned_data.jsonl
    python experiments/train_crf.py --train UIT-ViSD4SA/data/train.jsonl --dev UIT-ViSD4SA/data/dev.jsonl --test UIT-ViSD4SA/data/test.jsonl
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import ABSADataset, load_dataset_with_split
from src.models.crf_model import CRFModel
from src.evaluation.metrics import print_evaluation_report


def main():
    parser = argparse.ArgumentParser(description='Train CRF model for Vietnamese ABSA')

    # Data arguments
    parser.add_argument('--data', type=str, default=None,
                        help='Path to combined JSONL data with split field')
    parser.add_argument('--train', type=str, default=None,
                        help='Path to training data')
    parser.add_argument('--dev', type=str, default=None,
                        help='Path to development data')
    parser.add_argument('--test', type=str, default=None,
                        help='Path to test data')

    # Model arguments
    parser.add_argument('--c1', type=float, default=0.1,
                        help='L1 regularization (default: 0.1)')
    parser.add_argument('--c2', type=float, default=0.1,
                        help='L2 regularization (default: 0.1)')
    parser.add_argument('--max-iter', type=int, default=200,
                        help='Max iterations (default: 200)')
    parser.add_argument('--context-window', type=int, default=2,
                        help='Context window size (default: 2)')

    # Output arguments
    parser.add_argument('--output', type=str, default='outputs/models/crf_model.pkl',
                        help='Path to save model')
    parser.add_argument('--use-simple-tokenizer', action='store_true',
                        help='Use simple whitespace tokenizer instead of underthesea')

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    if args.data:
        train_data, dev_data, test_data = load_dataset_with_split(args.data)
    elif args.train:
        train_data = ABSADataset(args.train, use_simple_tokenizer=args.use_simple_tokenizer)
        dev_data = ABSADataset(args.dev, use_simple_tokenizer=args.use_simple_tokenizer) if args.dev else None
        test_data = ABSADataset(args.test, use_simple_tokenizer=args.use_simple_tokenizer) if args.test else None
    else:
        # Default path
        default_path = 'cleaned_data/cleaned_data.jsonl'
        if os.path.exists(default_path):
            train_data, dev_data, test_data = load_dataset_with_split(default_path)
        else:
            print(f"Error: No data file specified and default {default_path} not found")
            print("Use --data or --train/--dev/--test to specify data files")
            sys.exit(1)

    print(f"Train samples: {len(train_data)}")
    if dev_data:
        print(f"Dev samples: {len(dev_data)}")
    if test_data:
        print(f"Test samples: {len(test_data)}")

    # Create model
    config = {
        'c1': args.c1,
        'c2': args.c2,
        'max_iterations': args.max_iter,
        'context_window': args.context_window
    }
    model = CRFModel(config)

    # Prepare data
    train_tokens = train_data.get_all_tokens()
    train_tags = train_data.get_all_tags()

    dev_tokens = dev_data.get_all_tokens() if dev_data else None
    dev_tags = dev_data.get_all_tags() if dev_data else None

    # Train
    print("\n" + "=" * 60)
    print("Training CRF Model")
    print("=" * 60)

    results = model.train(train_tokens, train_tags, dev_tokens, dev_tags)

    # Evaluate on test set
    if test_data:
        print("\n" + "=" * 60)
        print("Evaluating on Test Set")
        print("=" * 60)

        test_tokens = test_data.get_all_tokens()
        test_tags = test_data.get_all_tags()

        predictions = model.predict(test_tokens)
        print_evaluation_report(test_tags, predictions)

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    print(f"\nModel saved to: {output_path}")

    # Show top features
    print("\n" + "=" * 60)
    print("Top Transitions")
    print("=" * 60)
    for weight, from_label, to_label in model.get_top_transitions(10):
        print(f"  {weight:+.4f}: {from_label} -> {to_label}")

    print("\n" + "=" * 60)
    print("Top Features")
    print("=" * 60)
    for weight, feature, label in model.get_top_features(10):
        print(f"  {weight:+.4f}: {feature} -> {label}")


if __name__ == '__main__':
    main()
