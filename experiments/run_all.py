#!/usr/bin/env python
"""
Unified training script for all Vietnamese ABSA models.

Usage:
    python experiments/run_all.py --model crf --data cleaned_data/cleaned_data.jsonl
    python experiments/run_all.py --model logreg --data cleaned_data/cleaned_data.jsonl
    python experiments/run_all.py --model svm --data cleaned_data/cleaned_data.jsonl
    python experiments/run_all.py --model phobert --data cleaned_data/cleaned_data.jsonl
    python experiments/run_all.py --model all --data cleaned_data/cleaned_data.jsonl
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import ABSADataset, load_dataset
from src.evaluation.metrics import print_evaluation_report, evaluate_sequence_labeling


def train_crf(train_data, dev_data, test_data, config, output_dir):
    """Train CRF model."""
    from src.models.crf_model import CRFModel

    print("\n" + "=" * 60)
    print("Training CRF Model")
    print("=" * 60)

    model = CRFModel(config.get('crf', {}))

    results = model.train(
        train_data.get_all_tokens(),
        train_data.get_all_tags(),
        dev_data.get_all_tokens() if dev_data else None,
        dev_data.get_all_tags() if dev_data else None
    )

    # Evaluate on test
    if test_data:
        predictions = model.predict(test_data.get_all_tokens())
        test_metrics = evaluate_sequence_labeling(test_data.get_all_tags(), predictions)
        results['test_f1_macro'] = test_metrics['f1']
        print(f"Test F1 (macro): {test_metrics['f1']:.4f}")

    # Save model
    model_path = output_dir / 'crf_model.pkl'
    model.save(str(model_path))
    print(f"Model saved to: {model_path}")

    return results


def train_logreg(train_data, dev_data, test_data, config, output_dir):
    """Train Logistic Regression model."""
    from src.models.logreg_model import LogRegModel

    print("\n" + "=" * 60)
    print("Training Logistic Regression Model")
    print("=" * 60)

    model = LogRegModel(config.get('logreg', {}))

    results = model.train(
        train_data.get_all_tokens(),
        train_data.get_all_tags(),
        dev_data.get_all_tokens() if dev_data else None,
        dev_data.get_all_tags() if dev_data else None
    )

    # Evaluate on test
    if test_data:
        predictions = model.predict(test_data.get_all_tokens())
        test_metrics = evaluate_sequence_labeling(test_data.get_all_tags(), predictions)
        results['test_f1_macro'] = test_metrics['f1']
        print(f"Test F1 (macro): {test_metrics['f1']:.4f}")

    # Save model
    model_path = output_dir / 'logreg_model.pkl'
    model.save(str(model_path))
    print(f"Model saved to: {model_path}")

    return results


def train_svm(train_data, dev_data, test_data, config, output_dir):
    """Train SVM model."""
    from src.models.svm_model import SVMModel

    print("\n" + "=" * 60)
    print("Training SVM Model")
    print("=" * 60)

    model = SVMModel(config.get('svm', {}))

    results = model.train(
        train_data.get_all_tokens(),
        train_data.get_all_tags(),
        dev_data.get_all_tokens() if dev_data else None,
        dev_data.get_all_tags() if dev_data else None
    )

    # Evaluate on test
    if test_data:
        predictions = model.predict(test_data.get_all_tokens())
        test_metrics = evaluate_sequence_labeling(test_data.get_all_tags(), predictions)
        results['test_f1_macro'] = test_metrics['f1']
        print(f"Test F1 (macro): {test_metrics['f1']:.4f}")

    # Save model
    model_path = output_dir / 'svm_model.pkl'
    model.save(str(model_path))
    print(f"Model saved to: {model_path}")

    return results


def train_phobert(train_data, dev_data, test_data, config, output_dir):
    """Train PhoBERT + BiLSTM-CRF model."""
    try:
        from src.models.phobert_bilstm_crf import PhoBERTModel
    except ImportError as e:
        print(f"Error: Cannot import PhoBERT model: {e}")
        print("Install with: pip install transformers pytorch-crf torch")
        return {'error': str(e)}

    print("\n" + "=" * 60)
    print("Training PhoBERT + BiLSTM-CRF Model")
    print("=" * 60)

    model = PhoBERTModel(config.get('phobert', {}))

    results = model.train(
        train_data.get_all_tokens(),
        train_data.get_all_tags(),
        dev_data.get_all_tokens() if dev_data else None,
        dev_data.get_all_tags() if dev_data else None
    )

    # Evaluate on test
    if test_data:
        predictions = model.predict(test_data.get_all_tokens())
        test_metrics = evaluate_sequence_labeling(test_data.get_all_tags(), predictions)
        results['test_f1_macro'] = test_metrics['f1']
        print(f"Test F1 (macro): {test_metrics['f1']:.4f}")

    # Save model
    model_path = output_dir / 'phobert_model.pt'
    model.save(str(model_path))
    print(f"Model saved to: {model_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train Vietnamese ABSA models')

    # Model selection
    parser.add_argument('--model', type=str, required=True,
                        choices=['crf', 'logreg', 'svm', 'phobert', 'all'],
                        help='Model to train')

    # Data arguments
    parser.add_argument('--data', type=str, default='cleaned_data/cleaned_data.jsonl',
                        help='Path to combined JSONL data with split field')
    parser.add_argument('--use-simple-tokenizer', action='store_true',
                        help='Use simple whitespace tokenizer')

    # Output
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')

    args = parser.parse_args()

    # Load config
    config = {}
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Default configs
    default_config = {
        'crf': {
            'c1': 0.1,
            'c2': 0.1,
            'max_iterations': 200,
            'context_window': 2
        },
        'logreg': {
            'max_features': 10000,
            'ngram_range': (1, 2),
            'context_window': 2,
            'C': 1.0
        },
        'svm': {
            'max_features': 15000,
            'ngram_range': (1, 3),
            'context_window': 3,
            'kernel': 'rbf',
            'C': 1.0
        },
        'phobert': {
            'epochs': 20,
            'batch_size': 16,
            'bert_lr': 2e-5,
            'lstm_lr': 1e-3,
            'crf_lr': 1e-2,
            'lstm_hidden': 256,
            'lstm_layers': 2,
            'dropout': 0.1,
            'max_length': 256
        }
    }

    # Merge configs
    for model_name in default_config:
        if model_name not in config:
            config[model_name] = default_config[model_name]
        else:
            for key, value in default_config[model_name].items():
                if key not in config[model_name]:
                    config[model_name][key] = value

    # Load data
    print("Loading data...")
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)

    train_data, dev_data, test_data = load_dataset(args.data)

    print(f"Train samples: {len(train_data)}")
    print(f"Dev samples: {len(dev_data)}")
    print(f"Test samples: {len(test_data)}")

    # Setup output directory
    output_dir = Path(args.output_dir) / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train models
    all_results = {}
    models_to_train = ['crf', 'logreg', 'svm', 'phobert'] if args.model == 'all' else [args.model]

    for model_name in models_to_train:
        if model_name == 'crf':
            results = train_crf(train_data, dev_data, test_data, config, output_dir)
        elif model_name == 'logreg':
            results = train_logreg(train_data, dev_data, test_data, config, output_dir)
        elif model_name == 'svm':
            results = train_svm(train_data, dev_data, test_data, config, output_dir)
        elif model_name == 'phobert':
            results = train_phobert(train_data, dev_data, test_data, config, output_dir)

        all_results[model_name] = results

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    print(f"\n{'Model':<20} {'Dev F1':>12} {'Test F1':>12}")
    print("-" * 44)

    for model_name, results in all_results.items():
        dev_f1 = results.get('dev_f1_macro', results.get('best_dev_f1', 'N/A'))
        test_f1 = results.get('test_f1_macro', 'N/A')

        if isinstance(dev_f1, float):
            dev_f1 = f"{dev_f1:.4f}"
        if isinstance(test_f1, float):
            test_f1 = f"{test_f1:.4f}"

        print(f"{model_name:<20} {dev_f1:>12} {test_f1:>12}")

    # Save results
    results_path = Path(args.output_dir) / 'logs' / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
