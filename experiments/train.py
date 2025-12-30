#!/usr/bin/env python
"""
Unified Training Script for Vietnamese ABSA Models

Usage:
    python experiments/train.py --model crf
    python experiments/train.py --model crf --data-config configs/data_config.yaml
    python experiments/train.py --model all
"""

import argparse
import sys
import json
import random
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_data_config, load_model_config
from src.data.dataset import ABSADataset
from src.evaluation.metrics import evaluate_sequence_labeling, print_evaluation_report


def split_data(dataset: ABSADataset, config: dict):
    """
    Split dataset according to config settings.

    Args:
        dataset: Full dataset
        config: Data configuration with split settings

    Returns:
        Tuple of (train_data, dev_data, test_data)
    """
    split_config = config['split']

    train_size = split_config['train_size']
    test_size = split_config['test_size']
    dev_size = split_config.get('dev_size', 0.0)
    random_seed = split_config['random_seed']
    shuffle = split_config.get('shuffle', True)

    samples = dataset.samples.copy()

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
    train_data = ABSADataset()
    train_data.samples = train_samples
    train_data.label2id = dataset.label2id
    train_data.id2label = dataset.id2label

    dev_data = None
    if dev_samples:
        dev_data = ABSADataset()
        dev_data.samples = dev_samples
        dev_data.label2id = dataset.label2id
        dev_data.id2label = dataset.id2label

    test_data = ABSADataset()
    test_data.samples = test_samples
    test_data.label2id = dataset.label2id
    test_data.id2label = dataset.id2label

    return train_data, dev_data, test_data


def train_model(model_name: str, train_data, dev_data, test_data, model_config: dict, output_dir: Path):
    """
    Train a specific model.

    Args:
        model_name: Model name (crf, logreg, svm, phobert)
        train_data: Training dataset
        dev_data: Development dataset (optional)
        test_data: Test dataset
        model_config: Model configuration
        output_dir: Output directory for saving model

    Returns:
        Results dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"Training {model_name.upper()}")
    print(f"{'=' * 60}")

    # Get tokens and tags
    train_tokens = train_data.get_all_tokens()
    train_tags = train_data.get_all_tags()

    dev_tokens = dev_data.get_all_tokens() if dev_data else None
    dev_tags = dev_data.get_all_tags() if dev_data else None

    test_tokens = test_data.get_all_tokens()
    test_tags = test_data.get_all_tags()

    # Create and train model based on type
    if model_name == 'crf':
        from src.models.crf_model import CRFModel

        # Extract config with defaults
        features = model_config.get('features', {})
        hyperparams = model_config.get('hyperparameters', {})

        config = {
            'c1': hyperparams.get('c1', 0.1),
            'c2': hyperparams.get('c2', 0.1),
            'max_iterations': hyperparams.get('max_iterations', 200),
            'context_window': features.get('context_window', 2)
        }

        model = CRFModel(config)
        results = model.train(train_tokens, train_tags, dev_tokens, dev_tags)
        model.save(str(output_dir / 'crf_model.pkl'))

    elif model_name == 'logreg':
        from src.models.logreg_model import LogRegModel

        # Extract config with defaults
        features = model_config.get('features', {})
        hyperparams = model_config.get('hyperparameters', {})
        ngram = features.get('ngram_range', [1, 2])

        config = {
            'max_features': features.get('max_features', 10000),
            'ngram_range': tuple(ngram) if isinstance(ngram, list) else ngram,
            'context_window': features.get('context_window', 2),
            'C': hyperparams.get('C', 1.0),
            'max_iter': hyperparams.get('max_iter', 1000)
        }

        model = LogRegModel(config)
        results = model.train(train_tokens, train_tags, dev_tokens, dev_tags)
        model.save(str(output_dir / 'logreg_model.pkl'))

    elif model_name == 'svm':
        from src.models.svm_model import SVMModel

        # Extract config with defaults
        features = model_config.get('features', {})
        hyperparams = model_config.get('hyperparameters', {})
        ngram = features.get('ngram_range', [1, 3])

        config = {
            'max_features': features.get('max_features', 15000),
            'ngram_range': tuple(ngram) if isinstance(ngram, list) else ngram,
            'context_window': features.get('context_window', 3),
            'kernel': hyperparams.get('kernel', 'linear'),
            'C': hyperparams.get('C', 1.0),
            'gamma': hyperparams.get('gamma', 'scale')
        }

        model = SVMModel(config)
        results = model.train(train_tokens, train_tags, dev_tokens, dev_tags)
        model.save(str(output_dir / 'svm_model.pkl'))

    elif model_name == 'phobert':
        try:
            from src.models.phobert_bilstm_crf import PhoBERTModel
        except ImportError as e:
            print(f"Error: Cannot import PhoBERT model: {e}")
            return {'error': str(e)}, None, None

        config = {
            'pretrained_model': model_config['model']['pretrained'],
            'lstm_hidden': model_config['architecture']['lstm_hidden'],
            'lstm_layers': model_config['architecture']['lstm_layers'],
            'dropout': model_config['architecture']['dropout'],
            'lstm_dropout': model_config['architecture']['lstm_dropout'],
            'freeze_bert': model_config['architecture']['freeze_bert'],
            'epochs': model_config['training']['epochs'],
            'batch_size': model_config['training']['batch_size'],
            'gradient_clip': model_config['training']['gradient_clip'],
            'bert_lr': model_config['optimizer']['bert_lr'],
            'lstm_lr': model_config['optimizer']['lstm_lr'],
            'crf_lr': model_config['optimizer']['crf_lr'],
            'weight_decay': model_config['optimizer']['weight_decay']
        }

        model = PhoBERTModel(config)
        results = model.train(train_tokens, train_tags, dev_tokens, dev_tags)
        model.save(str(output_dir / 'phobert_model.pt'))

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Evaluate on test set
    predictions = model.predict(test_tokens)
    test_metrics = evaluate_sequence_labeling(test_tags, predictions)

    results['test_precision'] = test_metrics['precision']
    results['test_recall'] = test_metrics['recall']
    results['test_f1_macro'] = test_metrics['f1']

    print(f"\nTest Results:")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Macro:  {test_metrics['f1']:.4f}")

    return results, predictions, test_tags


def main():
    parser = argparse.ArgumentParser(description='Train Vietnamese ABSA models')

    parser.add_argument('--model', type=str, required=True,
                        choices=['crf', 'logreg', 'svm', 'phobert', 'all'],
                        help='Model to train')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to data file (overrides config)')
    parser.add_argument('--data-config', type=str, default='configs/data_config.yaml',
                        help='Path to data configuration file')
    parser.add_argument('--model-config', type=str, default=None,
                        help='Path to model configuration file (optional)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (overrides config)')

    args = parser.parse_args()

    # Load data configuration
    print("Loading configuration...")
    data_config = load_data_config(args.data_config)

    # Override data path if specified
    if args.data:
        data_config['data']['path'] = args.data

    print(f"\nData Configuration:")
    print(f"  Data path: {data_config['data']['path']}")
    print(f"  Train size: {data_config['split']['train_size']}")
    print(f"  Test size: {data_config['split']['test_size']}")
    print(f"  Random seed: {data_config['split']['random_seed']}")

    # Load data
    data_path = data_config['data']['path']
    if not Path(data_path).exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    print(f"\nLoading data from: {data_path}")
    use_simple = data_config['preprocessing'].get('use_simple_tokenizer', False)
    full_dataset = ABSADataset(data_path, use_simple_tokenizer=use_simple)

    # Split data
    train_data, dev_data, test_data = split_data(full_dataset, data_config)

    print(f"\nData Split:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Dev:   {len(dev_data) if dev_data else 0} samples")
    print(f"  Test:  {len(test_data)} samples")

    # Setup output directory
    output_config = data_config.get('output', {})
    output_dir = Path(args.output_dir or output_config.get('model_dir', 'outputs/models'))
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(output_config.get('log_dir', 'outputs/logs'))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Train models
    models_to_train = ['crf', 'logreg', 'svm', 'phobert'] if args.model == 'all' else [args.model]
    all_results = {}

    for model_name in models_to_train:
        # Load model config
        if args.model_config:
            model_config = load_model_config(model_name, args.model_config)
        else:
            model_config = load_model_config(model_name)

        results, predictions, test_tags = train_model(
            model_name, train_data, dev_data, test_data,
            model_config, output_dir
        )

        all_results[model_name] = results

        # Print detailed report for last model
        if predictions is not None and len(predictions) > 0:
            print("\nDetailed Classification Report:")
            print_evaluation_report(test_tags, predictions)

    # Print summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")

    print(f"\nConfiguration:")
    print(f"  Train/Test Split: {data_config['split']['train_size']}/{data_config['split']['test_size']}")
    print(f"  Random Seed: {data_config['split']['random_seed']}")

    print(f"\n{'Model':<25} {'Precision':>12} {'Recall':>12} {'F1 Macro':>12}")
    print("-" * 61)

    for model_name, results in all_results.items():
        if 'error' in results:
            print(f"{model_name:<25} {'ERROR':>12}")
        else:
            p = results.get('test_precision', 0)
            r = results.get('test_recall', 0)
            f1 = results.get('test_f1_macro', 0)
            print(f"{model_name:<25} {p:>12.4f} {r:>12.4f} {f1:>12.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = log_dir / f'results_{timestamp}.json'

    # Include config in results
    save_data = {
        'config': {
            'data': data_config,
            'timestamp': timestamp
        },
        'results': all_results
    }

    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")
    print(f"Models saved to: {output_dir}")


if __name__ == '__main__':
    main()
