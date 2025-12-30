#!/usr/bin/env python
"""
Evaluation Script for Vietnamese ABSA Models

Load trained model checkpoints and evaluate on test data.

Usage:
    python experiments/eval.py --model bilstm_crf --checkpoint outputs/models/bilstm_crf_model.pkl
    python experiments/eval.py --model bilstm_crf_xlmr --checkpoint outputs/models/bilstm_crf_xlmr_model.pkl
    python experiments/eval.py --model crf --checkpoint outputs/models/crf_model.pkl
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

from src.utils.config import load_data_config
from src.data.dataset import ABSADataset
from src.evaluation.metrics import evaluate_sequence_labeling, print_evaluation_report


def split_data(dataset: ABSADataset, config: dict):
    """
    Split dataset according to config settings.
    """
    split_config = config['split']

    train_size = split_config['train_size']
    random_seed = split_config['random_seed']
    shuffle = split_config.get('shuffle', True)

    samples = dataset.samples.copy()

    if shuffle:
        random.seed(random_seed)
        random.shuffle(samples)

    total = len(samples)
    train_end = int(total * train_size)

    train_samples = samples[:train_end]
    test_samples = samples[train_end:]

    train_data = ABSADataset()
    train_data.samples = train_samples
    train_data.label2id = dataset.label2id
    train_data.id2label = dataset.id2label

    test_data = ABSADataset()
    test_data.samples = test_samples
    test_data.label2id = dataset.label2id
    test_data.id2label = dataset.id2label

    return train_data, test_data


def load_model(model_name: str, checkpoint_path: str):
    """
    Load a trained model from checkpoint.
    """
    print(f"Loading {model_name} from {checkpoint_path}...")

    if model_name == 'crf':
        from src.models.crf_model import CRFModel
        model = CRFModel()
        model.load(checkpoint_path)

    elif model_name == 'logreg':
        from src.models.logreg_model import LogRegModel
        model = LogRegModel()
        model.load(checkpoint_path)

    elif model_name == 'svm':
        from src.models.svm_model import SVMModel
        model = SVMModel()
        model.load(checkpoint_path)

    elif model_name == 'bilstm_crf':
        from src.models.bilstm_crf import BiLSTMCRFModel
        model = BiLSTMCRFModel()
        model.load(checkpoint_path)

    elif model_name == 'bilstm_crf_xlmr':
        from src.models.bilstm_crf_xlmr import BiLSTMCRFXLMRModel
        model = BiLSTMCRFXLMRModel()
        model.load(checkpoint_path)

    elif model_name == 'phobert':
        from src.models.phobert_bilstm_crf import PhoBERTModel
        model = PhoBERTModel()
        model.load(checkpoint_path)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def evaluate_model(model, test_tokens, test_tags, model_name: str):
    """
    Evaluate model on test data.
    """
    print(f"\nEvaluating {model_name}...")
    print(f"Test samples: {len(test_tokens)}")

    # Predict
    predictions = model.predict(test_tokens)

    # Evaluate
    metrics = evaluate_sequence_labeling(test_tags, predictions)

    print(f"\nTest Results for {model_name}:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Macro:  {metrics['f1']:.4f}")

    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description='Evaluate Vietnamese ABSA models')

    parser.add_argument('--model', type=str, required=True,
                        choices=['crf', 'logreg', 'svm', 'phobert', 'bilstm_crf', 'bilstm_crf_xlmr'],
                        help='Model type to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint file')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to data file (overrides config)')
    parser.add_argument('--data-config', type=str, default='configs/data_config.yaml',
                        help='Path to data configuration file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for predictions (optional)')
    parser.add_argument('--detailed', action='store_true',
                        help='Print detailed classification report')

    args = parser.parse_args()

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

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

    # Split data (use same split as training)
    train_data, test_data = split_data(full_dataset, data_config)

    print(f"\nData Split:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Test:  {len(test_data)} samples")

    # Get test tokens and tags
    test_tokens = test_data.get_all_tokens()
    test_tags = test_data.get_all_tags()

    # Load model
    model = load_model(args.model, args.checkpoint)

    # Evaluate
    metrics, predictions = evaluate_model(model, test_tokens, test_tags, args.model)

    # Print detailed report if requested
    if args.detailed:
        print("\n" + "=" * 60)
        print("Detailed Classification Report:")
        print("=" * 60)
        print_evaluation_report(test_tags, predictions)

    # Save predictions if output specified
    if args.output:
        output_data = {
            'model': args.model,
            'checkpoint': args.checkpoint,
            'metrics': metrics,
            'predictions': predictions,
            'ground_truth': test_tags,
            'timestamp': datetime.now().isoformat()
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nPredictions saved to: {args.output}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test samples: {len(test_tokens)}")
    print(f"\nMetrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Macro:  {metrics['f1']:.4f}")


if __name__ == '__main__':
    main()
