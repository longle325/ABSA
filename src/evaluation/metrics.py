"""
Evaluation Metrics for Vietnamese ABSA Sequence Labeling

Provides F1 macro, precision, recall metrics using seqeval library.
Includes per-aspect and per-sentiment evaluation.
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict


def evaluate_sequence_labeling(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Evaluate sequence labeling using seqeval.

    Args:
        y_true: List of ground truth BIO tag sequences
        y_pred: List of predicted BIO tag sequences
        average: Averaging method ('micro', 'macro', 'weighted')

    Returns:
        Dictionary with precision, recall, f1 metrics
    """
    try:
        from seqeval.metrics import (
            f1_score,
            precision_score,
            recall_score,
            classification_report
        )

        return {
            'precision': precision_score(y_true, y_pred, average=average),
            'recall': recall_score(y_true, y_pred, average=average),
            'f1': f1_score(y_true, y_pred, average=average),
            'report': classification_report(y_true, y_pred, digits=4)
        }

    except ImportError:
        print("Warning: seqeval not installed. Using fallback metrics.")
        return compute_metrics_fallback(y_true, y_pred)


def compute_metrics_fallback(
    y_true: List[List[str]],
    y_pred: List[List[str]]
) -> Dict[str, float]:
    """
    Fallback metrics computation when seqeval is not available.

    Uses token-level evaluation (less accurate than span-level).

    Args:
        y_true: Ground truth sequences
        y_pred: Predicted sequences

    Returns:
        Metrics dictionary
    """
    # Flatten sequences
    all_true = []
    all_pred = []

    for true_seq, pred_seq in zip(y_true, y_pred):
        all_true.extend(true_seq)
        all_pred.extend(pred_seq)

    # Compute per-label metrics
    labels = set(all_true) | set(all_pred)
    labels.discard('O')

    total_tp = 0
    total_fp = 0
    total_fn = 0
    label_f1s = []

    for label in labels:
        tp = sum(1 for t, p in zip(all_true, all_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(all_true, all_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(all_true, all_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        label_f1s.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Macro F1
    macro_f1 = sum(label_f1s) / len(label_f1s) if label_f1s else 0

    # Micro metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) \
        if (micro_precision + micro_recall) > 0 else 0

    return {
        'precision': micro_precision,
        'recall': micro_recall,
        'f1': macro_f1,
        'f1_micro': micro_f1,
        'report': f"Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}"
    }


def evaluate_by_aspect(
    y_true: List[List[str]],
    y_pred: List[List[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate performance per aspect category.

    Args:
        y_true: Ground truth sequences
        y_pred: Predicted sequences

    Returns:
        Dictionary mapping aspect to metrics
    """
    aspect_metrics = {}

    # Extract true and predicted spans
    true_spans_by_aspect = defaultdict(list)
    pred_spans_by_aspect = defaultdict(list)

    for seq_idx, (true_seq, pred_seq) in enumerate(zip(y_true, y_pred)):
        true_spans = _extract_spans(true_seq)
        pred_spans = _extract_spans(pred_seq)

        for span in true_spans:
            aspect = span['label'].split('#')[0] if '#' in span['label'] else span['label']
            true_spans_by_aspect[aspect].append((seq_idx, span['start'], span['end'], span['label']))

        for span in pred_spans:
            aspect = span['label'].split('#')[0] if '#' in span['label'] else span['label']
            pred_spans_by_aspect[aspect].append((seq_idx, span['start'], span['end'], span['label']))

    # Compute metrics per aspect
    all_aspects = set(true_spans_by_aspect.keys()) | set(pred_spans_by_aspect.keys())

    for aspect in all_aspects:
        true_set = set(true_spans_by_aspect[aspect])
        pred_set = set(pred_spans_by_aspect[aspect])

        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        aspect_metrics[aspect] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': len(true_set)
        }

    return aspect_metrics


def evaluate_by_sentiment(
    y_true: List[List[str]],
    y_pred: List[List[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate performance per sentiment category.

    Args:
        y_true: Ground truth sequences
        y_pred: Predicted sequences

    Returns:
        Dictionary mapping sentiment to metrics
    """
    sentiment_metrics = {}

    # Extract true and predicted spans
    true_spans_by_sent = defaultdict(list)
    pred_spans_by_sent = defaultdict(list)

    for seq_idx, (true_seq, pred_seq) in enumerate(zip(y_true, y_pred)):
        true_spans = _extract_spans(true_seq)
        pred_spans = _extract_spans(pred_seq)

        for span in true_spans:
            if '#' in span['label']:
                sentiment = span['label'].split('#')[1]
                true_spans_by_sent[sentiment].append(
                    (seq_idx, span['start'], span['end'], span['label'])
                )

        for span in pred_spans:
            if '#' in span['label']:
                sentiment = span['label'].split('#')[1]
                pred_spans_by_sent[sentiment].append(
                    (seq_idx, span['start'], span['end'], span['label'])
                )

    # Compute metrics per sentiment
    all_sentiments = set(true_spans_by_sent.keys()) | set(pred_spans_by_sent.keys())

    for sentiment in all_sentiments:
        true_set = set(true_spans_by_sent[sentiment])
        pred_set = set(pred_spans_by_sent[sentiment])

        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        sentiment_metrics[sentiment] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': len(true_set)
        }

    return sentiment_metrics


def _extract_spans(bio_tags: List[str]) -> List[Dict]:
    """
    Extract spans from BIO tag sequence.

    Args:
        bio_tags: BIO tag sequence

    Returns:
        List of span dictionaries with start, end, label
    """
    spans = []
    current_span = None

    for i, tag in enumerate(bio_tags):
        if tag.startswith('B-'):
            if current_span is not None:
                spans.append(current_span)
            current_span = {
                'start': i,
                'end': i + 1,
                'label': tag[2:]
            }
        elif tag.startswith('I-'):
            label = tag[2:]
            if current_span is not None and current_span['label'] == label:
                current_span['end'] = i + 1
            else:
                if current_span is not None:
                    spans.append(current_span)
                current_span = {
                    'start': i,
                    'end': i + 1,
                    'label': label
                }
        else:  # O tag
            if current_span is not None:
                spans.append(current_span)
                current_span = None

    if current_span is not None:
        spans.append(current_span)

    return spans


def print_evaluation_report(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    show_per_aspect: bool = True,
    show_per_sentiment: bool = True
) -> None:
    """
    Print comprehensive evaluation report.

    Args:
        y_true: Ground truth sequences
        y_pred: Predicted sequences
        show_per_aspect: Whether to show per-aspect metrics
        show_per_sentiment: Whether to show per-sentiment metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    # Overall metrics
    metrics = evaluate_sequence_labeling(y_true, y_pred)
    print(f"\nOverall Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 (macro): {metrics['f1']:.4f}")

    # Detailed report
    if 'report' in metrics and isinstance(metrics['report'], str):
        print(f"\nDetailed Report:")
        print(metrics['report'])

    # Per-aspect metrics
    if show_per_aspect:
        aspect_metrics = evaluate_by_aspect(y_true, y_pred)
        print(f"\nPer-Aspect Metrics:")
        print(f"{'Aspect':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 55)
        for aspect, m in sorted(aspect_metrics.items()):
            print(f"{aspect:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                  f"{m['f1']:>10.4f} {m['support']:>10}")

    # Per-sentiment metrics
    if show_per_sentiment:
        sentiment_metrics = evaluate_by_sentiment(y_true, y_pred)
        print(f"\nPer-Sentiment Metrics:")
        print(f"{'Sentiment':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 55)
        for sentiment, m in sorted(sentiment_metrics.items()):
            print(f"{sentiment:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                  f"{m['f1']:>10.4f} {m['support']:>10}")


if __name__ == "__main__":
    # Test evaluation metrics
    print("Testing Evaluation Metrics")
    print("=" * 60)

    # Sample data
    y_true = [
        ['B-BATTERY#POSITIVE', 'I-BATTERY#POSITIVE', 'O', 'B-CAMERA#POSITIVE'],
        ['O', 'B-PERFORMANCE#NEGATIVE', 'I-PERFORMANCE#NEGATIVE', 'O'],
    ]

    y_pred = [
        ['B-BATTERY#POSITIVE', 'I-BATTERY#POSITIVE', 'O', 'B-CAMERA#POSITIVE'],
        ['O', 'B-PERFORMANCE#NEGATIVE', 'O', 'O'],
    ]

    print("\nGround truth:")
    for seq in y_true:
        print(f"  {seq}")

    print("\nPredictions:")
    for seq in y_pred:
        print(f"  {seq}")

    # Evaluate
    print_evaluation_report(y_true, y_pred)
