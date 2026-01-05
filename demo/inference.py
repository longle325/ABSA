"""
Inference Pipeline for Vietnamese ABSA Demo
===========================================
Handles the full inference pipeline:
1. Text preprocessing
2. Tokenization
3. Model prediction (BIO tags)
4. BIO tags → span conversion
5. Format output for UI

Supports both BiLSTM-CRF and BiLSTM-CRF-XLMR models.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.preprocessing import preprocess_text
from src.data.tokenizer import tokenize_with_offsets, get_tokens_only
from src.data.bio_converter import bio_tags_to_spans
from difflib import SequenceMatcher


def predict_with_model(
    raw_text: str,
    model: object,
    return_tokens: bool = False
) -> Tuple[str, List[List], Optional[List[str]]]:
    """
    Run full inference pipeline on raw text.

    Pipeline:
    1. Preprocess text (clean, normalize)
    2. Tokenize with offsets
    3. Model prediction → BIO tags
    4. Convert BIO tags → character-level spans
    5. Format as [start, end, "ASPECT#SENTIMENT"]

    Args:
        raw_text: Raw input text (Vietnamese)
        model: Loaded BiLSTMCRFModel or BiLSTMCRFXLMRModel instance
        return_tokens: If True, also return tokens for debugging

    Returns:
        Tuple of (cleaned_text, labels, tokens)
        - cleaned_text: Preprocessed text
        - labels: List of [start, end, "ASPECT#SENTIMENT"]
        - tokens: List of tokens (if return_tokens=True, else None)
    """
    # Step 1: Preprocess
    cleaned_text = preprocess_text(raw_text)

    if not cleaned_text.strip():
        return cleaned_text, [], None if not return_tokens else []

    # Step 2: Tokenize with character offsets
    tokens_with_offsets = tokenize_with_offsets(cleaned_text)
    tokens = get_tokens_only(tokens_with_offsets)

    if not tokens:
        return cleaned_text, [], None if not return_tokens else []

    # Step 3: Model prediction
    try:
        # Model expects List[List[str]] for batch processing
        # We pass a single sequence wrapped in a list
        bio_tags_batch = model.predict([tokens])
        bio_tags = bio_tags_batch[0]  # Get first (and only) sequence

    except Exception as e:
        print(f"Error during model prediction: {e}")
        return cleaned_text, [], tokens if return_tokens else None

    # Step 4: Convert BIO tags to spans
    spans = bio_tags_to_spans(bio_tags, tokens_with_offsets)

    # Step 5: Format spans as [start, end, "ASPECT#SENTIMENT"]
    labels = []
    for span in spans:
        labels.append([
            span['char_start'],
            span['char_end'],
            span['label']
        ])

    return cleaned_text, labels, tokens if return_tokens else None


def inference_with_fallback(
    raw_text: str,
    model: object,
    lookup_func: callable
) -> Tuple[str, List[List], str]:
    """
    Smart inference with dataset lookup fallback.

    Logic:
    1. Preprocess the raw text
    2. Check if preprocessed text exists in dataset
    3. If yes → return dataset labels (fast)
    4. If no → run model inference (accurate)

    Args:
        raw_text: Raw input text
        model: Loaded model instance
        lookup_func: Function to lookup text in dataset
                    Should return dict with 'labels' key or None

    Returns:
        Tuple of (cleaned_text, labels, source)
        - cleaned_text: Preprocessed text
        - labels: List of [start, end, "ASPECT#SENTIMENT"]
        - source: "dataset" or "model" to indicate where labels came from
    """
    # Step 1: Preprocess
    cleaned_text = preprocess_text(raw_text)

    if not cleaned_text.strip():
        return cleaned_text, [], "empty"

    # Step 2: Try dataset lookup first
    result = lookup_func(cleaned_text)

    if result and result.get('labels'):
        # Found in dataset - use those labels
        return cleaned_text, result['labels'], "dataset"

    # Step 3: Not in dataset - use model inference
    if model is None:
        return cleaned_text, [], "no_model"

    try:
        _, labels, _ = predict_with_model(raw_text, model, return_tokens=False)
        return cleaned_text, labels, "model"

    except Exception as e:
        print(f"Error during model inference: {e}")
        return cleaned_text, [], "error"


def map_labels_to_raw_text(
    raw_text: str,
    cleaned_text: str,
    labels_on_cleaned: List[List]
) -> List[List]:
    """
    Map labels from cleaned text positions back to raw text positions.

    Uses sequence alignment to find corresponding positions in raw text.

    Args:
        raw_text: Original raw text
        cleaned_text: Preprocessed/cleaned text
        labels_on_cleaned: Labels with positions on cleaned text
                          Format: [[start, end, "ASPECT#SENTIMENT"], ...]

    Returns:
        Labels with positions mapped to raw text
        Format: [[start, end, "ASPECT#SENTIMENT"], ...]
    """
    if not labels_on_cleaned:
        return []

    # Build mapping from cleaned positions to raw positions
    matcher = SequenceMatcher(None, raw_text, cleaned_text)

    # Create a position map: cleaned_idx -> raw_idx
    cleaned_to_raw = {}

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Direct character mapping
            for offset in range(i2 - i1):
                cleaned_to_raw[j1 + offset] = i1 + offset
        elif tag == 'replace':
            # Approximate mapping
            old_len = i2 - i1
            new_len = j2 - j1
            for offset in range(new_len):
                raw_offset = int(offset * old_len / new_len) if new_len > 0 else 0
                cleaned_to_raw[j1 + offset] = i1 + min(raw_offset, old_len - 1) if old_len > 0 else i1
        elif tag == 'insert':
            # Cleaned text has extra chars, map to same position in raw
            for offset in range(j2 - j1):
                cleaned_to_raw[j1 + offset] = i1
        # 'delete' doesn't affect cleaned positions

    # Map each label
    mapped_labels = []

    for label in labels_on_cleaned:
        start_cleaned, end_cleaned, aspect_sentiment = label

        # Map start and end positions
        start_raw = cleaned_to_raw.get(start_cleaned)
        end_raw = cleaned_to_raw.get(end_cleaned - 1)  # end is exclusive

        if start_raw is not None and end_raw is not None:
            # Adjust end to be exclusive
            end_raw = end_raw + 1

            # Validate
            if start_raw < end_raw and start_raw >= 0 and end_raw <= len(raw_text):
                mapped_labels.append([start_raw, end_raw, aspect_sentiment])

    return mapped_labels


def format_prediction_info(source: str) -> str:
    """
    Format a user-friendly message about prediction source.

    Args:
        source: One of "dataset", "model", "empty", "no_model", "error"

    Returns:
        Formatted message string
    """
    messages = {
        "dataset": "Found in training dataset",
        "model": "Predicted by model",
        "empty": "Empty text",
        "no_model": "No model loaded",
        "error": "Error during prediction"
    }

    return messages.get(source, "Unknown source")


if __name__ == "__main__":
    # Test the inference pipeline
    print("Testing Inference Pipeline")
    print("=" * 60)

    # Test preprocessing
    test_text = "Máy này quááááá tệ!!! hok dc sài @@@"
    cleaned = preprocess_text(test_text)
    print(f"\nPreprocessing test:")
    print(f"  Raw: {test_text}")
    print(f"  Cleaned: {cleaned}")

    # Test tokenization
    tokens_with_offsets = tokenize_with_offsets(cleaned)
    print(f"\nTokenization test:")
    for token, start, end in tokens_with_offsets:
        print(f"  '{token}' [{start}:{end}] -> '{cleaned[start:end]}'")

    print("\nNote: To test full inference, load a model using model_manager.py")
