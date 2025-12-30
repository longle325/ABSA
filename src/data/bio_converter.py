"""
BIO Tag Converter for Vietnamese Aspect-Based Sentiment Analysis

Converts character-level span annotations to token-level BIO tags
for sequence labeling models (CRF, BiLSTM-CRF, etc.).

BIO Schema:
- B-ASPECT#SENTIMENT: Beginning of aspect span
- I-ASPECT#SENTIMENT: Inside aspect span (continuation)
- O: Outside any aspect span
"""

from typing import List, Tuple, Dict, Callable, Optional
from .tokenizer import tokenize_with_offsets, simple_tokenize_with_offsets


# All possible aspect categories
ASPECTS = [
    'BATTERY', 'CAMERA', 'DESIGN', 'FEATURES', 'GENERAL',
    'PERFORMANCE', 'PRICE', 'SCREEN', 'SER&ACC', 'STORAGE'
]

# All possible sentiments
SENTIMENTS = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']


def get_all_labels() -> List[str]:
    """
    Generate all possible BIO labels.

    Returns:
        List of all possible labels: O + B-ASPECT#SENTIMENT + I-ASPECT#SENTIMENT
        Total: 1 + 10*3*2 = 61 labels
    """
    labels = ['O']

    for aspect in ASPECTS:
        for sentiment in SENTIMENTS:
            label = f"{aspect}#{sentiment}"
            labels.append(f"B-{label}")
            labels.append(f"I-{label}")

    return sorted(labels)


def build_label_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build label-to-id and id-to-label mappings.

    Returns:
        Tuple of (label2id, id2label) dictionaries
    """
    labels = get_all_labels()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def char_span_to_bio_tags(
    text: str,
    labels: List[Tuple[int, int, str]],
    tokenizer_func: Optional[Callable] = None
) -> Tuple[List[str], List[Tuple[str, int, int]]]:
    """
    Convert character-level span annotations to token-level BIO tags.

    Args:
        text: Original Vietnamese text
        labels: List of (start_char, end_char, "ASPECT#SENTIMENT") tuples
        tokenizer_func: Optional tokenizer function. Defaults to tokenize_with_offsets.

    Returns:
        Tuple of (bio_tags, tokens_with_offsets)
        - bio_tags: List of BIO tags aligned with tokens
        - tokens_with_offsets: List of (token, start, end) tuples
    """
    if tokenizer_func is None:
        tokenizer_func = tokenize_with_offsets

    # Tokenize text with character offsets
    tokens_with_offsets = tokenizer_func(text)
    num_tokens = len(tokens_with_offsets)

    # Initialize all tags as 'O' (outside)
    bio_tags = ['O'] * num_tokens

    # Sort labels by start position (handle potential overlaps)
    sorted_labels = sorted(labels, key=lambda x: (x[0], -x[1]))

    for label_start, label_end, label_str in sorted_labels:
        # Track if we've assigned a B- tag for this span
        span_started = False

        for i, (token, tok_start, tok_end) in enumerate(tokens_with_offsets):
            # Check if token overlaps with label span
            # Token is inside span if: tok_start >= label_start AND tok_end <= label_end
            # Or partial overlap: max(tok_start, label_start) < min(tok_end, label_end)

            overlap_start = max(tok_start, label_start)
            overlap_end = min(tok_end, label_end)
            has_overlap = overlap_start < overlap_end

            if has_overlap:
                # Calculate overlap ratio
                token_len = tok_end - tok_start
                overlap_len = overlap_end - overlap_start
                overlap_ratio = overlap_len / token_len if token_len > 0 else 0

                # Only assign tag if overlap is significant (>50%)
                if overlap_ratio > 0.5 and bio_tags[i] == 'O':
                    if not span_started:
                        bio_tags[i] = f'B-{label_str}'
                        span_started = True
                    else:
                        bio_tags[i] = f'I-{label_str}'

    return bio_tags, tokens_with_offsets


def bio_tags_to_spans(
    bio_tags: List[str],
    tokens_with_offsets: Optional[List[Tuple[str, int, int]]] = None
) -> List[Dict]:
    """
    Convert BIO tags back to span annotations.

    Args:
        bio_tags: List of BIO tags
        tokens_with_offsets: Optional list of (token, start, end) tuples
                            If provided, spans will include character offsets.

    Returns:
        List of span dictionaries with keys:
        - token_start: Start token index
        - token_end: End token index (exclusive)
        - label: ASPECT#SENTIMENT label
        - char_start: Character start offset (if tokens_with_offsets provided)
        - char_end: Character end offset (if tokens_with_offsets provided)
        - text: Span text (if tokens_with_offsets provided)
    """
    spans = []
    current_span = None

    for i, tag in enumerate(bio_tags):
        if tag.startswith('B-'):
            # Start new span
            if current_span is not None:
                spans.append(current_span)

            label = tag[2:]  # Remove 'B-' prefix
            current_span = {
                'token_start': i,
                'token_end': i + 1,
                'label': label
            }

        elif tag.startswith('I-'):
            label = tag[2:]  # Remove 'I-' prefix

            if current_span is not None and current_span['label'] == label:
                # Continue current span
                current_span['token_end'] = i + 1
            else:
                # Invalid I- tag (no matching B-), treat as B-
                if current_span is not None:
                    spans.append(current_span)
                current_span = {
                    'token_start': i,
                    'token_end': i + 1,
                    'label': label
                }

        else:  # 'O' tag
            if current_span is not None:
                spans.append(current_span)
                current_span = None

    # Don't forget the last span
    if current_span is not None:
        spans.append(current_span)

    # Add character offsets and text if tokens_with_offsets is provided
    if tokens_with_offsets is not None:
        for span in spans:
            token_start_idx = span['token_start']
            token_end_idx = span['token_end'] - 1  # Last token index (inclusive)

            _, char_start, _ = tokens_with_offsets[token_start_idx]
            _, _, char_end = tokens_with_offsets[token_end_idx]

            span['char_start'] = char_start
            span['char_end'] = char_end

            # Get span text
            span_tokens = [tokens_with_offsets[j][0]
                          for j in range(span['token_start'], span['token_end'])]
            span['text'] = ' '.join(span_tokens).replace('_', ' ')

    return spans


def parse_label(label_str: str) -> Tuple[str, str]:
    """
    Parse ASPECT#SENTIMENT label into components.

    Args:
        label_str: Label in format "ASPECT#SENTIMENT"

    Returns:
        Tuple of (aspect, sentiment)
    """
    parts = label_str.split('#')
    if len(parts) == 2:
        return parts[0], parts[1]
    else:
        raise ValueError(f"Invalid label format: {label_str}")


def validate_bio_sequence(bio_tags: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate BIO tag sequence for correctness.

    Rules:
    - I- tag must follow B- or I- with same label
    - B- can appear anywhere
    - O can appear anywhere

    Args:
        bio_tags: List of BIO tags

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    prev_tag = 'O'

    for i, tag in enumerate(bio_tags):
        if tag.startswith('I-'):
            label = tag[2:]

            # Check if previous tag is B- or I- with same label
            if prev_tag == 'O':
                errors.append(f"Position {i}: I-{label} follows O (should follow B-{label} or I-{label})")
            elif prev_tag.startswith('B-') and prev_tag[2:] != label:
                errors.append(f"Position {i}: I-{label} follows {prev_tag} (label mismatch)")
            elif prev_tag.startswith('I-') and prev_tag[2:] != label:
                errors.append(f"Position {i}: I-{label} follows {prev_tag} (label mismatch)")

        prev_tag = tag

    return len(errors) == 0, errors


if __name__ == "__main__":
    # Test the BIO converter
    print("Testing BIO Tag Converter")
    print("=" * 60)

    # Test case from dataset
    text = "Pin cực trâu, cam ổn, vân tay khoá các kiểu cực nhạy."
    labels = [
        (0, 12, "BATTERY#POSITIVE"),  # "Pin cực trâu"
        (14, 20, "CAMERA#POSITIVE"),   # "cam ổn"
    ]

    print(f"\nText: {text}")
    print(f"Labels: {labels}")

    bio_tags, tokens = char_span_to_bio_tags(text, labels)

    print("\nTokens with BIO tags:")
    for (token, start, end), tag in zip(tokens, bio_tags):
        print(f"  {token:15} [{start:2}:{end:2}] -> {tag}")

    # Validate
    is_valid, errors = validate_bio_sequence(bio_tags)
    print(f"\nSequence valid: {is_valid}")
    if errors:
        for err in errors:
            print(f"  Error: {err}")

    # Convert back to spans
    spans = bio_tags_to_spans(bio_tags, tokens)
    print("\nExtracted spans:")
    for span in spans:
        print(f"  {span}")

    # Print label vocabulary
    print("\n" + "=" * 60)
    print("Label vocabulary:")
    labels = get_all_labels()
    print(f"Total labels: {len(labels)}")
    print(f"Labels: {labels[:10]}... (showing first 10)")
