"""
Vietnamese Tokenizer with Character Offset Tracking

This module provides tokenization for Vietnamese text using underthesea,
with accurate character offset tracking for span-based ABSA.
"""

from typing import List, Tuple
import re


def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokenize Vietnamese text and track character offsets.

    Uses underthesea for word segmentation. Handles multi-syllable words
    (e.g., "thế_giới") by finding the original text span.

    Args:
        text: Vietnamese text to tokenize

    Returns:
        List of (token, start_offset, end_offset) tuples.
        Offsets are character-level, following Python slicing convention (end is exclusive).
    """
    try:
        from underthesea import word_tokenize
    except ImportError:
        raise ImportError(
            "underthesea is required for Vietnamese tokenization. "
            "Install with: pip install underthesea"
        )

    # Tokenize using underthesea
    tokens = word_tokenize(text)

    tokens_with_offsets = []
    current_pos = 0

    for token in tokens:
        # underthesea uses underscore for multi-syllable words
        # e.g., "thế_giới" represents "thế giới" in original text
        search_token = token.replace('_', ' ')

        # Find the token in the original text starting from current position
        start = text.find(search_token, current_pos)

        if start == -1:
            # Fallback: try to find without space replacement
            start = text.find(token, current_pos)

        if start == -1:
            # If still not found, use current position (shouldn't happen often)
            start = current_pos
            end = start + len(search_token)
        else:
            end = start + len(search_token)

        tokens_with_offsets.append((token, start, end))
        current_pos = end

    return tokens_with_offsets


def simple_tokenize(text: str) -> List[str]:
    """
    Simple whitespace-based tokenization (fallback).

    Args:
        text: Text to tokenize

    Returns:
        List of tokens
    """
    return text.split()


def simple_tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """
    Simple whitespace-based tokenization with offset tracking.

    Useful as a fallback when underthesea is not available or
    for testing purposes.

    Args:
        text: Text to tokenize

    Returns:
        List of (token, start_offset, end_offset) tuples
    """
    tokens_with_offsets = []

    # Find all non-whitespace sequences
    for match in re.finditer(r'\S+', text):
        token = match.group()
        start = match.start()
        end = match.end()
        tokens_with_offsets.append((token, start, end))

    return tokens_with_offsets


def get_tokens_only(tokens_with_offsets: List[Tuple[str, int, int]]) -> List[str]:
    """
    Extract only the tokens from tokens_with_offsets list.

    Args:
        tokens_with_offsets: List of (token, start, end) tuples

    Returns:
        List of tokens only
    """
    return [token for token, _, _ in tokens_with_offsets]


def validate_offsets(text: str, tokens_with_offsets: List[Tuple[str, int, int]]) -> bool:
    """
    Validate that token offsets correctly map back to original text.

    Args:
        text: Original text
        tokens_with_offsets: List of (token, start, end) tuples

    Returns:
        True if all offsets are valid, False otherwise
    """
    for token, start, end in tokens_with_offsets:
        # Handle underscore in multi-syllable tokens
        expected_text = token.replace('_', ' ')
        actual_text = text[start:end]

        if actual_text != expected_text:
            return False

    return True


if __name__ == "__main__":
    # Test the tokenizer
    test_texts = [
        "Pin cực trâu, cam ổn, vân tay khoá các kiểu cực nhạy.",
        "Máy đẹp pin trâu, còn thiếu những tiện ích thông dụng.",
        "Nhân viên thế giới di động nhiệt tình và vui vẻ.",
    ]

    print("Testing Vietnamese Tokenizer")
    print("=" * 50)

    for text in test_texts:
        print(f"\nText: {text}")
        tokens_with_offsets = tokenize_with_offsets(text)

        print("Tokens with offsets:")
        for token, start, end in tokens_with_offsets:
            original = text[start:end]
            print(f"  '{token}' [{start}:{end}] -> '{original}'")

        # Validate
        is_valid = validate_offsets(text, tokens_with_offsets)
        print(f"Offsets valid: {is_valid}")
