"""
Preprocessing Script for UIT-ViSD4SA Dataset
=============================================
This script merges train/dev/test JSONL files, cleans text, and validates labels.

Features:
- Remove emoji
- Remove URLs (https://...)
- Fix stuck words (e.g., "đẹpNói" -> "đẹp. Nói")
- Normalize repeated punctuation (... -> ., !!! -> !)
- Remove special characters (@#$%^&* etc.)
- Normalize repeated characters (haizzzzz -> haiz, quááááá -> quá)
- Normalize teencode to standard Vietnamese (hok -> không, dc -> được)
- Validate and recalculate label positions
- Separate samples without annotations

Output:
- cleaned_data/cleaned_data.jsonl
- cleaned_data/no_annotation_samples.jsonl
- cleaned_data/preprocessing_report.json
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Dict

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "UIT-ViSD4SA" / "data"
OUTPUT_DIR = SCRIPT_DIR.parent / "cleaned_data"

# Splits to process
SPLITS = ["train", "dev", "test"]

# ============================================================================
# TEENCODE DICTIONARY
# ============================================================================

TEENCODE_MAP = {
    # Common abbreviations
    r'\bk\b': 'không',
    r'\bko\b': 'không',
    r'\bkg\b': 'không',
    r'\bhok\b': 'không',
    r'\bkh\b': 'không',
    r'\bkhong\b': 'không',
    r'\bdc\b': 'được',
    r'\bđc\b': 'được',
    r'\bduoc\b': 'được',
    r'\bđt\b': 'điện thoại',
    r'\bnv\b': 'nhân viên',
    r'\btgdd\b': 'thế giới di động',
    r'\btgdđ\b': 'thế giới di động',
    r'\bdmx\b': 'điện máy xanh',
    r'\bsp\b': 'sản phẩm',
    r'\bmk\b': 'mình',
    r'\bbn\b': 'bạn',
    r'\bj\b': 'gì',
    r'\bz\b': 'gì',
    r'\blq\b': 'liên quân',
    r'\blqmb\b': 'liên quân mobile',
    r'\bpubg\b': 'PUBG',
    r'\bfb\b': 'Facebook',
    r'\byt\b': 'YouTube',

    # Common misspellings
    r'\bsài\b': 'xài',
    r'\bbin\b': 'pin',
    r'\bbìn\b': 'pin',
    r'\bcam\b': 'camera',
    r'\bcamera\b': 'camera',
    r'\bwifi\b': 'WiFi',
    r'\bwf\b': 'WiFi',

    # Other abbreviations
    r'\bvs\b': 'với',
    r'\bvới\b': 'với',
    r'\bok\b': 'OK',
    r'\boki\b': 'OK',
    r'\bokie\b': 'OK',
    r'\bbt\b': 'bình thường',
    r'\bae\b': 'anh em',
    r'\bss\b': 'Samsung',
    r'\bip\b': 'iPhone',
}

# ============================================================================
# TEXT CLEANING FUNCTIONS
# ============================================================================

def remove_emoji(text: str) -> str:
    """Remove all emoji characters from text."""
    # Emoji pattern covering most emoji ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002600-\U000026FF"  # Misc symbols
        "\U00002700-\U000027BF"  # Dingbats
        "\U0000FE00-\U0000FE0F"  # Variation Selectors
        "\U0001F000-\U0001F02F"  # Mahjong Tiles
        "\U0001F0A0-\U0001F0FF"  # Playing Cards
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)


def remove_urls(text: str) -> str:
    """Remove URLs starting with http:// or https://"""
    # Pattern to match URLs
    url_pattern = re.compile(
        r'https?://[^\s]+',
        flags=re.IGNORECASE
    )
    return url_pattern.sub('', text)


def fix_stuck_words(text: str) -> str:
    """
    Add space after punctuation when followed by uppercase letter.
    Example: "đẹpNói chung" -> "đẹp. Nói chung"
    """
    # Vietnamese lowercase letters including diacritics
    vn_lower = r'a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ'
    # Vietnamese uppercase letters including diacritics
    vn_upper = r'A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ'

    # Pattern: lowercase + punctuation + uppercase (no space between)
    # Add space after punctuation
    patterns = [
        # After period
        (rf'([{vn_lower}])\.([{vn_upper}])', r'\1. \2'),
        # After comma
        (rf'([{vn_lower}]),([{vn_upper}])', r'\1, \2'),
        # After colon
        (rf'([{vn_lower}]):([{vn_upper}])', r'\1: \2'),
        # After semicolon
        (rf'([{vn_lower}]);([{vn_upper}])', r'\1; \2'),
        # After exclamation
        (rf'([{vn_lower}])!([{vn_upper}])', r'\1! \2'),
        # After question mark
        (rf'([{vn_lower}])\?([{vn_upper}])', r'\1? \2'),
        # Direct lowercase to uppercase transition (no punctuation)
        (rf'([{vn_lower}])([{vn_upper}])', r'\1 \2'),
    ]

    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)

    return result


def normalize_punctuation(text: str) -> str:
    """Normalize repeated punctuation marks."""
    # Multiple periods -> single period
    text = re.sub(r'\.{2,}', '.', text)
    # Multiple exclamation marks -> single
    text = re.sub(r'!{2,}', '!', text)
    # Multiple question marks -> single
    text = re.sub(r'\?{2,}', '?', text)
    # Multiple commas -> single
    text = re.sub(r',{2,}', ',', text)

    return text


def normalize_repeated_chars(text: str) -> str:
    """
    Normalize repeated characters (more than 2 consecutive same chars).
    Example: "quááááá" -> "quá", "haizzzzz" -> "haiz", "okkkkk" -> "ok"
    """
    # Pattern: any character repeated more than 2 times -> keep only 1
    # This handles both Vietnamese and ASCII characters
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text


def remove_special_chars(text: str) -> str:
    """
    Remove special characters, keeping only:
    - Vietnamese letters (with diacritics)
    - ASCII letters and digits
    - Basic punctuation: . , : ; ! ? - ' " / ( )
    - Whitespace
    """
    # Pattern to keep: Vietnamese chars, ASCII alphanumeric, basic punctuation, whitespace
    # Vietnamese character ranges
    allowed_pattern = re.compile(
        r'[^'
        r'a-zA-Z0-9'  # ASCII alphanumeric
        r'àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ'  # Vietnamese lowercase
        r'ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ'  # Vietnamese uppercase
        r'\s'  # Whitespace
        r'\.,:;!?\-\'"/()'  # Basic punctuation
        r']'
    )
    text = allowed_pattern.sub('', text)
    return text


def normalize_whitespace(text: str) -> str:
    """Remove extra whitespace and trim."""
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Trim leading/trailing whitespace
    text = text.strip()
    return text


def normalize_teencode(text: str) -> str:
    """Convert teencode/abbreviations to standard Vietnamese."""
    result = text
    for pattern, replacement in TEENCODE_MAP.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def clean_text(text: str) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    """
    Main text cleaning pipeline.
    Returns cleaned text and character position mapping.

    The mapping is a list of (old_start, old_end, new_start, new_end) tuples
    that track how each character moved.
    """
    # Build character mapping as we clean
    # This is a simplified approach - we track the cumulative offset

    # Step 1: Remove URLs first (they can be long)
    text = remove_urls(text)

    # Step 2: Remove emoji
    text = remove_emoji(text)

    # Step 3: Fix stuck words (add spaces)
    text = fix_stuck_words(text)

    # Step 4: Normalize punctuation (... -> ., !!! -> !)
    text = normalize_punctuation(text)

    # Step 5: Remove special characters (@#$%^&* etc.)
    text = remove_special_chars(text)

    # Step 6: Normalize repeated characters (haizzzzz -> haiz)
    text = normalize_repeated_chars(text)

    # Step 7: Normalize teencode (hok -> không, dc -> được, etc.)
    text = normalize_teencode(text)

    # Step 8: Normalize whitespace
    text = normalize_whitespace(text)

    return text, []  # Mapping will be computed differently


def build_char_mapping(original: str, cleaned: str) -> Dict[int, int]:
    """
    Build a mapping from original character positions to cleaned positions.
    Uses sequence alignment to find the best mapping.
    """
    # Simple approach: use difflib to align sequences
    from difflib import SequenceMatcher

    mapping = {}
    matcher = SequenceMatcher(None, original, cleaned)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Characters are the same, direct mapping
            for offset in range(i2 - i1):
                mapping[i1 + offset] = j1 + offset
        elif tag == 'replace':
            # Characters replaced, approximate mapping
            old_len = i2 - i1
            new_len = j2 - j1
            for offset in range(old_len):
                # Map to proportional position in new text
                new_offset = int(offset * new_len / old_len) if old_len > 0 else 0
                mapping[i1 + offset] = j1 + min(new_offset, new_len - 1) if new_len > 0 else j1
        elif tag == 'delete':
            # Characters deleted, map to the position before deletion
            for offset in range(i2 - i1):
                mapping[i1 + offset] = j1  # Map to where deletion happened
        # 'insert' doesn't affect original positions

    return mapping


def adjust_labels(
    original_text: str,
    cleaned_text: str,
    labels: List[List]
) -> Tuple[List[List], Dict[str, int]]:
    """
    Adjust label positions after text cleaning.

    Returns:
        - Adjusted labels
        - Statistics about label adjustments
    """
    stats = {
        'total': len(labels),
        'valid': 0,
        'adjusted': 0,
        'removed': 0
    }

    if not labels:
        return [], stats

    # Build character mapping
    char_map = build_char_mapping(original_text, cleaned_text)

    adjusted_labels = []

    for label in labels:
        start, end, aspect_sentiment = label

        # Validate original positions
        if start < 0 or end > len(original_text) or start >= end:
            stats['removed'] += 1
            continue

        # Map to new positions
        new_start = char_map.get(start)
        new_end = char_map.get(end - 1)  # end is exclusive

        if new_start is None or new_end is None:
            stats['removed'] += 1
            continue

        new_end = new_end + 1  # Make end exclusive again

        # Validate new positions
        if new_start >= new_end or new_start < 0 or new_end > len(cleaned_text):
            stats['removed'] += 1
            continue

        # Check if positions changed
        if new_start != start or new_end != end:
            stats['adjusted'] += 1
        else:
            stats['valid'] += 1

        adjusted_labels.append([new_start, new_end, aspect_sentiment])

    return adjusted_labels, stats


# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_jsonl(filepath: Path) -> List[Dict]:
    """Load a JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], filepath: Path):
    """Save data to a JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def process_sample(
    sample: Dict,
    split: str,
    sample_id: str
) -> Tuple[Dict, Dict[str, int]]:
    """
    Process a single sample.

    Returns:
        - Processed sample
        - Label statistics
    """
    original_text = sample['text']
    original_labels = sample.get('labels', [])

    # Clean text
    cleaned_text, _ = clean_text(original_text)

    # Adjust labels
    adjusted_labels, label_stats = adjust_labels(
        original_text, cleaned_text, original_labels
    )

    processed = {
        'id': sample_id,
        'text': cleaned_text,
        'labels': adjusted_labels,
        'original_text': original_text,
        'split': split
    }

    return processed, label_stats


def main():
    """Main processing pipeline."""
    print("=" * 60)
    print("UIT-ViSD4SA Preprocessing Script")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Statistics
    total_stats = {
        'total_samples': 0,
        'with_annotations': 0,
        'without_annotations': 0,
        'labels': {
            'total': 0,
            'valid': 0,
            'adjusted': 0,
            'removed': 0
        },
        'per_split': {}
    }

    cleaned_data = []
    no_annotation_data = []

    # Process each split
    for split in SPLITS:
        filepath = DATA_DIR / f"{split}.jsonl"

        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping...")
            continue

        print(f"\nProcessing {split}...")
        data = load_jsonl(filepath)

        split_stats = {
            'total': len(data),
            'with_annotations': 0,
            'without_annotations': 0
        }

        for idx, sample in enumerate(data):
            sample_id = f"{split}_{idx:05d}"

            processed, label_stats = process_sample(sample, split, sample_id)

            # Update label statistics
            for key in ['total', 'valid', 'adjusted', 'removed']:
                total_stats['labels'][key] += label_stats[key]

            # Separate samples by annotation status
            if processed['labels']:
                cleaned_data.append(processed)
                split_stats['with_annotations'] += 1
            else:
                no_annotation_data.append(processed)
                split_stats['without_annotations'] += 1

        total_stats['per_split'][split] = split_stats
        total_stats['total_samples'] += split_stats['total']
        total_stats['with_annotations'] += split_stats['with_annotations']
        total_stats['without_annotations'] += split_stats['without_annotations']

        print(f"  - Total: {split_stats['total']}")
        print(f"  - With annotations: {split_stats['with_annotations']}")
        print(f"  - Without annotations: {split_stats['without_annotations']}")

    # Save outputs
    print("\n" + "-" * 40)
    print("Saving outputs...")

    # Save cleaned data
    cleaned_path = OUTPUT_DIR / "cleaned_data.jsonl"
    save_jsonl(cleaned_data, cleaned_path)
    print(f"  - Saved {len(cleaned_data)} samples to {cleaned_path}")

    # Save no annotation samples
    no_annotation_path = OUTPUT_DIR / "no_annotation_samples.jsonl"
    save_jsonl(no_annotation_data, no_annotation_path)
    print(f"  - Saved {len(no_annotation_data)} samples to {no_annotation_path}")

    # Save report
    report_path = OUTPUT_DIR / "preprocessing_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(total_stats, f, indent=2, ensure_ascii=False)
    print(f"  - Saved report to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples processed: {total_stats['total_samples']}")
    print(f"Samples with annotations: {total_stats['with_annotations']}")
    print(f"Samples without annotations: {total_stats['without_annotations']}")
    print(f"\nLabel statistics:")
    print(f"  - Total labels: {total_stats['labels']['total']}")
    print(f"  - Valid (unchanged): {total_stats['labels']['valid']}")
    print(f"  - Adjusted: {total_stats['labels']['adjusted']}")
    print(f"  - Removed (invalid): {total_stats['labels']['removed']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
