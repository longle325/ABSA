"""
Utility functions for Demo Application
Provides highlighting and formatting functions.
"""

from typing import List
import html

# =============================================================================
# COLOR CONSTANTS
# =============================================================================

# Aspect background colors (light, pastel colors for readability)
ASPECT_COLORS = {
    "BATTERY": "#FFD6E0",      # Light Pink
    "CAMERA": "#D6F5FF",       # Light Cyan
    "DESIGN": "#E8D6FF",       # Light Lavender
    "FEATURES": "#D6FFE8",     # Light Mint
    "GENERAL": "#FFF5D6",      # Light Yellow
    "PERFORMANCE": "#D6FFD6",  # Light Green
    "PRICE": "#FFE8D6",        # Light Orange
    "SCREEN": "#D6E8FF",       # Light Blue
    "SER&ACC": "#E8E8E8",      # Light Gray
    "STORAGE": "#F5EED6",      # Light Beige
}

# Sentiment text colors
SENTIMENT_COLORS = {
    "POSITIVE": "#008000",  # Green
    "NEGATIVE": "#CC0000",  # Red
    "NEUTRAL": "#666666",   # Gray
}


# =============================================================================
# HIGHLIGHTING FUNCTIONS
# =============================================================================

def highlight_text(text: str, labels: List[List]) -> str:
    """
    Convert text + labels to HTML with highlighting.

    Args:
        text: Cleaned text
        labels: List of [start, end, "ASPECT#SENTIMENT"]

    Returns:
        HTML string with highlighted spans
    """
    if not labels:
        # No labels - just return the text in a styled div
        return f'<div style="font-size:16px;line-height:2;padding:10px;">{html.escape(text)}</div>'

    # Sort labels by start position (reverse to avoid index shifting when inserting HTML)
    sorted_labels = sorted(labels, key=lambda x: x[0], reverse=True)

    # Escape HTML in the original text first
    # But we need to track positions, so we'll build the HTML differently
    result = list(text)  # Convert to list for easier manipulation

    # Process each label (from end to start to avoid position shifting)
    for label in sorted_labels:
        start, end, label_str = label

        # Validate positions
        if start < 0 or end > len(text) or start >= end:
            continue

        # Parse aspect and sentiment
        try:
            aspect, sentiment = label_str.split("#")
        except ValueError:
            continue

        # Get colors
        bg_color = ASPECT_COLORS.get(aspect, "#FFFFFF")
        text_color = SENTIMENT_COLORS.get(sentiment, "#000000")

        # Get the span text and escape it
        span_text = html.escape(text[start:end])

        # Create highlighted HTML
        highlighted = (
            f'<span style="'
            f'background-color:{bg_color};'
            f'color:{text_color};'
            f'padding:2px 4px;'
            f'border-radius:3px;'
            f'font-weight:500;'
            f'">'
            f'{span_text}'
            f'<sub style="color:#888;font-size:0.65em;margin-left:2px;">{label_str}</sub>'
            f'</span>'
        )

        # Replace the span in result
        result[start:end] = [highlighted]

    # Join and wrap in container
    final_html = ''.join(result)

    return f'''
    <div style="
        font-size:16px;
        line-height:2.2;
        padding:15px;
        background:#fafafa;
        border-radius:8px;
        border:1px solid #eee;
    ">
        {final_html}
    </div>
    '''


def format_labels(labels: List[List]) -> str:
    """
    Format labels for display.

    Args:
        labels: List of [start, end, "ASPECT#SENTIMENT"]

    Returns:
        Formatted string with each label on a new line
    """
    if not labels:
        return "No labels found"

    lines = []
    for i, label in enumerate(labels, 1):
        start, end, label_str = label
        lines.append(f"{i}. [{start}, {end}] {label_str}")

    return "\n".join(lines)

