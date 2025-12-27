"""
Vietnamese Aspect-Based Sentiment Analysis Demo
================================================
A Gradio-based web application for demonstrating ABSA on Vietnamese text.

Usage:
    python app.py
    # Open http://localhost:7860
"""

from importlib import reload
import sys
from pathlib import Path

# Add parent directory to path for importing preprocessing
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from scripts.preprocessing import preprocess_text
from data_loader import lookup_text, get_sample_texts
from utils import highlight_text, format_labels


def process(raw_text: str):
    """
    Main processing function.

    Args:
        raw_text: Raw input text from user

    Returns:
        Tuple of (cleaned_text, highlighted_html, labels_str)
    """
    if not raw_text or not raw_text.strip():
        return "", "<div style='color:#888;'>Please enter some text</div>", "", ""

    # Step 1: Preprocess the text
    cleaned = preprocess_text(raw_text)

    # Step 2: Lookup in dataset
    result = lookup_text(cleaned)

    if result:
        labels = result["labels"]
    else:
        labels = []

    # Step 3: Create highlighted HTML
    highlighted = highlight_text(cleaned, labels)

    # Step 4: Format labels for display
    labels_str = format_labels(labels)

    return cleaned, highlighted, labels_str


def load_example(example_text: str):
    """Load an example into the input box."""
    return example_text


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

# Get sample texts for examples
sample_texts = get_sample_texts(5)

# Custom CSS for better styling
custom_css = """
.gradio-container {
    max-width: 1200px !important;
}
"""

# Build the interface
with gr.Blocks(
    title="Vietnamese ABSA Demo",
    css=custom_css,
    theme=gr.themes.Soft()
) as demo:


    # Input Section
    with gr.Row():
        input_text = gr.Textbox(
            label="Input (Raw Text)",
            placeholder="Nhập text tiếng Việt tại đây...\nVí dụ: Máy đẹp, sang, pin trâu, camera chụp tốt.",
            lines=4,
            max_lines=10
        )

    # Generate Button
    with gr.Row():
        btn = gr.Button("Generate", variant="primary", size="lg")
        clear_btn = gr.Button("Clear", variant="secondary", size="lg")

    # Preprocessed Text
    with gr.Row():
        cleaned_text = gr.Textbox(
            label="Preprocessed Text",
            lines=2,
            interactive=False
        )

    # Output Section
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Output (Highlighted)")
            output_html = gr.HTML()

        with gr.Column(scale=1):
            labels_text = gr.Textbox(
                label="Labels",
                lines=8,
                interactive=False
            )

    # Examples Section
    if sample_texts:
        gr.Markdown("---")
        gr.Markdown("### Examples (click to load)")
        with gr.Row():
            for i, text in enumerate(sample_texts[:3]):
                # Truncate long texts for button display
                display_text = text[:50] + "..." if len(text) > 50 else text
                example_btn = gr.Button(display_text, size="sm")
                example_btn.click(
                    fn=lambda t=text: t,
                    outputs=input_text
                )

    # Event handlers
    btn.click(
        fn=process,
        inputs=input_text,
        outputs=[cleaned_text, output_html, labels_text]
    )

    clear_btn.click(
        fn=lambda: ("", "", "", ""),
        outputs=[input_text, cleaned_text, labels_text]
    )

    # Also trigger on Enter key in input
    input_text.submit(
        fn=process,
        inputs=input_text,
        outputs=[cleaned_text, output_html, labels_text]
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Starting Vietnamese ABSA Demo...")
    print("Open http://localhost:7860 in your browser")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
