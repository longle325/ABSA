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
from model_manager import get_model_manager
from inference import inference_with_fallback, map_labels_to_raw_text

# Global mapping from display name to model file
model_name_to_file = {}


def process(raw_text: str, selected_model_display: str):
    """
    Main processing function with model inference.

    Args:
        raw_text: Raw input text from user
        selected_model_display: Selected model display name from dropdown

    Returns:
        Tuple of (highlighted_html, labels_str)
    """
    if not raw_text or not raw_text.strip():
        return "<div style='color:#888;'>Please enter some text</div>", ""

    # Get model manager
    manager = get_model_manager()

    # Get actual model filename from display name
    model_file = model_name_to_file.get(selected_model_display)

    # Get model from cache (already pre-loaded)
    model = manager.get_current_model()
    if model_file:
        try:
            model = manager.load_model(model_file)
        except Exception as e:
            return f"<div style='color:#cc0000;'>Error loading model</div>", ""

    # Smart inference with fallback (dataset lookup first, then model)
    cleaned, labels_on_cleaned, source = inference_with_fallback(
        raw_text, model, lookup_text
    )

    # Map labels from cleaned text back to raw text positions
    labels_on_raw = map_labels_to_raw_text(raw_text, cleaned, labels_on_cleaned)

    # Create highlighted HTML on RAW text
    highlighted = highlight_text(raw_text, labels_on_raw)

    # Format labels for display
    labels_str = format_labels(labels_on_raw)

    return highlighted, labels_str


def load_example(example_text: str):
    """Load an example into the input box."""
    return example_text


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

# Initialize model manager and pre-load models
print("=" * 60)
print("Initializing Vietnamese ABSA Demo...")
print("=" * 60)

manager = get_model_manager()
available_models = manager.get_available_models()

print(f"Found {len(available_models)} models:")
for model in available_models:
    print(f"  - {model['name']} ({model['type']})")

# Pre-load models: CRF, PhoBERT-CRF, and BiLSTM-CRF-XLMR
print("\nPre-loading models...")
model_files = ['crf_model.pkl', 'phobert_crf_model.pkl', 'bilstm_crf_xlmr_model.pkl']
for i, model_file in enumerate(model_files):
    try:
        print(f"  Loading {model_file}...")
        manager.load_model(model_file)
        print(f"  ✓ {model_file} loaded")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n✓ Models ready!")
print("=" * 60)

# Create model choices for dropdown with mapping
model_choices = []
model_name_to_file = {}

for model_file, model_info in zip(model_files, available_models):
    display_name = f"{model_info['name']} ({model_info['type']})"
    model_choices.append(display_name)
    model_name_to_file[display_name] = model_file

default_model = model_choices[0] if model_choices else None

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

    gr.Markdown("# Vietnamese Aspect-Based Sentiment Analysis")
    gr.Markdown("Analyze Vietnamese reviews for aspect-based sentiment")

    # Model Selection
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=model_choices,
            value=default_model,
            label="Select Model",
            info=f"Choose from {len(model_choices)} pre-loaded models"
        )

    gr.Markdown("---")

    # Input Section
    with gr.Row():
        input_text = gr.Textbox(
            label="Input",
            placeholder="Nhập text tiếng Việt tại đây...\nVí dụ: Máy đẹp, sang, pin trâu, camera chụp tốt.",
            lines=5,
            max_lines=10
        )

    # Buttons - centered
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=2):
            with gr.Row():
                btn = gr.Button("Analyze", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", variant="secondary", size="lg")
        with gr.Column(scale=1):
            pass

    # Output Section - 65% output, 35% labels
    with gr.Row():
        with gr.Column(scale=65):
            gr.Markdown("### Output")
            output_html = gr.HTML()

        with gr.Column(scale=35):
            gr.Markdown("### Labels")
            labels_text = gr.Textbox(
                label="",
                lines=12,
                interactive=False,
                show_label=False
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
        inputs=[input_text, model_dropdown],
        outputs=[output_html, labels_text]
    )

    clear_btn.click(
        fn=lambda: ("", ""),
        outputs=[input_text, labels_text]
    )

    # Also trigger on Enter key in input
    input_text.submit(
        fn=process,
        inputs=[input_text, model_dropdown],
        outputs=[output_html, labels_text]
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
