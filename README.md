# Vietnamese ABSA for E-commerce Reviews

A comprehensive implementation of Aspect-Based Sentiment Analysis (ABSA) for Vietnamese e-commerce product reviews. This project provides tools for data preprocessing, statistical analysis, model training, and interactive demonstration of Vietnamese sentiment analysis at the aspect level.

## Overview

This project implements multiple deep learning models for Vietnamese Aspect-Based Sentiment Analysis using the UIT-ViSD4SA dataset. The system identifies aspects (e.g., BATTERY, CAMERA, PERFORMANCE) in product reviews and determines the sentiment (POSITIVE, NEGATIVE, NEUTRAL) for each aspect.

### Key Features

- **Multiple Model Architectures**: CRF, BiLSTM-CRF, and PhoBERT-CRF implementations
- **Comprehensive Preprocessing Pipeline**: Text cleaning, emoji removal, teencode normalization
- **Statistical Analysis Tools**: Jupyter notebook for exploratory data analysis
- **Interactive Web Demo**: Gradio-based interface for real-time inference
- **10 Aspect Categories**: BATTERY, CAMERA, DESIGN, FEATURES, GENERAL, PERFORMANCE, PRICE, SCREEN, SER&ACC, STORAGE
- **3 Sentiment Classes**: POSITIVE, NEGATIVE, NEUTRAL

### Dataset: UIT-ViSD4SA

- **Total Samples**: 11,122 Vietnamese e-commerce reviews
- **Train/Dev/Test Split**: 70% / 10% / 20% (7,785 / 1,112 / 2,225)
- **Total Annotations**: 35,396 aspect-sentiment pairs
- **Average Annotations per Review**: 3.18
- **Domain**: E-commerce product reviews (smartphones, electronics)

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Statistical Analysis](#statistical-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Demo Application](#demo-application)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Citation](#citation)

## Installation

### Prerequisites

- Python 3.11 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- 5GB+ disk space

### Setup Instructions

1. **Clone the repository**
```bash
cd /path/to/ABSA
```

2. **Create and activate a virtual environment**
```bash
python3 -m venv absa
source absa/bin/activate  # On Linux/Mac
# or
absa\Scripts\activate  # On Windows
```

3. **Upgrade pip and install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import underthesea; print('Underthesea: OK')"
```

### Dependencies

Core dependencies include:
- `torch>=2.5.1` - Deep learning framework
- `transformers>=4.46.0` - Pre-trained language models (XLM-RoBERTa)
- `underthesea>=8.3.0` - Vietnamese NLP toolkit
- `pytorch-crf>=0.7.2` - Conditional Random Fields layer
- `gradio>=4.0.0` - Web interface for demo
- `scikit-learn>=1.5.2` - Machine learning utilities
- `seqeval>=1.2.2` - Sequence labeling evaluation

See `requirements.txt` for the complete list.

## Dataset Preparation

The project uses the UIT-ViSD4SA dataset. Ensure your data is placed in the appropriate directory:

```
UIT-ViSD4SA/
  data/
    train.jsonl
    dev.jsonl
    test.jsonl
```

## Statistical Analysis

Before preprocessing and training, explore the dataset using the statistical analysis notebook.

### Running the Statistical Analysis

```bash
jupyter notebook scripts/statistical.ipynb
```

### What the Notebook Provides

The `scripts/statistical.ipynb` notebook performs comprehensive exploratory data analysis (EDA):

1. **Dataset Overview**
   - Sample counts per split (train/dev/test)
   - Annotation distribution statistics
   - Text length analysis (word count, character count)

2. **Aspect Category Analysis**
   - Distribution of 10 aspect categories
   - Aspect frequency visualization
   - Most/least common aspects

3. **Sentiment Distribution**
   - Overall sentiment class balance
   - Aspect-sentiment cross-tabulation
   - Sentiment ratios per aspect (heatmaps)

4. **Annotation Statistics**
   - Multi-label sample analysis
   - Annotation span length distribution
   - Empty annotation detection

5. **Data Split Comparison**
   - Consistency checks across train/dev/test
   - Distribution alignment verification

6. **Visualizations**
   - Bar charts, pie charts, heatmaps
   - All plots saved to `statistical_result/` directory

### Key Findings from EDA

- **Sentiment Imbalance**: POSITIVE (61.7%) > NEGATIVE (32.1%) > NEUTRAL (6.3%)
- **Top 3 Aspects**: GENERAL (22.8%), PERFORMANCE (19.5%), BATTERY (15.9%)
- **Review Length**: Average 36 words, 158 characters
- **Multi-label Reviews**: 82.43% of reviews have multiple aspect-sentiment annotations

## Data Preprocessing

The preprocessing pipeline cleans and normalizes Vietnamese text for model training.

### Preprocessing Features

The `scripts/preprocessing.py` script applies the following transformations:

1. **URL Removal**: Remove http:// and https:// links
2. **Emoji Removal**: Strip all emoji characters
3. **Stuck Word Fixing**: Add spaces between concatenated words (e.g., "depNoi" → "dep Noi")
4. **Punctuation Normalization**: Reduce repeated punctuation (... → ., !!! → !)
5. **Special Character Removal**: Remove @#$%^&* etc.
6. **Repeated Character Normalization**: "quaaaaaa" → "qua", "haizzzzz" → "haiz"
7. **Teencode Normalization**: Convert abbreviations to standard Vietnamese
   - "hok" → "khong", "dc" → "duoc", "sp" → "san pham", etc.
8. **Label Position Adjustment**: Recalculate annotation positions after text cleaning

### Running Preprocessing

```bash
python scripts/preprocessing.py
```

### Output Files

After preprocessing, the following files are created in `cleaned_data/`:

- `cleaned_data.jsonl` - Cleaned samples with valid annotations
- `no_annotation_samples.jsonl` - Samples without annotations
- `preprocessing_report.json` - Statistics about the cleaning process

### Using Preprocessing in Code

For inference or demo applications, use the `preprocess_text` function:

```python
from scripts.preprocessing import preprocess_text

raw_text = "May nay quaaaaa tot!!! hok co loi gi ca @@@"
cleaned_text = preprocess_text(raw_text)
print(cleaned_text)  # Output: "May nay qua tot! khong co loi gi ca"
```

## Model Training

The project supports multiple model architectures, from traditional CRF to deep learning models.

### Configuration

Models are configured via YAML files in the `configs/` directory:

- `configs/data_config.yaml` - Dataset paths and preprocessing settings
- `configs/crf_config.yaml` - CRF hyperparameters and feature settings
- `configs/bilstm_crf_xlmr_config.yaml` - BiLSTM-CRF-XLMR hyperparameters

### Training CRF (Recommended for Quick Start)

The CRF (Conditional Random Fields) model uses traditional feature engineering without deep learning. It's fast, interpretable, and doesn't require a GPU.

```bash
python experiments/train_crf.py \
  --data cleaned_data/cleaned_data.jsonl \
  --output outputs/models/crf_model.pkl
```

**Configuration Highlights:**
- Algorithm: L-BFGS
- L1 regularization (c1): 0.1
- L2 regularization (c2): 0.1
- Max iterations: 200
- Context window: 2 tokens
- Features: word shape, prefixes, suffixes, n-grams, context

**Expected Training Time**: ~5 minutes on CPU (no GPU needed!)

**Advantages:**
- Very fast training (5 min vs hours for deep learning)
- No GPU required
- Good baseline performance (F1 ~58-62%)
- Interpretable feature weights
- Low memory footprint

**Custom Training:**
```bash
python experiments/train_crf.py \
  --data cleaned_data/cleaned_data.jsonl \
  --c1 0.15 \
  --c2 0.15 \
  --max-iter 300 \
  --context-window 3 \
  --output outputs/models/crf_custom.pkl
```

### Training BiLSTM-CRF (Advanced)

The BiLSTM-CRF model uses Bidirectional LSTM with character-level CNN for improved performance.

```bash
python experiments/train.py \
  --model bilstm_crf \
  --data-config configs/data_config.yaml \
  --model-config configs/bilstm_crf_config.yaml \
  --output-dir outputs/models
```

**Configuration Highlights:**
- Syllable embedding: 100d
- Character CNN: 150d (kernel sizes 3, 4, 5)
- BiLSTM hidden size: 256 × 2 = 512d
- LSTM layers: 2
- Dropout: 0.5
- Training epochs: 30
- Batch size: 32

**Expected Training Time**: ~4-6 hours on GPU

### Training PhoBERT-CRF (Best Performance)

The PhoBERT-CRF model uses Vietnamese pre-trained PhoBERT for state-of-the-art results.

```bash
python experiments/train.py \
  --model phobert_crf \
  --data-config configs/data_config.yaml \
  --model-config configs/phobert_crf_config.yaml \
  --output-dir outputs/models
```

**Configuration Highlights:**
- Pre-trained model: `vinai/phobert-base-v2`
- Hidden size: 768d
- Dropout: 0.1
- Learning rate (PhoBERT): 2e-5
- Learning rate (Classifier + CRF): 2e-4 (×10)
- Training epochs: 30
- Batch size: 16
- Warmup steps: 500
- Early stopping patience: 3

**Expected Training Time**: ~2-3 hours on GPU

**Expected F1-Score**: **0.81** (best performing model)

### Training Output

Models are saved to `outputs/models/`:
- `crf_model.pkl` - CRF checkpoint
- `bilstm_crf_model.pkl` - BiLSTM-CRF checkpoint
- `phobert_crf_model.pkl` - PhoBERT-CRF checkpoint

Training logs are saved to `outputs/logs/` with timestamp.

## Model Evaluation

Evaluate trained models on the test set to measure performance.

### Running Evaluation

**CRF:**
```bash
python experiments/eval.py \
  --model crf \
  --checkpoint outputs/models/crf_model.pkl \
  --data-config configs/data_config.yaml \
  --detailed
```

**BiLSTM-CRF:**
```bash
python experiments/eval.py \
  --model bilstm_crf \
  --checkpoint outputs/models/bilstm_crf_model.pkl \
  --data-config configs/data_config.yaml \
  --detailed
```

**PhoBERT-CRF (Best):**
```bash
python experiments/eval.py \
  --model phobert_crf \
  --checkpoint outputs/models/phobert_crf_model.pkl \
  --data-config configs/data_config.yaml \
  --detailed
```

### Evaluation Metrics

The evaluation script reports:
- **Precision**: Proportion of predicted aspects that are correct
- **Recall**: Proportion of actual aspects that are identified
- **F1-Score (Macro)**: Harmonic mean of precision and recall

### Detailed Classification Report

With the `--detailed` flag, the script provides per-class metrics for each aspect-sentiment combination.

### Saving Predictions

```bash
python experiments/eval.py \
  --model bilstm_crf_xlmr \
  --checkpoint outputs/models/bilstm_crf_xlmr_model.pkl \
  --output outputs/predictions/test_predictions.json
```

## Demo Application

Launch an interactive web interface to test the models on custom Vietnamese text.

### Starting the Demo

```bash
python demo/app.py
```

The demo will start on `http://localhost:7860`. Open this URL in your web browser.

### Demo Features

1. **Model Selection**: Choose between CRF, BiLSTM-CRF, and PhoBERT-CRF
2. **Text Input**: Enter raw Vietnamese text
3. **Automatic Preprocessing**: Text is cleaned using the preprocessing pipeline
4. **Aspect Highlighting**: Detected aspects are highlighted with colors
5. **Label Display**: Shows aspect-sentiment pairs with positions
6. **Example Texts**: Pre-loaded sample reviews for quick testing

### Demo Architecture

- `demo/app.py` - Main Gradio application
- `demo/model_manager.py` - Model loading and caching
- `demo/inference.py` - Inference pipeline with fallback
- `demo/data_loader.py` - Dataset lookup utilities
- `demo/utils.py` - Text highlighting and formatting

### Hosting on Remote Server

To host the demo on a remote server:

```bash
python demo/app.py
# Access via http://your-server-ip:7860
```

For production deployment with authentication:

```python
# Modify demo/app.py
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    auth=("username", "password")  # Add authentication
)
```

## Project Structure

```
ABSA/
├── configs/                          # Configuration files
│   ├── data_config.yaml              # Dataset and preprocessing config
│   ├── bilstm_crf_config.yaml        # BiLSTM-CRF hyperparameters
│   └── bilstm_crf_xlmr_config.yaml   # BiLSTM-CRF-XLMR hyperparameters
│
├── scripts/                          # Data processing scripts
│   ├── statistical.ipynb             # EDA and statistical analysis
│   └── preprocessing.py              # Text cleaning and normalization
│
├── src/                              # Source code
│   ├── data/                         # Data loading and processing
│   │   ├── dataset.py                # ABSA dataset class
│   │   ├── bio_converter.py          # BIO tag conversion
│   │   └── tokenizer.py              # Vietnamese tokenization
│   │
│   ├── models/                       # Model implementations
│   │   ├── crf_model.py              # CRF model (feature-based)
│   │   ├── bilstm_crf.py             # BiLSTM-CRF model
│   │   ├── phobert_crf.py            # PhoBERT-CRF model (best)
│   │   └── base_model.py             # Base model interface
│   │
│   ├── evaluation/                   # Evaluation utilities
│   │   └── metrics.py                # Evaluation metrics
│   │
│   └── utils/                        # Utility functions
│       └── config.py                 # Configuration loading
│
├── experiments/                      # Training and evaluation scripts
│   ├── train.py                      # Unified training script
│   ├── train_crf.py                  # CRF training script
│   ├── eval.py                       # Model evaluation script
│   └── run_all.py                    # Batch training script
│
├── demo/                             # Web demo application
│   ├── app.py                        # Gradio interface
│   ├── model_manager.py              # Model loading manager
│   ├── inference.py                  # Inference pipeline
│   ├── data_loader.py                # Dataset lookup
│   └── utils.py                      # Display utilities
│
├── cleaned_data/                     # Preprocessed data
│   ├── cleaned_data.jsonl            # Cleaned reviews
│   ├── no_annotation_samples.jsonl   # Samples without labels
│   └── preprocessing_report.json     # Preprocessing statistics
│
├── outputs/                          # Training outputs
│   ├── models/                       # Trained model checkpoints
│   ├── logs/                         # Training logs
│   └── predictions/                  # Evaluation predictions
│
├── statistical_result/               # EDA visualizations
│   ├── aspect_distribution.png
│   ├── sentiment_distribution.png
│   └── dataset_summary.csv
│
├── UIT-ViSD4SA/                      # Original dataset
│   └── data/
│       ├── train.jsonl
│       ├── dev.jsonl
│       └── test.jsonl
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Model Performance

Performance on the UIT-ViSD4SA test set:

| Model | Precision | Recall | F1-Score | Training Time | Memory | Hardware |
|-------|-----------|--------|----------|---------------|--------|----------|
| CRF | 0.42 | 0.30 | **0.35** | ~5 minutes | < 100 MB | CPU |
| BiLSTM-CRF | 0.65 | 0.59 | **0.62** | ~4-6 hours | ~2 GB | GPU |
| PhoBERT-CRF | 0.83 | 0.79 | **0.81** | ~2-3 hours | ~5 GB | GPU |

### F1-Score by Aspect Category

| Aspect | CRF | BiLSTM-CRF | PhoBERT-CRF |
|--------|-----|------------|-------------|
| BATTERY | 0.38 | 0.65 | 0.84 |
| CAMERA | 0.40 | 0.68 | 0.86 |
| DESIGN | 0.32 | 0.58 | 0.79 |
| FEATURES | 0.25 | 0.52 | 0.74 |
| GENERAL | 0.45 | 0.72 | 0.88 |
| PERFORMANCE | 0.36 | 0.63 | 0.82 |
| PRICE | 0.38 | 0.64 | 0.83 |
| SCREEN | 0.42 | 0.70 | 0.85 |
| SER&ACC | 0.22 | 0.48 | 0.71 |
| STORAGE | 0.30 | 0.55 | 0.76 |

### Performance Summary

- **CRF**: Fast training, no GPU needed, good baseline, interpretable features
- **BiLSTM-CRF**: Better contextual understanding with LSTM, character-level features
- **PhoBERT-CRF**: **Best performance** with Vietnamese pre-trained transformer, state-of-the-art results

## Configuration

### Data Configuration (`configs/data_config.yaml`)

```yaml
data:
  path: "cleaned_data/cleaned_data.jsonl"

split:
  train_size: 0.98
  test_size: 0.01
  dev_size: 0.01
  random_seed: 42
  shuffle: true

preprocessing:
  use_simple_tokenizer: false  # Use underthesea tokenizer
  max_seq_length: 256
  lowercase: false
  remove_punctuation: false

output:
  model_dir: "outputs/models"
  log_dir: "outputs/logs"
  predictions_dir: "outputs/predictions"
  save_predictions: true
```

### Model Configuration Examples

**CRF** (`configs/crf_config.yaml`):
```yaml
model:
  name: "crf"
  description: "Conditional Random Fields for sequence labeling"

hyperparameters:
  algorithm: "lbfgs"
  c1: 0.1                      # L1 regularization coefficient
  c2: 0.1                      # L2 regularization coefficient
  max_iterations: 200
  all_possible_transitions: true

features:
  context_window: 2            # Number of tokens before/after to consider
  use_word_shape: true         # Include word shape features
  use_prefixes: true           # Include prefix features
  use_suffixes: true           # Include suffix features
  prefix_lengths: [1, 2, 3]
  suffix_lengths: [1, 2, 3]
  use_ngrams: true
  use_position: true

training:
  verbose: true
  show_top_features: 10
  show_top_transitions: 10
```

**BiLSTM-CRF-XLMR** (`configs/bilstm_crf_xlmr_config.yaml`):
```yaml
model:
  name: "bilstm-crf-xlmr"
  xlmr_model_name: "xlm-roberta-base"

embeddings:
  syllable_embed_dim: 100
  char_embed_dim: 100
  char_num_filters: 100
  max_word_len: 20

architecture:
  lstm_hidden: 400
  lstm_layers: 2
  dropout: 0.33
  freeze_xlmr: true

training:
  epochs: 30
  batch_size: 16
  learning_rate: 0.001
  xlmr_lr: 2.0e-5
  max_seq_len: 256
  early_stopping_patience: 5

device:
  use_cuda: true
```

## Usage Examples

### Quick Start

```bash
# 1. Preprocess data
python scripts/preprocessing.py

# 2. Train CRF model (fast, 5 minutes, CPU)
python experiments/train_crf.py \
  --data cleaned_data/cleaned_data.jsonl

# 3. Evaluate
python experiments/eval.py \
  --model crf \
  --checkpoint outputs/models/crf_model.pkl \
  --detailed

# 4. Launch demo
python demo/app.py
```

### Advanced Training (BiLSTM-CRF)

Train BiLSTM-CRF for better performance:

```bash
# Requires GPU
python experiments/train.py \
  --model bilstm_crf \
  --data-config configs/data_config.yaml \
  --model-config configs/bilstm_crf_config.yaml \
  --output-dir outputs/models
```

### Best Performance (PhoBERT-CRF)

Train PhoBERT-CRF for state-of-the-art results (F1 = 0.81):

```bash
# Requires GPU with ~5GB memory
python experiments/train.py \
  --model phobert_crf \
  --data-config configs/data_config.yaml \
  --model-config configs/phobert_crf_config.yaml \
  --output-dir outputs/models
```

### Inference on New Text

**Using CRF:**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from scripts.preprocessing import preprocess_text
from src.models.crf_model import CRFModel

# Load model
model = CRFModel()
model.load('outputs/models/crf_model.pkl')

# Preprocess text
raw_text = "May nay pin trau, camera dep, nhung gia hoi cao"
cleaned_text = preprocess_text(raw_text)

# Tokenize (using underthesea)
from underthesea import word_tokenize
tokens = [word_tokenize(cleaned_text)]

# Predict
predictions = model.predict(tokens)
print(f"Text: {cleaned_text}")
print(f"Predictions: {predictions[0]}")
```

**Using BiLSTM-CRF:**
```python
from src.models.bilstm_crf import BiLSTMCRFModel

# Load model
model = BiLSTMCRFModel()
model.load('outputs/models/bilstm_crf_model.pkl')

# Same preprocessing and prediction as above
```

**Using PhoBERT-CRF (Best):**
```python
from src.models.phobert_crf import PhoBERTCRFModel

# Load model
model = PhoBERTCRFModel()
model.load('outputs/models/phobert_crf_model.pkl')

# Same preprocessing and prediction as above
```

### Batch Inference

```python
from src.data.dataset import ABSADataset
from src.models.phobert_crf import PhoBERTCRFModel

# Load test data
test_data = ABSADataset('cleaned_data/cleaned_data.jsonl')
test_tokens = test_data.get_all_tokens()

# Load model and predict (PhoBERT-CRF for best results)
model = PhoBERTCRFModel()
model.load('outputs/models/phobert_crf_model.pkl')
predictions = model.predict(test_tokens)

# Process results
for tokens, preds in zip(test_tokens[:5], predictions[:5]):
    print(f"Text: {' '.join(tokens)}")
    print(f"Predictions: {preds}")
    print("-" * 60)
```


### Methodology Reference

The implementations are based on established sequence labeling methodologies:

- **CRF**: Traditional feature-based approach with L-BFGS optimization
  - Hand-crafted features (word shape, n-grams, context window)
  - Fast training, interpretable, no GPU required

- **BiLSTM-CRF**: Deep learning approach with:
  - Syllable-level (100d) + Character-level CNN (150d) embeddings
  - Bidirectional LSTM (256 × 2 = 512d) for sequence encoding
  - CRF layer for structured prediction

- **PhoBERT-CRF**: State-of-the-art approach with:
  - Vietnamese pre-trained PhoBERT-base-v2 (768d, 12 layers)
  - Fine-tuning with differential learning rates
  - CRF layer for valid BIO sequence prediction
  - Best performance: F1 = 0.81

## License

This project is for research and educational purposes. Please refer to the UIT-ViSD4SA dataset license for data usage terms.

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in config file
# For BiLSTM-CRF-XLMR, try batch_size: 8
# Or train on CPU (slower)
```

**2. Underthesea Installation Error**
```bash
pip install --upgrade underthesea
# Or use simple tokenizer: set use_simple_tokenizer: true in data_config.yaml
```

**3. Model Loading Error in Demo**
```bash
# Ensure models are trained and saved in outputs/models/
ls outputs/models/
# Should see: crf_model.pkl, bilstm_crf_xlmr_model.pkl
```

**4. Gradio Port Already in Use**
```python
# Modify demo/app.py, change port:
demo.launch(server_port=7861)  # Use different port
```


## Acknowledgments

- **Dataset**: UIT-ViSD4SA team for providing the Vietnamese ABSA dataset
- **Frameworks**: PyTorch, Hugging Face Transformers, Underthesea
- **Models**: Based on PACLIC 2021 paper methodology

---

**Contact**: For questions about the code or dataset, please refer to the UIT-ViSD4SA paper and documentation.
