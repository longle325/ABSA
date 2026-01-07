"""
PhoBERT-CRF Model for Vietnamese ABSA

Architecture:
- PhoBERT-base-v2: Vietnamese pre-trained BERT (768d, 12 layers)
- Dropout layer (0.1)
- Linear projection (768 → num_labels)
- CRF layer for structured prediction

This model follows 2025 best practices:
- BERT + CRF (no BiLSTM - BERT already bidirectional)
- Direct PhoBERT fine-tuning
- Simpler than BiLSTM-CRF-XLMR but expected better performance

Expected F1: 65-68% (best performing model)
Training time: 2-3 hours on GPU
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

try:
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from torchcrf import CRF
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False

from .base_model import BaseABSAModel


class PhoBERTCRFDataset(Dataset):
    """PyTorch Dataset for PhoBERT-CRF sequence labeling."""

    def __init__(
        self,
        tokens: List[List[str]],
        tags: List[List[str]],
        label2id: Dict[str, int],
        phobert_tokenizer,
        max_seq_len: int = 256
    ):
        self.tokens = tokens
        self.tags = tags
        self.label2id = label2id
        self.tokenizer = phobert_tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        tags = self.tags[idx]

        # Limit sequence length (leave room for <s> and </s>)
        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[:self.max_seq_len - 2]
            tags = tags[:self.max_seq_len - 2]

        # PhoBERT uses RobertaTokenizer which doesn't have word_ids()
        # We need manual word-to-subword alignment

        # Tokenize each word individually to track subwords
        input_ids = [self.tokenizer.cls_token_id]  # <s> token
        word_mask = [0]  # No word at <s>
        label_ids = [-100]  # Ignore <s> in loss

        for word, tag in zip(tokens, tags):
            # Tokenize word (without special tokens)
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)

            if len(input_ids) + len(word_tokens) + 1 > self.max_seq_len:
                # Would exceed max length
                break

            # Add subword tokens
            input_ids.extend(word_tokens)

            # First subword gets: word_mask=1, label=tag
            # Other subwords get: word_mask=0, label=-100
            for i, _ in enumerate(word_tokens):
                if i == 0:
                    word_mask.append(1)
                    label_ids.append(self.label2id.get(tag, 0))
                else:
                    word_mask.append(0)
                    label_ids.append(-100)

        # Add </s> token
        input_ids.append(self.tokenizer.sep_token_id)
        word_mask.append(0)
        label_ids.append(-100)

        # Pad to max_seq_len
        seq_len = len(input_ids)
        attention_mask = [1] * seq_len

        padding_len = self.max_seq_len - seq_len
        if padding_len > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
            attention_mask.extend([0] * padding_len)
            word_mask.extend([0] * padding_len)
            label_ids.extend([-100] * padding_len)

        # Count actual words (where word_mask=1)
        num_words = sum(word_mask)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'word_mask': torch.tensor(word_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'num_words': num_words
        }


def collate_fn_phobert(batch):
    """Collate function for PhoBERT batching."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    word_mask = torch.stack([item['word_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    num_words = [item['num_words'] for item in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'word_mask': word_mask,
        'labels': labels,
        'num_words': torch.tensor(num_words, dtype=torch.long)
    }


class PhoBERTCRFNetwork(nn.Module):
    """
    PhoBERT-CRF Network for sequence labeling.

    Architecture:
    - PhoBERT-base-v2 (768d hidden size)
    - Dropout (0.1 default)
    - Linear projection (768 → num_labels)
    - CRF layer
    """

    def __init__(
        self,
        num_labels: int,
        phobert_model_name: str = 'vinai/phobert-base-v2',
        dropout: float = 0.1,
        freeze_phobert: bool = False
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")
        if not CRF_AVAILABLE:
            raise ImportError("pytorch-crf is required. Install with: pip install pytorch-crf")

        # PhoBERT model
        self.phobert = AutoModel.from_pretrained(phobert_model_name)
        self.phobert_dim = self.phobert.config.hidden_size  # 768 for base

        if freeze_phobert:
            for param in self.phobert.parameters():
                param.requires_grad = False

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Linear projection
        self.classifier = nn.Linear(self.phobert_dim, num_labels)

        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)

    def forward(
        self,
        input_ids,
        attention_mask,
        word_mask,
        num_words,
        labels=None
    ):
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len) - PhoBERT subword IDs
            attention_mask: (batch_size, seq_len) - Attention mask
            word_mask: (batch_size, seq_len) - 1 for first subword of each word
            num_words: (batch_size,) - Number of words in each sequence
            labels: (batch_size, max_num_words) - Optional labels for training

        Returns:
            If labels provided: loss (scalar)
            If labels not provided: predictions (list of lists)
        """
        # PhoBERT encoding
        outputs = self.phobert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, 768)

        # Aggregate subwords to words using word_mask
        # Extract only the first subword representation of each word
        batch_size, seq_len, hidden_dim = sequence_output.size()
        max_num_words = word_mask.sum(dim=1).max().item()

        # Create word-level representations
        word_repr = []
        for i in range(batch_size):
            # Get indices where word_mask is 1
            word_indices = (word_mask[i] == 1).nonzero(as_tuple=True)[0]
            # Extract representations at those positions
            words = sequence_output[i, word_indices, :]  # (num_words, 768)
            word_repr.append(words)

        # Pad to same length
        word_repr_padded = torch.zeros(batch_size, max_num_words, hidden_dim, device=sequence_output.device)
        for i, words in enumerate(word_repr):
            word_repr_padded[i, :words.size(0), :] = words

        # Dropout + Linear
        word_repr_padded = self.dropout(word_repr_padded)
        emissions = self.classifier(word_repr_padded)  # (batch_size, max_num_words, num_labels)

        # Create mask for CRF (based on actual number of words)
        mask = torch.zeros(batch_size, max_num_words, dtype=torch.bool, device=emissions.device)
        for i, n in enumerate(num_words):
            mask[i, :n] = True

        if labels is not None:
            # Training: compute CRF loss
            # Extract word-level labels (ignore -100 labels)
            word_labels = []
            for i in range(batch_size):
                # Get non-ignored labels
                sample_labels = labels[i][labels[i] != -100]
                word_labels.append(sample_labels[:num_words[i]])

            # Pad labels
            word_labels_padded = torch.zeros(batch_size, max_num_words, dtype=torch.long, device=labels.device)
            for i, wl in enumerate(word_labels):
                word_labels_padded[i, :len(wl)] = wl

            # CRF negative log-likelihood loss
            loss = -self.crf(emissions, word_labels_padded, mask=mask, reduction='mean')
            return loss
        else:
            # Inference: CRF decoding
            predictions = self.crf.decode(emissions, mask=mask)
            return predictions


class PhoBERTCRFModel(BaseABSAModel):
    """
    PhoBERT-CRF model for Vietnamese ABSA.

    Simpler and more effective than BiLSTM-CRF-XLMR:
    - Uses PhoBERT-base-v2 (Vietnamese-specific)
    - No BiLSTM (BERT already bidirectional)
    - No syllable/character embeddings (PhoBERT handles well)
    - Direct BERT → CRF architecture (2025 best practice)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PhoBERT-CRF model.

        Args:
            config: Configuration dictionary with keys:
                - phobert_model_name: HuggingFace model name (default: vinai/phobert-base-v2)
                - dropout: Dropout rate (default: 0.1)
                - freeze_phobert: Freeze PhoBERT weights (default: False)
                - epochs: Number of training epochs (default: 10)
                - batch_size: Batch size (default: 16)
                - learning_rate: Learning rate (default: 2e-5)
                - max_seq_len: Maximum sequence length (default: 256)
                - warmup_steps: Number of warmup steps (default: 500)
                - weight_decay: Weight decay (default: 0.01)
        """
        super().__init__(config)

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")

        # Config
        self.phobert_model_name = config.get('phobert_model_name', 'vinai/phobert-base-v2')
        self.dropout = config.get('dropout', 0.1)
        self.freeze_phobert = config.get('freeze_phobert', False)
        self.epochs = config.get('epochs', 10)
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.max_seq_len = config.get('max_seq_len', 256)
        self.warmup_steps = config.get('warmup_steps', 500)
        self.weight_decay = config.get('weight_decay', 0.01)

        # Initialize tokenizer (PhoBERT doesn't have fast tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.phobert_model_name,
            use_fast=False
        )

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Vocabularies
        self.label2id = {}
        self.id2label = {}

        # Model
        self.model = None

    def build_vocabs(self, tokens: List[List[str]], tags: List[List[str]]):
        """Build label vocabulary."""
        # Collect all unique labels
        all_labels = set()
        for tag_seq in tags:
            all_labels.update(tag_seq)

        # Create label2id and id2label
        self.label2id = {label: idx for idx, label in enumerate(sorted(all_labels))}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        print(f"Found {len(self.label2id)} unique labels")

    def train(
        self,
        train_tokens: List[List[str]],
        train_tags: List[List[str]],
        dev_tokens: Optional[List[List[str]]] = None,
        dev_tags: Optional[List[List[str]]] = None
    ) -> Dict[str, float]:
        """
        Train the PhoBERT-CRF model.

        Args:
            train_tokens: Training token sequences
            train_tags: Training BIO tag sequences
            dev_tokens: Optional development token sequences
            dev_tags: Optional development tag sequences

        Returns:
            Dictionary with training metrics
        """
        print("Building vocabularies...")
        self.build_vocabs(train_tokens, train_tags)

        # Create datasets
        print("Creating datasets...")
        train_dataset = PhoBERTCRFDataset(
            train_tokens, train_tags, self.label2id,
            self.tokenizer, self.max_seq_len
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_phobert
        )

        dev_loader = None
        if dev_tokens and dev_tags:
            dev_dataset = PhoBERTCRFDataset(
                dev_tokens, dev_tags, self.label2id,
                self.tokenizer, self.max_seq_len
            )
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn_phobert
            )

        # Initialize model
        print(f"Initializing PhoBERT-CRF model ({self.phobert_model_name})...")
        self.model = PhoBERTCRFNetwork(
            num_labels=len(self.label2id),
            phobert_model_name=self.phobert_model_name,
            dropout=self.dropout,
            freeze_phobert=self.freeze_phobert
        ).to(self.device)

        # Optimizer with different learning rates for PhoBERT and classifier
        if self.freeze_phobert:
            # Only train classifier and CRF
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in self.model.named_parameters()
                              if 'phobert' not in n and p.requires_grad],
                    'lr': self.learning_rate
                }
            ]
        else:
            # Fine-tune PhoBERT with smaller LR, train classifier with higher LR
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in self.model.named_parameters()
                              if 'phobert' in n and p.requires_grad],
                    'lr': self.learning_rate
                },
                {
                    'params': [p for n, p in self.model.named_parameters()
                              if 'phobert' not in n and p.requires_grad],
                    'lr': self.learning_rate * 10  # Classifier gets higher LR
                }
            ]

        optimizer = AdamW(optimizer_grouped_parameters, weight_decay=self.weight_decay)

        # Learning rate scheduler
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        # Training loop
        print(f"\nTraining for {self.epochs} epochs...")
        print(f"Total steps: {total_steps}")
        print(f"Device: {self.device}")

        best_dev_f1 = 0.0
        patience_counter = 0
        max_patience = 3

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print("-" * 60)

            # Training
            self.model.train()
            train_loss = 0.0
            train_steps = 0

            progress_bar = tqdm(train_loader, desc="Training")
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                word_mask = batch['word_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                num_words = batch['num_words']

                # Forward pass
                loss = self.model(input_ids, attention_mask, word_mask, num_words, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_steps += 1

                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train_loss = train_loss / train_steps
            print(f"Average training loss: {avg_train_loss:.4f}")

            # Validation
            if dev_loader:
                from ..evaluation.metrics import evaluate_sequence_labeling

                self.model.eval()
                dev_predictions = []
                dev_ground_truth = []

                with torch.no_grad():
                    for batch in tqdm(dev_loader, desc="Validation"):
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        word_mask = batch['word_mask'].to(self.device)
                        labels = batch['labels']
                        num_words = batch['num_words']

                        # Predictions
                        predictions = self.model(input_ids, attention_mask, word_mask, num_words)

                        # Convert to BIO tags
                        for i, pred in enumerate(predictions):
                            pred_tags = [self.id2label[p] for p in pred]
                            dev_predictions.append(pred_tags)

                            # Get ground truth
                            true_labels = labels[i][labels[i] != -100][:num_words[i]]
                            true_tags = [self.id2label[l.item()] for l in true_labels]
                            dev_ground_truth.append(true_tags)

                # Evaluate
                metrics = evaluate_sequence_labeling(dev_ground_truth, dev_predictions)
                dev_f1 = metrics['f1']

                print(f"Dev Precision: {metrics['precision']:.4f}")
                print(f"Dev Recall:    {metrics['recall']:.4f}")
                print(f"Dev F1:        {dev_f1:.4f}")

                # Early stopping
                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    patience_counter = 0
                    print(f"New best F1: {best_dev_f1:.4f}")
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{max_patience}")

                    if patience_counter >= max_patience:
                        print("Early stopping triggered!")
                        break

        self.is_trained = True
        return {
            'best_dev_f1': best_dev_f1 if dev_loader else 0.0,
            'final_train_loss': avg_train_loss
        }

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        """
        Predict BIO tags for token sequences.

        Args:
            tokens: List of token sequences

        Returns:
            List of predicted BIO tag sequences
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self.model.eval()

        # Create dataset
        dummy_tags = [['O'] * len(seq) for seq in tokens]  # Dummy tags for dataset
        dataset = PhoBERTCRFDataset(
            tokens, dummy_tags, self.label2id,
            self.tokenizer, self.max_seq_len
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn_phobert
        )

        all_predictions = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                word_mask = batch['word_mask'].to(self.device)
                num_words = batch['num_words']

                # Predictions
                predictions = self.model(input_ids, attention_mask, word_mask, num_words)

                # Convert to BIO tags
                for pred in predictions:
                    pred_tags = [self.id2label[p] for p in pred]
                    all_predictions.append(pred_tags)

        return all_predictions

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")

        import pickle
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'config': self.config,
            'label2id': self.label2id,
            'id2label': self.id2label,
            'model_state_dict': self.model.state_dict()
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        import pickle

        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.config = save_dict['config']
        self.label2id = save_dict['label2id']
        self.id2label = save_dict['id2label']

        # Reinitialize tokenizer (PhoBERT doesn't have fast tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get('phobert_model_name', 'vinai/phobert-base-v2'),
            use_fast=False
        )

        # Rebuild model
        self.model = PhoBERTCRFNetwork(
            num_labels=len(self.label2id),
            phobert_model_name=self.config.get('phobert_model_name', 'vinai/phobert-base-v2'),
            dropout=self.config.get('dropout', 0.1),
            freeze_phobert=self.config.get('freeze_phobert', False)
        ).to(self.device)

        self.model.load_state_dict(save_dict['model_state_dict'])
        self.is_trained = True

        print(f"Model loaded from {path}")
