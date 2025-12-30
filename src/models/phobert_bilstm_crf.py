"""
PhoBERT + BiLSTM-CRF Model for Vietnamese ABSA Sequence Labeling

Combines PhoBERT Vietnamese language model with BiLSTM and CRF layers
for state-of-the-art sequence labeling performance.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import numpy as np

try:
    from transformers import AutoModel, AutoTokenizer
    from torchcrf import CRF
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class PhoBERTDataset(Dataset):
    """PyTorch Dataset for PhoBERT sequence labeling."""

    def __init__(
        self,
        tokens: List[List[str]],
        tags: List[List[str]],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 256
    ):
        self.tokens = tokens
        self.tags = tags
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        tags = self.tags[idx]

        # Tokenize with PhoBERT tokenizer
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Get word IDs for subword-to-word alignment
        word_ids = encoding.word_ids()

        # Create label sequence aligned with subwords
        label_ids = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                # Special tokens get -100 (ignored in loss)
                label_ids.append(-100)
            elif word_id != previous_word_id:
                # First subword of a word gets the label
                if word_id < len(tags):
                    label_ids.append(self.label2id.get(tags[word_id], 0))
                else:
                    label_ids.append(-100)
            else:
                # Other subwords get -100 (or same label, depending on strategy)
                label_ids.append(-100)

            previous_word_id = word_id

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'word_ids': torch.tensor([w if w is not None else -1 for w in word_ids], dtype=torch.long)
        }


class PhoBERTBiLSTMCRF(nn.Module):
    """
    PhoBERT + BiLSTM + CRF model for sequence labeling.

    Architecture:
    1. PhoBERT encoder (768-dim)
    2. BiLSTM layer (bidirectional)
    3. Linear projection
    4. CRF layer for structured prediction
    """

    def __init__(
        self,
        num_labels: int,
        pretrained_model: str = 'vinai/phobert-base',
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        lstm_dropout: float = 0.3,
        freeze_bert: bool = False
    ):
        super().__init__()

        self.num_labels = num_labels

        # PhoBERT encoder
        self.phobert = AutoModel.from_pretrained(pretrained_model)
        self.hidden_size = self.phobert.config.hidden_size  # 768

        # Optionally freeze PhoBERT
        if freeze_bert:
            for param in self.phobert.parameters():
                param.requires_grad = False

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0
        )

        # Linear projection to label space
        self.fc = nn.Linear(lstm_hidden * 2, num_labels)

        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            labels: Optional labels for training (batch, seq_len)

        Returns:
            Tuple of (loss, emissions) during training
            Or (None, predictions) during inference
        """
        # PhoBERT encoding
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, 768)

        # Dropout
        sequence_output = self.dropout(sequence_output)

        # BiLSTM
        lstm_out, _ = self.lstm(sequence_output)  # (batch, seq_len, lstm_hidden*2)
        lstm_out = self.dropout(lstm_out)

        # Project to label space
        emissions = self.fc(lstm_out)  # (batch, seq_len, num_labels)

        if labels is not None:
            # Training: compute CRF loss
            # Create mask from attention_mask and valid labels
            mask = attention_mask.bool() & (labels != -100)

            # Replace -100 with 0 for CRF (will be masked anyway)
            labels_for_crf = labels.clone()
            labels_for_crf[labels == -100] = 0

            loss = -self.crf(emissions, labels_for_crf, mask=mask, reduction='mean')
            return loss, emissions
        else:
            # Inference: decode best path
            mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask=mask)
            return None, predictions


class PhoBERTModel:
    """
    High-level wrapper for PhoBERT + BiLSTM-CRF model.

    Handles training, evaluation, and prediction.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PhoBERT model.

        Args:
            config: Configuration dictionary
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and pytorch-crf required. "
                "Install with: pip install transformers pytorch-crf"
            )

        self.config = config or {}

        # Model config
        self.pretrained_model = self.config.get('pretrained_model', 'vinai/phobert-base')
        self.lstm_hidden = self.config.get('lstm_hidden', 256)
        self.lstm_layers = self.config.get('lstm_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        self.lstm_dropout = self.config.get('lstm_dropout', 0.3)
        self.freeze_bert = self.config.get('freeze_bert', False)
        self.max_length = self.config.get('max_length', 256)

        # Training config
        self.epochs = self.config.get('epochs', 20)
        self.batch_size = self.config.get('batch_size', 16)
        self.bert_lr = self.config.get('bert_lr', 2e-5)
        self.lstm_lr = self.config.get('lstm_lr', 1e-3)
        self.crf_lr = self.config.get('crf_lr', 1e-2)
        self.weight_decay = self.config.get('weight_decay', 0.01)
        self.gradient_clip = self.config.get('gradient_clip', 1.0)
        self.warmup_ratio = self.config.get('warmup_ratio', 0.1)

        # Initialize tokenizer (use_fast=True required for word_ids())
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model, use_fast=True)

        # Model and label mappings (set during training)
        self.model = None
        self.label2id = None
        self.id2label = None
        self.device = None
        self.is_trained = False

    def _build_label_vocab(self, tags: List[List[str]]) -> None:
        """Build label vocabulary from training tags."""
        labels = set()
        for seq in tags:
            labels.update(seq)

        self.label2id = {label: i for i, label in enumerate(sorted(labels))}
        self.id2label = {i: label for label, i in self.label2id.items()}

    def train(
        self,
        train_tokens: List[List[str]],
        train_tags: List[List[str]],
        dev_tokens: Optional[List[List[str]]] = None,
        dev_tags: Optional[List[List[str]]] = None
    ) -> Dict[str, float]:
        """
        Train PhoBERT model.

        Args:
            train_tokens: Training token sequences
            train_tags: Training BIO tag sequences
            dev_tokens: Optional development token sequences
            dev_tags: Optional development tag sequences

        Returns:
            Training metrics
        """
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Build label vocabulary
        self._build_label_vocab(train_tags)
        num_labels = len(self.label2id)
        print(f"Number of labels: {num_labels}")

        # Create datasets
        train_dataset = PhoBERTDataset(
            train_tokens, train_tags, self.tokenizer,
            self.label2id, self.max_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        dev_loader = None
        if dev_tokens and dev_tags:
            dev_dataset = PhoBERTDataset(
                dev_tokens, dev_tags, self.tokenizer,
                self.label2id, self.max_length
            )
            dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size)

        # Initialize model
        self.model = PhoBERTBiLSTMCRF(
            num_labels=num_labels,
            pretrained_model=self.pretrained_model,
            lstm_hidden=self.lstm_hidden,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            lstm_dropout=self.lstm_dropout,
            freeze_bert=self.freeze_bert
        ).to(self.device)

        # Optimizer with different learning rates
        optimizer = AdamW([
            {'params': self.model.phobert.parameters(), 'lr': self.bert_lr},
            {'params': self.model.lstm.parameters(), 'lr': self.lstm_lr},
            {'params': self.model.fc.parameters(), 'lr': self.lstm_lr},
            {'params': self.model.crf.parameters(), 'lr': self.crf_lr},
        ], weight_decay=self.weight_decay)

        # Learning rate scheduler
        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        # Training loop
        best_dev_f1 = 0.0
        results = {'train_losses': [], 'dev_f1s': []}

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            total_loss = 0.0

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                leave=False
            )

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                loss, _ = self.model(input_ids, attention_mask, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / len(train_loader)
            results['train_losses'].append(avg_loss)

            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

            # Evaluation on dev set
            if dev_loader:
                dev_f1 = self._evaluate(dev_loader)
                results['dev_f1s'].append(dev_f1)
                print(f"Dev F1: {dev_f1:.4f}")

                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1

        self.is_trained = True
        results['best_dev_f1'] = best_dev_f1
        results['final_loss'] = results['train_losses'][-1]

        return results

    def _evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on dataloader."""
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']

                _, predictions = self.model(input_ids, attention_mask)

                # Convert predictions and labels
                for pred_seq, label_seq, mask in zip(
                    predictions,
                    labels.tolist(),
                    attention_mask.tolist()
                ):
                    # Filter by mask and valid labels
                    pred_filtered = []
                    label_filtered = []

                    for p, l, m in zip(pred_seq, label_seq, mask):
                        if m == 1 and l != -100:
                            pred_filtered.append(self.id2label[p])
                            label_filtered.append(self.id2label[l])

                    if pred_filtered:
                        all_preds.append(pred_filtered)
                        all_labels.append(label_filtered)

        # Compute F1
        from ..evaluation.metrics import evaluate_sequence_labeling
        metrics = evaluate_sequence_labeling(all_labels, all_preds)
        return metrics['f1']

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        """
        Predict BIO tags for token sequences.

        Args:
            tokens: List of token sequences

        Returns:
            List of predicted BIO tag sequences
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for token_seq in tokens:
                # Tokenize
                encoding = self.tokenizer(
                    token_seq,
                    is_split_into_words=True,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                word_ids = encoding.word_ids()

                _, predictions = self.model(input_ids, attention_mask)
                pred_seq = predictions[0]

                # Map subword predictions back to words
                word_preds = []
                previous_word_id = None

                for i, word_id in enumerate(word_ids):
                    if word_id is not None and word_id != previous_word_id:
                        if i < len(pred_seq):
                            word_preds.append(self.id2label[pred_seq[i]])
                    previous_word_id = word_id

                # Pad or truncate to match input length
                while len(word_preds) < len(token_seq):
                    word_preds.append('O')
                word_preds = word_preds[:len(token_seq)]

                all_predictions.append(word_preds)

        return all_predictions

    def save(self, path: str) -> None:
        """Save model to disk."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'label2id': self.label2id,
            'id2label': self.id2label
        }, path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        checkpoint = torch.load(path, map_location='cpu')

        self.config = checkpoint['config']
        self.label2id = checkpoint['label2id']
        self.id2label = checkpoint['id2label']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = PhoBERTBiLSTMCRF(
            num_labels=len(self.label2id),
            pretrained_model=self.config.get('pretrained_model', 'vinai/phobert-base'),
            lstm_hidden=self.config.get('lstm_hidden', 256),
            lstm_layers=self.config.get('lstm_layers', 2),
            dropout=self.config.get('dropout', 0.1),
            lstm_dropout=self.config.get('lstm_dropout', 0.3),
            freeze_bert=self.config.get('freeze_bert', False)
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True


if __name__ == "__main__":
    print("PhoBERT + BiLSTM-CRF Model")
    print("=" * 60)

    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers and pytorch-crf not installed")
        print("Install with: pip install transformers pytorch-crf")
    else:
        print("Dependencies available!")

        # Sample test (will download PhoBERT model)
        print("\nTo test, run:")
        print("  from src.models.phobert_bilstm_crf import PhoBERTModel")
        print("  model = PhoBERTModel({'epochs': 1, 'batch_size': 2})")
        print("  model.train(train_tokens, train_tags)")
