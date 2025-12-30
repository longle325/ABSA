"""
BiLSTM-CRF Model with Syllable + Character Embeddings

Architecture:
- Syllable Embedding: Trainable lookup table for Vietnamese syllables
- Character Embedding: CNN over characters in each word
- BiLSTM: Bidirectional LSTM for sequence encoding
- CRF: Conditional Random Field for structured prediction
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

try:
    from torchcrf import CRF
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False


class BiLSTMCRFDataset(Dataset):
    """PyTorch Dataset for BiLSTM-CRF sequence labeling."""

    def __init__(
        self,
        tokens: List[List[str]],
        tags: List[List[str]],
        syllable2id: Dict[str, int],
        char2id: Dict[str, int],
        label2id: Dict[str, int],
        max_word_len: int = 20
    ):
        self.tokens = tokens
        self.tags = tags
        self.syllable2id = syllable2id
        self.char2id = char2id
        self.label2id = label2id
        self.max_word_len = max_word_len

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        tags = self.tags[idx]

        # Convert syllables to IDs
        syllable_ids = [
            self.syllable2id.get(tok.lower(), self.syllable2id.get('<unk>', 1))
            for tok in tokens
        ]

        # Convert characters to IDs for each word
        char_ids = []
        for tok in tokens:
            word_chars = [
                self.char2id.get(c, self.char2id.get('<unk>', 1))
                for c in tok[:self.max_word_len]
            ]
            # Pad to max_word_len
            word_chars = word_chars + [0] * (self.max_word_len - len(word_chars))
            char_ids.append(word_chars)

        # Convert labels to IDs
        label_ids = [self.label2id.get(tag, 0) for tag in tags]

        return {
            'syllable_ids': torch.tensor(syllable_ids, dtype=torch.long),
            'char_ids': torch.tensor(char_ids, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'length': len(tokens)
        }


def collate_fn(batch):
    """Collate function for variable-length sequences."""
    # Sort by length (descending) for pack_padded_sequence
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)

    syllable_ids = [item['syllable_ids'] for item in batch]
    char_ids = [item['char_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    lengths = [item['length'] for item in batch]

    # Pad sequences
    syllable_ids_padded = pad_sequence(syllable_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    # Pad char_ids (batch_size, max_seq_len, max_word_len)
    max_seq_len = max(lengths)
    max_word_len = char_ids[0].size(1)
    char_ids_padded = torch.zeros(len(batch), max_seq_len, max_word_len, dtype=torch.long)
    for i, chars in enumerate(char_ids):
        char_ids_padded[i, :chars.size(0), :] = chars

    return {
        'syllable_ids': syllable_ids_padded,
        'char_ids': char_ids_padded,
        'labels': labels_padded,
        'lengths': torch.tensor(lengths, dtype=torch.long)
    }


class CharCNN(nn.Module):
    """Character-level CNN for word representation."""

    def __init__(
        self,
        char_vocab_size: int,
        char_embed_dim: int = 30,
        num_filters: int = 50,
        kernel_sizes: List[int] = [3, 4, 5]
    ):
        super().__init__()
        self.char_embedding = nn.Embedding(
            char_vocab_size, char_embed_dim, padding_idx=0
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, num_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])

        self.output_dim = num_filters * len(kernel_sizes)

    def forward(self, char_ids):
        """
        Args:
            char_ids: (batch_size, seq_len, max_word_len)
        Returns:
            char_repr: (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, max_word_len = char_ids.size()

        # Reshape for batch processing
        char_ids = char_ids.view(-1, max_word_len)  # (batch*seq, max_word_len)

        # Embed characters
        char_embeds = self.char_embedding(char_ids)  # (batch*seq, max_word_len, char_embed_dim)
        char_embeds = char_embeds.transpose(1, 2)  # (batch*seq, char_embed_dim, max_word_len)

        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(char_embeds))  # (batch*seq, num_filters, max_word_len)
            pooled = torch.max(conv_out, dim=2)[0]  # (batch*seq, num_filters)
            conv_outputs.append(pooled)

        # Concatenate and reshape
        char_repr = torch.cat(conv_outputs, dim=1)  # (batch*seq, output_dim)
        char_repr = char_repr.view(batch_size, seq_len, -1)  # (batch, seq, output_dim)

        return char_repr


class BiLSTMCRFNetwork(nn.Module):
    """
    BiLSTM-CRF Network with Syllable + Character embeddings.

    Architecture:
    - Syllable Embedding (trainable)
    - Character CNN
    - BiLSTM
    - CRF
    """

    def __init__(
        self,
        syllable_vocab_size: int,
        char_vocab_size: int,
        num_labels: int,
        syllable_embed_dim: int = 100,
        char_embed_dim: int = 30,
        char_num_filters: int = 50,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()

        if not CRF_AVAILABLE:
            raise ImportError("pytorch-crf is required. Install with: pip install pytorch-crf")

        # Syllable embedding
        self.syllable_embedding = nn.Embedding(
            syllable_vocab_size, syllable_embed_dim, padding_idx=0
        )

        # Character CNN
        self.char_cnn = CharCNN(
            char_vocab_size,
            char_embed_dim=char_embed_dim,
            num_filters=char_num_filters
        )

        # Calculate input dimension for BiLSTM
        lstm_input_dim = syllable_embed_dim + self.char_cnn.output_dim

        # BiLSTM
        self.lstm = nn.LSTM(
            lstm_input_dim,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Linear projection to labels
        self.hidden2tag = nn.Linear(lstm_hidden * 2, num_labels)

        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)

        self.num_labels = num_labels

    def forward(self, syllable_ids, char_ids, lengths):
        """
        Get emission scores from BiLSTM.

        Args:
            syllable_ids: (batch_size, seq_len)
            char_ids: (batch_size, seq_len, max_word_len)
            lengths: (batch_size,)

        Returns:
            emissions: (batch_size, seq_len, num_labels)
        """
        batch_size, seq_len = syllable_ids.size()

        # Get syllable embeddings
        syllable_embeds = self.syllable_embedding(syllable_ids)  # (batch, seq, syllable_dim)

        # Get character embeddings
        char_embeds = self.char_cnn(char_ids)  # (batch, seq, char_dim)

        # Concatenate embeddings
        embeds = torch.cat([syllable_embeds, char_embeds], dim=-1)
        embeds = self.dropout(embeds)

        # Pack for LSTM
        packed = pack_padded_sequence(
            embeds, lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        # BiLSTM
        lstm_out, _ = self.lstm(packed)

        # Unpack
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_len)
        lstm_out = self.dropout(lstm_out)

        # Project to label space
        emissions = self.hidden2tag(lstm_out)

        return emissions

    def loss(self, syllable_ids, char_ids, labels, lengths, mask=None):
        """
        Compute negative log-likelihood loss.
        """
        emissions = self.forward(syllable_ids, char_ids, lengths)

        if mask is None:
            mask = labels != -100

        # Replace -100 with 0 for CRF (will be masked anyway)
        labels = labels.clone()
        labels[labels == -100] = 0

        # CRF loss (negative log-likelihood)
        loss = -self.crf(emissions, labels, mask=mask, reduction='mean')

        return loss

    def decode(self, syllable_ids, char_ids, lengths):
        """
        Viterbi decoding.
        """
        emissions = self.forward(syllable_ids, char_ids, lengths)

        # Create mask
        batch_size, seq_len = syllable_ids.size()
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=syllable_ids.device)
        for i, length in enumerate(lengths):
            mask[i, :length] = True

        # Decode
        predictions = self.crf.decode(emissions, mask=mask)

        return predictions


class BiLSTMCRFModel:
    """
    BiLSTM-CRF Model wrapper for training and inference.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Model hyperparameters
        self.syllable_embed_dim = self.config.get('syllable_embed_dim', 100)
        self.char_embed_dim = self.config.get('char_embed_dim', 30)
        self.char_num_filters = self.config.get('char_num_filters', 50)
        self.lstm_hidden = self.config.get('lstm_hidden', 256)
        self.lstm_layers = self.config.get('lstm_layers', 2)
        self.dropout = self.config.get('dropout', 0.5)

        # Training hyperparameters
        self.epochs = self.config.get('epochs', 30)
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.max_word_len = self.config.get('max_word_len', 20)

        # Vocabularies
        self.syllable2id = {'<pad>': 0, '<unk>': 1}
        self.char2id = {'<pad>': 0, '<unk>': 1}
        self.label2id = {}
        self.id2label = {}

        # Model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_vocab(self, tokens: List[List[str]], tags: List[List[str]]):
        """Build syllable, character, and label vocabularies."""
        # Build syllable vocab
        for seq in tokens:
            for tok in seq:
                tok_lower = tok.lower()
                if tok_lower not in self.syllable2id:
                    self.syllable2id[tok_lower] = len(self.syllable2id)

        # Build character vocab
        for seq in tokens:
            for tok in seq:
                for char in tok:
                    if char not in self.char2id:
                        self.char2id[char] = len(self.char2id)

        # Build label vocab
        for seq in tags:
            for tag in seq:
                if tag not in self.label2id:
                    self.label2id[tag] = len(self.label2id)

        self.id2label = {v: k for k, v in self.label2id.items()}

        print(f"Syllable vocab size: {len(self.syllable2id)}")
        print(f"Character vocab size: {len(self.char2id)}")
        print(f"Label vocab size: {len(self.label2id)}")

    def train(
        self,
        train_tokens: List[List[str]],
        train_tags: List[List[str]],
        dev_tokens: Optional[List[List[str]]] = None,
        dev_tags: Optional[List[List[str]]] = None
    ) -> Dict[str, float]:
        """
        Train the BiLSTM-CRF model.
        """
        # Build vocabularies
        self.build_vocab(train_tokens, train_tags)

        # Create model
        self.model = BiLSTMCRFNetwork(
            syllable_vocab_size=len(self.syllable2id),
            char_vocab_size=len(self.char2id),
            num_labels=len(self.label2id),
            syllable_embed_dim=self.syllable_embed_dim,
            char_embed_dim=self.char_embed_dim,
            char_num_filters=self.char_num_filters,
            lstm_hidden=self.lstm_hidden,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout
        ).to(self.device)

        # Create datasets
        train_dataset = BiLSTMCRFDataset(
            train_tokens, train_tags,
            self.syllable2id, self.char2id, self.label2id,
            max_word_len=self.max_word_len
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        dev_loader = None
        if dev_tokens and dev_tags:
            dev_dataset = BiLSTMCRFDataset(
                dev_tokens, dev_tags,
                self.syllable2id, self.char2id, self.label2id,
                max_word_len=self.max_word_len
            )
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )

        # Optimizer
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_f1 = 0.0
        results = {}

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for batch in progress_bar:
                optimizer.zero_grad()

                syllable_ids = batch['syllable_ids'].to(self.device)
                char_ids = batch['char_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['lengths']

                loss = self.model.loss(syllable_ids, char_ids, labels, lengths)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

            # Evaluate on dev set
            if dev_loader:
                f1 = self._evaluate(dev_loader)
                print(f"Dev F1: {f1:.4f}")
                if f1 > best_f1:
                    best_f1 = f1
                    results['best_dev_f1'] = best_f1
                    results['best_epoch'] = epoch + 1

        results['final_train_loss'] = avg_loss
        return results

    def _evaluate(self, data_loader) -> float:
        """Evaluate on a dataset and return F1 score."""
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                syllable_ids = batch['syllable_ids'].to(self.device)
                char_ids = batch['char_ids'].to(self.device)
                labels = batch['labels']
                lengths = batch['lengths']

                predictions = self.model.decode(syllable_ids, char_ids, lengths)

                for i, (pred, length) in enumerate(zip(predictions, lengths)):
                    true_labels = labels[i, :length].tolist()
                    # Filter out -100
                    valid_indices = [j for j, l in enumerate(true_labels) if l != -100]
                    all_preds.append([self.id2label[pred[j]] for j in valid_indices])
                    all_labels.append([self.id2label[true_labels[j]] for j in valid_indices])

        # Calculate F1 using seqeval
        try:
            from seqeval.metrics import f1_score
            return f1_score(all_labels, all_preds, average='macro')
        except ImportError:
            return 0.0

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        """Predict BIO tags for input tokens."""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        self.model.eval()

        # Create dummy tags for dataset
        dummy_tags = [['O'] * len(seq) for seq in tokens]

        dataset = BiLSTMCRFDataset(
            tokens, dummy_tags,
            self.syllable2id, self.char2id, self.label2id,
            max_word_len=self.max_word_len
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn)

        all_predictions = []

        with torch.no_grad():
            for batch in loader:
                syllable_ids = batch['syllable_ids'].to(self.device)
                char_ids = batch['char_ids'].to(self.device)
                lengths = batch['lengths']

                predictions = self.model.decode(syllable_ids, char_ids, lengths)

                for pred, length in zip(predictions, lengths):
                    pred_tags = [self.id2label[p] for p in pred[:length]]
                    all_predictions.append(pred_tags)

        return all_predictions

    def save(self, path: str):
        """Save model to disk."""
        import pickle
        save_dict = {
            'config': self.config,
            'syllable2id': self.syllable2id,
            'char2id': self.char2id,
            'label2id': self.label2id,
            'id2label': self.id2label,
            'model_state_dict': self.model.state_dict() if self.model else None
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
        self.syllable2id = save_dict['syllable2id']
        self.char2id = save_dict['char2id']
        self.label2id = save_dict['label2id']
        self.id2label = save_dict['id2label']

        # Rebuild model
        self.model = BiLSTMCRFNetwork(
            syllable_vocab_size=len(self.syllable2id),
            char_vocab_size=len(self.char2id),
            num_labels=len(self.label2id),
            syllable_embed_dim=self.config.get('syllable_embed_dim', 100),
            char_embed_dim=self.config.get('char_embed_dim', 30),
            char_num_filters=self.config.get('char_num_filters', 50),
            lstm_hidden=self.config.get('lstm_hidden', 256),
            lstm_layers=self.config.get('lstm_layers', 2),
            dropout=self.config.get('dropout', 0.5)
        ).to(self.device)

        self.model.load_state_dict(save_dict['model_state_dict'])
        print(f"Model loaded from {path}")
