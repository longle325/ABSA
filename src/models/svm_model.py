"""
SVM Model for Vietnamese ABSA Sequence Labeling

Uses TF-IDF features with RBF kernel for token-level classification.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from .base_model import BaseABSAModel


class SVMModel(BaseABSAModel):
    """
    SVM model for token-level ABSA classification.

    Uses TF-IDF features with RBF kernel.
    More powerful than Logistic Regression but computationally expensive.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SVM model.

        Args:
            config: Configuration dictionary with keys:
                - max_features: Max TF-IDF features (default: 15000)
                - ngram_range: N-gram range tuple (default: (1, 3))
                - context_window: Context window size (default: 3)
                - kernel: SVM kernel (default: 'rbf')
                - C: Regularization strength (default: 1.0)
                - gamma: Kernel coefficient (default: 'scale')
        """
        super().__init__(config)

        # Config with defaults
        self.max_features = config.get('max_features', 15000) if config else 15000
        self.ngram_range = config.get('ngram_range', (1, 3)) if config else (1, 3)
        self.context_window = config.get('context_window', 3) if config else 3
        self.kernel = config.get('kernel', 'rbf') if config else 'rbf'
        self.C = config.get('C', 1.0) if config else 1.0
        self.gamma = config.get('gamma', 'scale') if config else 'scale'

        # Initialize components
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
            lowercase=True
        )

        self.classifier = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            decision_function_shape='ovr',
            random_state=42
        )

        self.label_encoder = LabelEncoder()

    def _create_context_string(self, tokens: List[str], idx: int) -> str:
        """
        Create rich context string for a token.

        Args:
            tokens: Token sequence
            idx: Current token index

        Returns:
            Context string with position markers and n-grams
        """
        start = max(0, idx - self.context_window)
        end = min(len(tokens), idx + self.context_window + 1)

        parts = []

        # Current token (emphasized)
        current = tokens[idx]
        parts.append(f"CURR:{current}")
        parts.append(f"CURR_LOWER:{current.lower()}")

        # Position-aware context
        for i in range(start, end):
            if i == idx:
                continue

            offset = i - idx
            token = tokens[i]

            if offset < 0:
                parts.append(f"PREV{-offset}:{token}")
            else:
                parts.append(f"NEXT{offset}:{token}")

        # Local bigrams
        if idx > 0:
            parts.append(f"BIGRAM_PREV:{tokens[idx-1].lower()}_{current.lower()}")
        if idx < len(tokens) - 1:
            parts.append(f"BIGRAM_NEXT:{current.lower()}_{tokens[idx+1].lower()}")

        # Local trigrams
        if idx > 0 and idx < len(tokens) - 1:
            parts.append(f"TRIGRAM:{tokens[idx-1].lower()}_{current.lower()}_{tokens[idx+1].lower()}")

        # Raw context window
        parts.append(' '.join(tokens[start:end]))

        # Word shape features as text
        if current.isupper():
            parts.append("SHAPE:UPPER")
        elif current.istitle():
            parts.append("SHAPE:TITLE")
        elif current.isdigit():
            parts.append("SHAPE:DIGIT")

        return ' '.join(parts)

    def _prepare_data(
        self,
        tokens_list: List[List[str]],
        tags_list: Optional[List[List[str]]] = None
    ) -> tuple:
        """
        Prepare data for training/prediction.

        Args:
            tokens_list: List of token sequences
            tags_list: Optional list of tag sequences

        Returns:
            Tuple of (context_strings, labels) or just context_strings
        """
        contexts = []
        labels = []

        for seq_idx, tokens in enumerate(tokens_list):
            for tok_idx in range(len(tokens)):
                context = self._create_context_string(tokens, tok_idx)
                contexts.append(context)

                if tags_list is not None:
                    labels.append(tags_list[seq_idx][tok_idx])

        if tags_list is not None:
            return contexts, labels
        return contexts, None

    def train(
        self,
        train_tokens: List[List[str]],
        train_tags: List[List[str]],
        dev_tokens: Optional[List[List[str]]] = None,
        dev_tags: Optional[List[List[str]]] = None
    ) -> Dict[str, float]:
        """
        Train SVM model.

        Args:
            train_tokens: Training token sequences
            train_tags: Training BIO tag sequences
            dev_tokens: Optional development token sequences
            dev_tags: Optional development tag sequences

        Returns:
            Training metrics
        """
        print(f"Preparing {len(train_tokens)} training samples...")

        # Prepare data
        train_contexts, train_labels = self._prepare_data(train_tokens, train_tags)

        print(f"Created {len(train_contexts)} token instances")
        print(f"Fitting TF-IDF vectorizer (max_features={self.max_features}, ngram_range={self.ngram_range})...")

        # Fit vectorizer and transform
        X_train = self.vectorizer.fit_transform(train_contexts)

        # Encode labels
        y_train = self.label_encoder.fit_transform(train_labels)

        print(f"Training SVM (kernel={self.kernel}, C={self.C}, gamma={self.gamma})...")
        print("This may take a while...")

        # Train classifier
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        # Compute training F1 (on a sample for speed)
        sample_size = min(5000, len(train_contexts))
        sample_indices = np.random.choice(len(train_contexts), sample_size, replace=False)

        X_sample = X_train[sample_indices]
        y_sample = y_train[sample_indices]
        y_sample_pred = self.classifier.predict(X_sample)
        train_f1 = f1_score(y_sample, y_sample_pred, average='macro')

        results = {
            'train_f1_macro': train_f1,
            'num_features': X_train.shape[1],
            'num_classes': len(self.label_encoder.classes_),
            'num_support_vectors': self.classifier.n_support_.sum()
        }

        # Evaluate on dev set
        if dev_tokens and dev_tags:
            print(f"Evaluating on {len(dev_tokens)} dev samples...")
            dev_predictions = self.predict(dev_tokens)

            # Flatten for comparison
            dev_tags_flat = [tag for seq in dev_tags for tag in seq]
            dev_pred_flat = [tag for seq in dev_predictions for tag in seq]

            dev_f1 = f1_score(
                self.label_encoder.transform(dev_tags_flat),
                self.label_encoder.transform(dev_pred_flat),
                average='macro'
            )
            results['dev_f1_macro'] = dev_f1
            print(f"Dev F1 (macro): {dev_f1:.4f}")

        print(f"Training complete. Train F1 (macro, sampled): {train_f1:.4f}")

        return results

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

        all_predictions = []

        for token_seq in tokens:
            # Create context strings for each token
            contexts = [self._create_context_string(token_seq, i)
                       for i in range(len(token_seq))]

            # Transform and predict
            X = self.vectorizer.transform(contexts)
            y_pred = self.classifier.predict(X)

            # Decode labels
            pred_tags = self.label_encoder.inverse_transform(y_pred)
            all_predictions.append(list(pred_tags))

        return all_predictions


class LinearSVMModel(SVMModel):
    """
    Linear SVM variant - faster than RBF kernel.

    Use this for faster training when RBF is too slow.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        config['kernel'] = 'linear'
        super().__init__(config)


if __name__ == "__main__":
    # Test SVM model
    print("Testing SVM Model")
    print("=" * 60)

    # Sample data
    train_tokens = [
        ['Pin', 'cực', 'trâu', ',', 'camera', 'đẹp'],
        ['Máy', 'lag', 'quá', ',', 'pin', 'tệ'],
        ['Màn', 'hình', 'đẹp', 'và', 'pin', 'tốt'],
    ]

    train_tags = [
        ['B-BATTERY#POSITIVE', 'I-BATTERY#POSITIVE', 'I-BATTERY#POSITIVE', 'O', 'B-CAMERA#POSITIVE', 'I-CAMERA#POSITIVE'],
        ['B-PERFORMANCE#NEGATIVE', 'I-PERFORMANCE#NEGATIVE', 'I-PERFORMANCE#NEGATIVE', 'O', 'B-BATTERY#NEGATIVE', 'I-BATTERY#NEGATIVE'],
        ['B-SCREEN#POSITIVE', 'I-SCREEN#POSITIVE', 'I-SCREEN#POSITIVE', 'O', 'B-BATTERY#POSITIVE', 'I-BATTERY#POSITIVE'],
    ]

    # Create and train model (use linear for speed in test)
    config = {'max_features': 1000, 'kernel': 'linear', 'context_window': 2}
    model = SVMModel(config)

    results = model.train(train_tokens, train_tags)
    print(f"\nTraining results: {results}")

    # Predict
    test_tokens = [['Pin', 'rất', 'tốt']]
    predictions = model.predict(test_tokens)
    print(f"\nPredictions for {test_tokens[0]}: {predictions[0]}")
