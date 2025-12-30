"""
Logistic Regression Model for Vietnamese ABSA Sequence Labeling

Uses TF-IDF features with context windows for token-level classification.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from .base_model import BaseABSAModel


class LogRegModel(BaseABSAModel):
    """
    Logistic Regression model for token-level ABSA classification.

    Uses TF-IDF features with context window approach.
    Each token is classified independently based on its context.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Logistic Regression model.

        Args:
            config: Configuration dictionary with keys:
                - max_features: Max TF-IDF features (default: 10000)
                - ngram_range: N-gram range tuple (default: (1, 2))
                - context_window: Context window size (default: 2)
                - C: Regularization strength (default: 1.0)
                - max_iter: Max iterations (default: 1000)
        """
        super().__init__(config)

        # Config with defaults
        self.max_features = config.get('max_features', 10000) if config else 10000
        self.ngram_range = config.get('ngram_range', (1, 2)) if config else (1, 2)
        self.context_window = config.get('context_window', 2) if config else 2
        self.C = config.get('C', 1.0) if config else 1.0
        self.max_iter = config.get('max_iter', 1000) if config else 1000

        # Initialize components
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
            lowercase=True
        )

        self.classifier = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            multi_class='multinomial',
            solver='lbfgs',
            n_jobs=-1,
            random_state=42
        )

        self.label_encoder = LabelEncoder()

    def _create_context_string(self, tokens: List[str], idx: int) -> str:
        """
        Create context string for a token.

        Args:
            tokens: Token sequence
            idx: Current token index

        Returns:
            Context string combining surrounding tokens
        """
        start = max(0, idx - self.context_window)
        end = min(len(tokens), idx + self.context_window + 1)

        context_parts = []

        # Add position markers
        for i in range(start, end):
            offset = i - idx
            token = tokens[i]

            if offset == 0:
                context_parts.append(f"CURR:{token}")
            elif offset < 0:
                context_parts.append(f"PREV{-offset}:{token}")
            else:
                context_parts.append(f"NEXT{offset}:{token}")

        # Also add raw context
        context_parts.append(' '.join(tokens[start:end]))

        return ' '.join(context_parts)

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
        Train Logistic Regression model.

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
        print(f"Fitting TF-IDF vectorizer (max_features={self.max_features})...")

        # Fit vectorizer and transform
        X_train = self.vectorizer.fit_transform(train_contexts)

        # Encode labels
        y_train = self.label_encoder.fit_transform(train_labels)

        print(f"Training Logistic Regression (C={self.C}, max_iter={self.max_iter})...")

        # Train classifier
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        # Compute training F1
        y_train_pred = self.classifier.predict(X_train)
        train_f1 = f1_score(y_train, y_train_pred, average='macro')

        results = {
            'train_f1_macro': train_f1,
            'num_features': X_train.shape[1],
            'num_classes': len(self.label_encoder.classes_)
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

        print(f"Training complete. Train F1 (macro): {train_f1:.4f}")

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

    def get_feature_importance(self, label: str, top_n: int = 20) -> List[tuple]:
        """
        Get most important features for a label.

        Args:
            label: Target label
            top_n: Number of top features

        Returns:
            List of (feature_name, weight) tuples
        """
        if not self.is_trained:
            return []

        try:
            label_idx = self.label_encoder.transform([label])[0]
        except ValueError:
            return []

        feature_names = self.vectorizer.get_feature_names_out()
        weights = self.classifier.coef_[label_idx]

        # Get top positive and negative weights
        indices = np.argsort(weights)
        top_positive = [(feature_names[i], weights[i]) for i in indices[-top_n:]][::-1]
        top_negative = [(feature_names[i], weights[i]) for i in indices[:top_n]]

        return top_positive + top_negative


if __name__ == "__main__":
    # Test LogReg model
    print("Testing Logistic Regression Model")
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

    # Create and train model
    config = {'max_features': 1000, 'context_window': 2}
    model = LogRegModel(config)

    results = model.train(train_tokens, train_tags)
    print(f"\nTraining results: {results}")

    # Predict
    test_tokens = [['Pin', 'rất', 'tốt']]
    predictions = model.predict(test_tokens)
    print(f"\nPredictions for {test_tokens[0]}: {predictions[0]}")
