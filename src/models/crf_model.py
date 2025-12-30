"""
CRF Model for Vietnamese ABSA Sequence Labeling

Uses sklearn-crfsuite for Conditional Random Fields.
Features include word shapes, context windows, and n-grams.

Expected F1: 58-62%
Training time: ~5 minutes
"""

from typing import List, Dict, Any, Optional
import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics

from .base_model import BaseABSAModel


class CRFModel(BaseABSAModel):
    """
    CRF sequence labeling model for ABSA.

    Uses feature engineering with:
    - Word features (lowercase, shape, prefixes, suffixes)
    - Context features (surrounding words)
    - N-gram features
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CRF model.

        Args:
            config: Configuration dictionary with keys:
                - c1: L1 regularization (default: 0.1)
                - c2: L2 regularization (default: 0.1)
                - max_iterations: Max training iterations (default: 200)
                - context_window: Context window size (default: 2)
        """
        super().__init__(config)

        # Default config
        self.c1 = config.get('c1', 0.1) if config else 0.1
        self.c2 = config.get('c2', 0.1) if config else 0.1
        self.max_iterations = config.get('max_iterations', 200) if config else 200
        self.context_window = config.get('context_window', 2) if config else 2

        # Initialize CRF
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=self.c1,
            c2=self.c2,
            max_iterations=self.max_iterations,
            all_possible_transitions=True
        )

    def word2features(self, tokens: List[str], idx: int) -> Dict[str, Any]:
        """
        Extract features for a single token.

        Args:
            tokens: Token sequence
            idx: Current token index

        Returns:
            Feature dictionary for the token
        """
        token = tokens[idx]

        # Basic word features
        features = {
            'bias': 1.0,
            'word.lower': token.lower(),
            'word.isupper': token.isupper(),
            'word.istitle': token.istitle(),
            'word.isdigit': token.isdigit(),
            'word.isalpha': token.isalpha(),
            'word.length': len(token),
        }

        # Suffix features (last N characters)
        if len(token) >= 1:
            features['word[-1:]'] = token[-1:]
        if len(token) >= 2:
            features['word[-2:]'] = token[-2:]
        if len(token) >= 3:
            features['word[-3:]'] = token[-3:]

        # Prefix features (first N characters)
        if len(token) >= 1:
            features['word[:1]'] = token[:1]
        if len(token) >= 2:
            features['word[:2]'] = token[:2]
        if len(token) >= 3:
            features['word[:3]'] = token[:3]

        # Context features
        for offset in range(-self.context_window, self.context_window + 1):
            if offset == 0:
                continue

            context_idx = idx + offset
            prefix = f'{offset:+d}:'

            if 0 <= context_idx < len(tokens):
                context_token = tokens[context_idx]
                features[f'{prefix}word.lower'] = context_token.lower()
                features[f'{prefix}word.isupper'] = context_token.isupper()
                features[f'{prefix}word.istitle'] = context_token.istitle()
                features[f'{prefix}word.isdigit'] = context_token.isdigit()

                # N-grams with current token
                if offset == -1:
                    features['bigram(-1,0)'] = f"{context_token.lower()}_{token.lower()}"
                elif offset == 1:
                    features['bigram(0,+1)'] = f"{token.lower()}_{context_token.lower()}"
            else:
                # Boundary markers
                if offset < 0:
                    features[f'{prefix}BOS'] = True
                else:
                    features[f'{prefix}EOS'] = True

        # Position features
        features['position'] = idx
        features['is_first'] = (idx == 0)
        features['is_last'] = (idx == len(tokens) - 1)

        return features

    def sent2features(self, tokens: List[str]) -> List[Dict[str, Any]]:
        """
        Extract features for all tokens in a sequence.

        Args:
            tokens: Token sequence

        Returns:
            List of feature dictionaries
        """
        return [self.word2features(tokens, i) for i in range(len(tokens))]

    def train(
        self,
        train_tokens: List[List[str]],
        train_tags: List[List[str]],
        dev_tokens: Optional[List[List[str]]] = None,
        dev_tags: Optional[List[List[str]]] = None
    ) -> Dict[str, float]:
        """
        Train CRF model.

        Args:
            train_tokens: Training token sequences
            train_tags: Training BIO tag sequences
            dev_tokens: Optional development token sequences
            dev_tags: Optional development tag sequences

        Returns:
            Training metrics
        """
        print(f"Extracting features for {len(train_tokens)} training samples...")

        # Extract features
        X_train = [self.sent2features(tokens) for tokens in train_tokens]
        y_train = train_tags

        print(f"Training CRF model (max_iter={self.max_iterations}, c1={self.c1}, c2={self.c2})...")

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Compute training metrics
        y_train_pred = self.model.predict(X_train)
        train_f1 = crf_metrics.flat_f1_score(y_train, y_train_pred, average='macro')

        results = {
            'train_f1_macro': train_f1,
        }

        # Evaluate on dev set if provided
        if dev_tokens and dev_tags:
            print(f"Evaluating on {len(dev_tokens)} dev samples...")
            X_dev = [self.sent2features(tokens) for tokens in dev_tokens]
            y_dev_pred = self.model.predict(X_dev)
            dev_f1 = crf_metrics.flat_f1_score(dev_tags, y_dev_pred, average='macro')
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

        X = [self.sent2features(t) for t in tokens]
        return self.model.predict(X)

    def get_top_transitions(self, top_n: int = 20) -> List[tuple]:
        """
        Get top transition weights (helpful for debugging).

        Args:
            top_n: Number of top transitions to return

        Returns:
            List of (weight, from_label, to_label) tuples
        """
        if not self.is_trained:
            return []

        transitions = []
        for (from_label, to_label), weight in self.model.transition_features_.items():
            transitions.append((weight, from_label, to_label))

        transitions.sort(reverse=True)
        return transitions[:top_n]

    def get_top_features(self, top_n: int = 20) -> List[tuple]:
        """
        Get top state features (helpful for debugging).

        Args:
            top_n: Number of top features to return

        Returns:
            List of (weight, feature_name, label) tuples
        """
        if not self.is_trained:
            return []

        features = []
        for (feature_name, label), weight in self.model.state_features_.items():
            features.append((weight, feature_name, label))

        features.sort(reverse=True)
        return features[:top_n]


if __name__ == "__main__":
    # Test CRF model
    print("Testing CRF Model")
    print("=" * 60)

    # Sample data
    train_tokens = [
        ['Pin', 'cực', 'trâu', ',', 'camera', 'đẹp'],
        ['Máy', 'lag', 'quá', ',', 'pin', 'tệ'],
    ]

    train_tags = [
        ['B-BATTERY#POSITIVE', 'I-BATTERY#POSITIVE', 'I-BATTERY#POSITIVE', 'O', 'B-CAMERA#POSITIVE', 'I-CAMERA#POSITIVE'],
        ['B-PERFORMANCE#NEGATIVE', 'I-PERFORMANCE#NEGATIVE', 'I-PERFORMANCE#NEGATIVE', 'O', 'B-BATTERY#NEGATIVE', 'I-BATTERY#NEGATIVE'],
    ]

    # Create and train model
    config = {'c1': 0.1, 'c2': 0.1, 'max_iterations': 50}
    model = CRFModel(config)

    results = model.train(train_tokens, train_tags)
    print(f"\nTraining results: {results}")

    # Predict
    test_tokens = [['Pin', 'rất', 'tốt']]
    predictions = model.predict(test_tokens)
    print(f"\nPredictions for {test_tokens[0]}: {predictions[0]}")

    # Show top features
    print("\nTop transitions:")
    for weight, from_label, to_label in model.get_top_transitions(5):
        print(f"  {weight:+.4f}: {from_label} -> {to_label}")

    print("\nTop features:")
    for weight, feature, label in model.get_top_features(5):
        print(f"  {weight:+.4f}: {feature} -> {label}")
