"""
Base Model Interface for Vietnamese ABSA

Provides abstract base class that all ABSA models must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pickle
from pathlib import Path


class BaseABSAModel(ABC):
    """Abstract base class for all ABSA models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model.

        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.is_trained = False

    @abstractmethod
    def train(
        self,
        train_tokens: List[List[str]],
        train_tags: List[List[str]],
        dev_tokens: Optional[List[List[str]]] = None,
        dev_tags: Optional[List[List[str]]] = None
    ) -> Dict[str, float]:
        """
        Train the model.

        Args:
            train_tokens: List of token sequences for training
            train_tags: List of BIO tag sequences for training
            dev_tokens: Optional development token sequences
            dev_tags: Optional development tag sequences

        Returns:
            Dictionary with training metrics
        """
        pass

    @abstractmethod
    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        """
        Predict BIO tags for token sequences.

        Args:
            tokens: List of token sequences

        Returns:
            List of predicted BIO tag sequences
        """
        pass

    def predict_single(self, tokens: List[str]) -> List[str]:
        """
        Predict BIO tags for a single token sequence.

        Args:
            tokens: Token sequence

        Returns:
            Predicted BIO tag sequence
        """
        return self.predict([tokens])[0]

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'BaseABSAModel':
        """
        Load model from disk.

        Args:
            path: Path to load model from

        Returns:
            Loaded model instance
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def evaluate(
        self,
        tokens: List[List[str]],
        tags: List[List[str]]
    ) -> Dict[str, float]:
        """
        Evaluate model on given data.

        Args:
            tokens: Token sequences
            tags: Ground truth BIO tag sequences

        Returns:
            Evaluation metrics dictionary
        """
        from ..evaluation.metrics import evaluate_sequence_labeling

        predictions = self.predict(tokens)
        return evaluate_sequence_labeling(tags, predictions)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
