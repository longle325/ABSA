"""
Vietnamese ABSA Models

Available models:
- CRFModel: sklearn-crfsuite based CRF
"""

from .base_model import BaseABSAModel
from .crf_model import CRFModel


__all__ = [
    'BaseABSAModel',
    'CRFModel',
]
