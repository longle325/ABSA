"""
Vietnamese ABSA Models

Available models:
- CRFModel: sklearn-crfsuite based CRF
- LogRegModel: Logistic Regression with TF-IDF
- SVMModel: SVM with TF-IDF
- PhoBERTModel: PhoBERT + BiLSTM-CRF
"""

from .base_model import BaseABSAModel
from .crf_model import CRFModel
from .logreg_model import LogRegModel
from .svm_model import SVMModel

# PhoBERT model requires extra dependencies
try:
    from .phobert_bilstm_crf import PhoBERTModel, PhoBERTBiLSTMCRF
except ImportError:
    PhoBERTModel = None
    PhoBERTBiLSTMCRF = None

__all__ = [
    'BaseABSAModel',
    'CRFModel',
    'LogRegModel',
    'SVMModel',
    'PhoBERTModel',
    'PhoBERTBiLSTMCRF'
]
