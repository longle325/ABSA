"""
Model Manager for Vietnamese ABSA Demo
======================================
Manages loading, caching, and switching between trained models.

Supports:
- CRF models (Conditional Random Fields)
- PhoBERT-CRF models (Vietnamese BERT + CRF)
- BiLSTM-CRF-XLMR models (with XLM-RoBERTa)

Features:
- Singleton pattern for efficient memory usage
- Lazy loading (load on demand)
- Model caching (keep loaded models in memory)
- Auto-detection of model type from file name
"""

import sys
from pathlib import Path
from typing import Dict, Optional, List
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.crf_model import CRFModel
from src.models.phobert_crf import PhoBERTCRFModel
from src.models.bilstm_crf_xlmr import BiLSTMCRFXLMRModel


class ModelManager:
    """Singleton manager for ABSA models."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Model storage
        self._models: Dict[str, object] = {}
        self._current_model_name: Optional[str] = None

        # Model directory
        self.models_dir = Path(__file__).parent.parent / "outputs" / "models"

        self._initialized = True

    def get_available_models(self) -> List[Dict[str, str]]:
        """
        Scan the models directory and return available models.

        Returns:
            List of dicts with keys: name, path, type, size_mb
        """
        if not self.models_dir.exists():
            return []

        models = []
        # Add CRF model
        models.append({
                'name': 'CRF',
                'path': str(self.models_dir / 'crf_model.pkl'),
                'type': 'CRF'
            })

        # Add PhoBERT-CRF model
        models.append({
                'name': 'PhoBERT-CRF',
                'path': str(self.models_dir / 'phobert_crf_model.pkl'),
                'type': 'PhoBERT-CRF'
            })

        # Add BiLSTM-CRF-XLMR model
        models.append({
                'name': 'BiLSTM-CRF-XLMR',
                'path': str(self.models_dir / 'bilstm_crf_xlmr_model.pkl'),
                'type': 'BiLSTM-CRF-XLMR'
            })
        return models

    def _detect_model_type(self, filename: str) -> str:
        """Detect model type from filename."""
        filename_lower = filename.lower()

        # Check for XLM-R first (more specific)
        if 'xlmr' in filename_lower or 'xlm' in filename_lower:
            return 'BiLSTM-CRF-XLMR'
        # Check for PhoBERT-CRF
        elif 'phobert_crf' in filename_lower:
            return 'PhoBERT-CRF'
        # Check for pure CRF model (crf_model.pkl)
        elif filename_lower.startswith('crf_model') or filename_lower == 'crf.pkl':
            return 'CRF'
        else:
            return 'Unknown'

    def load_model(self, model_name: str, force_reload: bool = False) -> object:
        """
        Load a model by name.

        Args:
            model_name: Name of the model file (e.g., "bilstm_crf_model.pkl")
            force_reload: If True, reload even if already cached

        Returns:
            Loaded model instance

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        # Check cache
        if not force_reload and model_name in self._models:
            print(f"Using cached model: {model_name}")
            self._current_model_name = model_name
            return self._models[model_name]

        # Get model path
        model_path = self.models_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Detect model type
        model_type = self._detect_model_type(model_name)

        print(f"Loading {model_type} model: {model_name}...")

        # Load appropriate model
        try:
            if model_type == 'CRF':
                # CRF uses classmethod load()
                model = CRFModel.load(str(model_path))
            elif model_type == 'PhoBERT-CRF':
                # PhoBERT-CRF uses instance method load()
                model = PhoBERTCRFModel()
                model.load(str(model_path))
            elif model_type == 'BiLSTM-CRF-XLMR':
                # BiLSTM-CRF-XLMR uses instance method load()
                model = BiLSTMCRFXLMRModel()
                model.load(str(model_path))
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Cache the model
            self._models[model_name] = model
            self._current_model_name = model_name

            print(f"Model loaded successfully: {model_name}")
            return model

        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

    def get_current_model(self) -> Optional[object]:
        """Get the currently loaded model."""
        if self._current_model_name is None:
            return None
        return self._models.get(self._current_model_name)

    def get_current_model_name(self) -> Optional[str]:
        """Get the name of the currently loaded model."""
        return self._current_model_name

    def clear_cache(self):
        """Clear all cached models from memory."""
        self._models.clear()
        self._current_model_name = None

        # Force garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Model cache cleared")

    def unload_model(self, model_name: str):
        """Unload a specific model from cache."""
        if model_name in self._models:
            del self._models[model_name]
            if self._current_model_name == model_name:
                self._current_model_name = None
            print(f"Model unloaded: {model_name}")


# Global singleton instance
_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Get the global ModelManager instance."""
    return _manager


if __name__ == "__main__":
    # Test the model manager
    print("Testing Model Manager")
    print("=" * 60)

    manager = get_model_manager()

    # List available models
    models = manager.get_available_models()
    print(f"\nFound {len(models)} models:")
    for model in models:
        print(f"  - {model['name']}")
        print(f"    Type: {model['type']}")
        print(f"    Size: {model['size_mb']}")

    if models:
        # Test loading the first model
        first_model = models[0]['name']
        print(f"\nLoading model: {first_model}")
        try:
            model = manager.load_model(first_model)
            print(f"Model loaded successfully!")
            print(f"Current model: {manager.get_current_model_name()}")
        except Exception as e:
            print(f"Error loading model: {e}")
