"""
Configuration Loader for Vietnamese ABSA

Loads and validates YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config or {}


def load_data_config(config_path: str = "configs/data_config.yaml") -> Dict[str, Any]:
    """
    Load data configuration with defaults.

    Args:
        config_path: Path to data config file

    Returns:
        Data configuration dictionary
    """
    defaults = {
        'data': {
            'path': 'cleaned_data/cleaned_data.jsonl'
        },
        'split': {
            'train_size': 0.8,
            'test_size': 0.2,
            'dev_size': 0.0,
            'random_seed': 42,
            'shuffle': True,
            'stratify': False
        },
        'preprocessing': {
            'use_simple_tokenizer': False,
            'max_seq_length': 256,
            'lowercase': False
        },
        'output': {
            'model_dir': 'outputs/models',
            'log_dir': 'outputs/logs',
            'predictions_dir': 'outputs/predictions',
            'save_predictions': True
        }
    }

    try:
        config = load_config(config_path)
        return merge_configs(defaults, config)
    except FileNotFoundError:
        print(f"Warning: Config file not found, using defaults: {config_path}")
        return defaults


def load_model_config(
    model_name: str,
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load model-specific configuration.

    Args:
        model_name: Model name (crf, logreg, svm, phobert)
        config_path: Optional custom config path

    Returns:
        Model configuration dictionary
    """
    if config_path is None:
        config_path = f"configs/{model_name}_config.yaml"

    defaults = get_model_defaults(model_name)

    try:
        config = load_config(config_path)
        return merge_configs(defaults, config)
    except FileNotFoundError:
        print(f"Warning: Config file not found, using defaults: {config_path}")
        return defaults


def get_model_defaults(model_name: str) -> Dict[str, Any]:
    """Get default configuration for a model."""

    defaults = {
        'crf': {
            'model': {'name': 'crf'},
            'hyperparameters': {
                'algorithm': 'lbfgs',
                'c1': 0.1,
                'c2': 0.1,
                'max_iterations': 200,
                'all_possible_transitions': True
            },
            'features': {
                'context_window': 2,
                'use_word_shape': True,
                'use_ngrams': True
            },
            'training': {
                'verbose': True
            }
        },
        'logreg': {
            'model': {'name': 'logreg'},
            'hyperparameters': {
                'C': 1.0,
                'max_iter': 1000,
                'solver': 'lbfgs',
                'multi_class': 'multinomial',
                'random_state': 42
            },
            'features': {
                'max_features': 10000,
                'ngram_range': [1, 2],
                'context_window': 2,
                'sublinear_tf': True
            },
            'training': {
                'n_jobs': -1,
                'verbose': 1
            }
        },
        'svm': {
            'model': {'name': 'svm'},
            'hyperparameters': {
                'kernel': 'linear',
                'C': 1.0,
                'gamma': 'scale',
                'random_state': 42
            },
            'features': {
                'max_features': 15000,
                'ngram_range': [1, 3],
                'context_window': 3,
                'sublinear_tf': True
            },
            'training': {
                'verbose': True,
                'cache_size': 500
            }
        },
        'phobert': {
            'model': {
                'name': 'phobert-bilstm-crf',
                'pretrained': 'vinai/phobert-base'
            },
            'architecture': {
                'lstm_hidden': 256,
                'lstm_layers': 2,
                'dropout': 0.1,
                'lstm_dropout': 0.3,
                'freeze_bert': False
            },
            'training': {
                'epochs': 20,
                'batch_size': 16,
                'gradient_clip': 1.0,
                'warmup_ratio': 0.1
            },
            'optimizer': {
                'bert_lr': 2e-5,
                'lstm_lr': 1e-3,
                'crf_lr': 1e-2,
                'weight_decay': 0.01
            }
        },
        'bilstm_crf': {
            # Paper hyperparameters (PACLIC 2021)
            'model': {'name': 'bilstm-crf'},
            'embeddings': {
                'syllable_embed_dim': 100,   # Paper: 100
                'char_embed_dim': 100,       # Paper: 100
                'char_num_filters': 100,     # Paper: 100
                'max_word_len': 20
            },
            'architecture': {
                'lstm_hidden': 400,          # Paper: 400
                'lstm_layers': 2,
                'dropout': 0.33              # Paper: 0.33
            },
            'training': {
                'epochs': 30,                # Paper: 30
                'batch_size': 32,
                'learning_rate': 0.001
            }
        },
        'bilstm_crf_xlmr': {
            # Paper hyperparameters (PACLIC 2021)
            'model': {
                'name': 'bilstm-crf-xlmr',
                'xlmr_model_name': 'xlm-roberta-base'
            },
            'embeddings': {
                'syllable_embed_dim': 100,   # Paper: 100
                'char_embed_dim': 100,       # Paper: 100
                'char_num_filters': 100,     # Paper: 100
                'max_word_len': 20
            },
            'architecture': {
                'lstm_hidden': 400,          # Paper: 400
                'lstm_layers': 2,
                'dropout': 0.33,             # Paper: 0.33
                'freeze_xlmr': True
            },
            'training': {
                'epochs': 30,                # Paper: 30
                'batch_size': 16,
                'learning_rate': 0.001,
                'xlmr_lr': 2e-5,
                'max_seq_len': 256
            }
        }
    }

    return defaults.get(model_name, {})


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two config dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def flatten_config(config: Dict, prefix: str = '') -> Dict[str, Any]:
    """
    Flatten nested config to dot-notation keys.

    Args:
        config: Nested configuration
        prefix: Key prefix

    Returns:
        Flattened configuration
    """
    items = {}
    for key, value in config.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_config(value, new_key))
        else:
            items[new_key] = value
    return items


if __name__ == "__main__":
    # Test config loading
    print("Testing Config Loader")
    print("=" * 50)

    # Test data config
    data_config = load_data_config()
    print("\nData Config:")
    print(f"  Train size: {data_config['split']['train_size']}")
    print(f"  Test size: {data_config['split']['test_size']}")
    print(f"  Random seed: {data_config['split']['random_seed']}")

    # Test model configs
    for model_name in ['crf', 'logreg', 'svm', 'phobert', 'bilstm_crf', 'bilstm_crf_xlmr']:
        config = load_model_config(model_name)
        print(f"\n{model_name.upper()} Config:")
        flat = flatten_config(config)
        for k, v in list(flat.items())[:5]:
            print(f"  {k}: {v}")
