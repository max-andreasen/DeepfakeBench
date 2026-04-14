from .transformer import Transformer
from .linear_cls import LinearClassifier
from .bigru import BiGRU

# Registry used by training/train.py and evaluation/test.py so both sides
# instantiate the exact same class for a given model_type string.
MODELS = {
    'transformer': Transformer,
    'linear': LinearClassifier,
    'bigru': BiGRU,
}

__all__ = ['Transformer', 'LinearClassifier', 'BiGRU', 'MODELS']
