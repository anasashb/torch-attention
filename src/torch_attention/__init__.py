from importlib.metadata import PackageNotFoundError, version

from torch_attention.shared import ScaledDotProductAttention

__all__ = ["ScaledDotProductAttention", "__version__"]

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"
