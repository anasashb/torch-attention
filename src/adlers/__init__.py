from importlib.metadata import PackageNotFoundError, version

from adlers.nlp import LinearAttention
from adlers.shared import ScaledDotProductAttention
from adlers.time_series import ProbSparseAttention

__all__ = [
    "LinearAttention",
    "ProbSparseAttention",
    "ScaledDotProductAttention",
    "__version__",
]

try:
    __version__ = version("adlers")
except PackageNotFoundError:
    __version__ = "0.0.0"
