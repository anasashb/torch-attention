from importlib.metadata import PackageNotFoundError, version

from adlers.shared import ScaledDotProductAttention

__all__ = ["ScaledDotProductAttention", "__version__"]

try:
    __version__ = version("adlers")
except PackageNotFoundError:
    __version__ = "0.0.0"
