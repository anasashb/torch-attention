[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# ADLERS

**Attention for Deep Learning: Efficient, Readable & Standardized**

Unified attention mechanisms for PyTorch across NLP, vision, and time series.

## Usage

```python
import torch

from adlers import ScaledDotProductAttention

query = torch.randn(2, 4, 16, 32)
key = torch.randn(2, 4, 16, 32)
value = torch.randn(2, 4, 16, 32)

attention = ScaledDotProductAttention(backend="sdpa")
output, weights = attention(query=query, key=key, value=value)

print(output.shape)
print(weights)
```

Use `backend="einsum"` when attention weights are needed. Use `backend="sdpa"`
to delegate to PyTorch's optimized scaled dot-product attention implementation
when weights are not needed.

## Project status

ADLERS is in a very early stage. I'm just trying to build the attention mechanism
library I wish existed while I was working on my thesis, one that had readable
implementations, consistent APIs, and readily available benchmarks with optimized
PyTorch kernels.
