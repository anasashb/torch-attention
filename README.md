[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# torch-attention

A package collecting PyTorch-based attention mechanism implementations from
across deep learning research domains, including NLP, computer vision, and
time-series forecasting.

## Usage

```python
import torch

from torch_attention import ScaledDotProductAttention

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
