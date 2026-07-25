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

## Supported Tensor and mask shapes

`ScaledDotProductAttention` expects query, key, and value tensors with the
attention heads already split:

```text
query: [batch_size, num_heads, num_queries, head_dim]
key:   [batch_size, num_heads, num_keys, head_dim]
value: [batch_size, num_heads, num_keys, head_dim]
```

The query and key lengths can differ, so the same module works for both
self-attention and cross-attention. But, ey and value lengths must match. Batch
size, head count, and head dimension must match across all three tensors.

For now, attention masks must use `torch.bool`. `True` marks a position that
should be masked out, while `False` marks a position that can be attended to.
The supported mask shapes are:

- `[num_queries, num_keys]`
- `[batch_size, num_queries, num_keys]`
- `[batch_size, 1, num_queries, num_keys]`
- `[batch_size, num_heads, num_queries, num_keys]`

The smaller forms broadcast across batches or heads. When `is_causal=True`, a
supplied mask is applied together with the causal mask. If every key is masked
for a query, its output is zero. The `einsum` backend also returns zero
attention weights for that query.

## Project status

ADLERS is in a very early stage. I'm just trying to build the attention mechanism
library I wish existed while I was working on my thesis, one that had readable
implementations, consistent APIs, and readily available benchmarks with optimized
PyTorch kernels.

*Notes*:

- A `MultiHeadAttention`-style class is not yet implemented for this project, and
will probably be added once we have a decent collection of attention mechanisms.
- I'll add support for float masks later on.
