import pytest
import torch

from tests._typing import MakeQKV, QKVTensors


@pytest.fixture
def make_qkv() -> MakeQKV:
    """Returns a factory for deterministic query, key, value tensors."""

    def _make_qkv(
        batch_size: int = 4,
        num_heads: int = 2,
        num_queries: int = 8,
        num_keys: int = 8,
        head_dim: int = 6,
    ) -> QKVTensors:
        generator = torch.Generator().manual_seed(66)
        query = torch.randn(
            batch_size, num_heads, num_queries, head_dim, generator=generator
        )
        key = torch.randn(
            batch_size, num_heads, num_keys, head_dim, generator=generator
        )
        value = torch.randn(
            batch_size, num_heads, num_keys, head_dim, generator=generator
        )
        return query, key, value

    return _make_qkv
