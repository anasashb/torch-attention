import pytest
import torch
import torch.nn.functional as F

from tests._typing import MakeQKV
from torch_attention.shared import ScaledDotProductAttention


@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("output_attention_scores", [False, True])
def test_scaled_dot_product(
    use_mask: bool,
    output_attention_scores: bool,
    make_qkv: MakeQKV,
) -> None:
    """Checks default einsum attention shapes and optional attention scores."""
    batch_size = 32
    num_heads = 4
    num_queries = 96
    num_keys = 96
    head_dim = 16

    query, key, value = make_qkv(
        batch_size=batch_size,
        num_heads=num_heads,
        num_queries=num_queries,
        num_keys=num_keys,
        head_dim=head_dim,
    )

    attention = ScaledDotProductAttention(
        use_mask=use_mask,
        dropout_rate=0.0,
        backend="einsum",
        output_attention_scores=output_attention_scores,
        strict_mode=True,
        custom_scale_factor=None,
    )

    if use_mask:
        # let it generate the triangular mask on its own
        out, weights = attention(query=query, key=key, value=value, mask=None)
        assert torch.isfinite(out).all()
    else:
        out, weights = attention(query=query, key=key, value=value, mask=None)

    assert out.shape == (batch_size, num_heads, num_queries, head_dim)

    if output_attention_scores:
        assert weights.shape == (batch_size, num_heads, num_queries, num_keys)

        summed_weights = weights.sum(dim=-1)
        assert torch.allclose(
            summed_weights, torch.ones_like(summed_weights), atol=1e-5
        )


@pytest.mark.parametrize("use_mask", [False, True])
def test_sdpa_backend_matches_torch_scaled_dot_product_attention(
    use_mask: bool,
    make_qkv: MakeQKV,
) -> None:
    """Checks that the SDPA backend matches PyTorch's native implementation."""
    query, key, value = make_qkv()

    attention = ScaledDotProductAttention(
        use_mask=use_mask,
        dropout_rate=0.0,
        backend="sdpa",
        output_attention_scores=False,
        strict_mode=True,
        custom_scale_factor=None,
    )

    out, weights = attention(query=query, key=key, value=value, mask=None)
    expected_out = F.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=use_mask,
    )

    assert weights is None
    torch.testing.assert_close(out, expected_out)


def test_sdpa_backend_matches_einsum_backend_with_explicit_mask(
    make_qkv: MakeQKV,
) -> None:
    """Checks that explicit masks have matching semantics across backends."""
    query, key, value = make_qkv(batch_size=2, num_heads=3)
    num_queries = query.shape[-2]
    num_keys = key.shape[-2]
    mask = torch.zeros(num_queries, num_keys, dtype=torch.bool)
    mask[:, -1] = True

    einsum_attention = ScaledDotProductAttention(
        use_mask=True,
        dropout_rate=0.0,
        backend="einsum",
        output_attention_scores=True,
        strict_mode=True,
        custom_scale_factor=None,
    )
    sdpa_attention = ScaledDotProductAttention(
        use_mask=True,
        dropout_rate=0.0,
        backend="sdpa",
        output_attention_scores=False,
        strict_mode=True,
        custom_scale_factor=None,
    )

    expected_out, _ = einsum_attention(
        query=query,
        key=key,
        value=value,
        mask=mask,
    )
    out, weights = sdpa_attention(
        query=query,
        key=key,
        value=value,
        mask=mask,
    )

    assert weights is None
    torch.testing.assert_close(out, expected_out)


def test_sdpa_backend_rejects_attention_scores() -> None:
    """Checks that SDPA rejects unsupported attention score output."""
    with pytest.raises(ValueError, match="does not support"):
        ScaledDotProductAttention(
            backend="sdpa",
            output_attention_scores=True,
        )


def test_scaled_dot_product_rejects_invalid_backend() -> None:
    """Checks that unknown attention backends fail fast."""
    with pytest.raises(ValueError, match="Invalid backend"):
        ScaledDotProductAttention(backend="supermaxx")
