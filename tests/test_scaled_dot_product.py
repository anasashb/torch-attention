import pytest
import torch

from torch_attention.shared import ScaledDotProductAttention


@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("output_attention_scores", [False, True])
def test_scaled_dot_product(
    use_mask: bool, output_attention_scores: bool
) -> None:
    torch.manual_seed(66)

    batch_size = 32
    num_heads = 4
    num_queries = 96
    num_keys = 96
    num_values = 96
    head_dim = 16

    Q = torch.randn(batch_size, num_heads, num_queries, head_dim)
    K = torch.randn(batch_size, num_heads, num_keys, head_dim)
    V = torch.randn(batch_size, num_heads, num_values, head_dim)

    attention = ScaledDotProductAttention(
        use_mask=use_mask,
        dropout_rate=0.0,
        output_attention_scores=output_attention_scores,
        strict_mode=True,
        custom_scale_factor=None,
    )

    if use_mask:
        # we let it generate the triangular mask on its own
        out, weights = attention(query=Q, key=K, value=V, mask=None)
        assert torch.isfinite(out).all()  # in case sth goes wrong with mask
    else:
        out, weights = attention(query=Q, key=K, value=V, mask=None)

    assert out.shape == (batch_size, num_heads, num_queries, head_dim)

    if output_attention_scores:
        assert weights.shape == (batch_size, num_heads, num_queries, num_keys)

        summed_weights = weights.sum(dim=-1)
        assert torch.allclose(
            summed_weights, torch.ones_like(summed_weights), atol=1e-5
        )
