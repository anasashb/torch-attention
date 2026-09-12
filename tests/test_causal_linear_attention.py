from types import SimpleNamespace

import torch

from adlers.nlp.causal_linear_attention import CausalLinearAttention


def test_causal_linear_attention_uses_bhld_layout() -> None:
    """Checks the pinned causal attention output using the BHLD layout."""
    query = torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]).repeat(1, 3, 1, 1)
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]).repeat(1, 3, 1, 1)
    value = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]).repeat(1, 3, 1, 1)
    attn_mask = SimpleNamespace(lower_triangular=True)
    key_lengths = SimpleNamespace(float_matrix=torch.ones((1, 2)))
    attention = CausalLinearAttention(eps=1e-6)

    output = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        query_lengths=None,
        key_lengths=key_lengths,
    )

    expected_output = torch.tensor(
        [[[[0.99999976, 1.99999952], [2.22975826, 3.22975802]]]]
    ).repeat(1, 3, 1, 1)
    torch.testing.assert_close(actual=output, expected=expected_output)
    assert output.is_contiguous()
