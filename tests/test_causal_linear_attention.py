import torch

from adlers.nlp.causal_linear_attention import CausalLinearAttention


def test_causal_linear_attention_is_causal_without_explicit_mask() -> None:
    """Checks causal behavior without an explicit triangular mask."""
    query = torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]).repeat(1, 3, 1, 1)
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]).repeat(1, 3, 1, 1)
    value = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]).repeat(1, 3, 1, 1)
    attention = CausalLinearAttention(eps=1e-6)

    output = attention(
        query=query,
        key=key,
        value=value,
    )

    expected_output = torch.tensor(
        [[[[0.99999976, 1.99999952], [2.22975826, 3.22975802]]]]
    ).repeat(1, 3, 1, 1)
    torch.testing.assert_close(actual=output, expected=expected_output)
    assert output.is_contiguous()
