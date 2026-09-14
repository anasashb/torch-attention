import pytest
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


def test_causal_linear_attention_applies_key_padding_mask() -> None:
    """Checks that padded keys do not contribute to causal attention."""
    query = torch.zeros((1, 1, 2, 1))
    key = torch.zeros((1, 1, 2, 1))
    value = torch.tensor([[[[2.0], [4.0]]]])
    attn_mask = torch.tensor([[[[False, True]]]])
    attention = CausalLinearAttention(
        feature_map=torch.ones_like,
        eps=0.0,
    )

    output = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
    )

    expected_output = torch.tensor([[[[2.0], [2.0]]]])
    torch.testing.assert_close(actual=output, expected=expected_output)


def test_causal_linear_attention_rejects_non_boolean_key_padding_masks() -> (
    None
):
    """Checks that causal Linear Attention requires boolean masks."""
    query = torch.zeros((1, 1, 2, 2))
    attn_mask = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
    attention = CausalLinearAttention()

    with pytest.raises(TypeError) as error:
        attention(
            query=query,
            key=query,
            value=query,
            attn_mask=attn_mask,
        )

    assert str(error.value) == (
        "Only boolean attention masks are supported; "
        "got mask dtype torch.float32. Use a torch.bool mask "
        "with True for positions that should be masked out and "
        "False for positions that can be attended to."
    )


def test_causal_linear_attention_rejects_query_dependent_attention_masks() -> (
    None
):
    """Checks that causal Linear Attention rejects query-dependent masks."""
    query = torch.zeros((1, 1, 2, 2))
    attn_mask = torch.zeros((2, 2), dtype=torch.bool)
    attention = CausalLinearAttention()

    with pytest.raises(ValueError) as error:
        attention(
            query=query,
            key=query,
            value=query,
            attn_mask=attn_mask,
        )

    assert str(error.value) == (
        "Linear attention only supports key-padding masks shaped "
        "[batch_size, 1, 1, num_keys]; got shape (2, 2)."
    )
