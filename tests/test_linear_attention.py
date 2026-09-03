import pytest
import torch

from adlers.nlp.linear_attention import LinearAttention


def test_linear_attention_matches_pinned_fast_transformers_behavior() -> None:
    """Checks the pinned fast-transformers unmasked attention output."""
    query = torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]).repeat(1, 3, 1, 1)
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]).repeat(1, 3, 1, 1)
    value = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]).repeat(1, 3, 1, 1)
    attention = LinearAttention(eps=1e-6)

    output, attn_weights = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )

    expected_output = torch.tensor(
        [[[[1.77024138, 2.77024126], [2.22975802, 3.22975779]]]]
    ).repeat(1, 3, 1, 1)
    assert attn_weights is None
    torch.testing.assert_close(output, expected_output)


def test_linear_attention_accepts_tensor_feature_map() -> None:
    """Checks that feature maps operate directly on tensors."""
    query = torch.zeros((1, 1, 2, 1))
    key = torch.zeros((1, 1, 2, 1))
    value = torch.tensor([[[[2.0], [4.0]]]])
    attention = LinearAttention(feature_map=torch.ones_like, eps=0.0)

    output, _ = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )

    torch.testing.assert_close(output, torch.full_like(value, 3.0))


def test_linear_attention_applies_key_padding_mask() -> None:
    """Checks that padded keys do not contribute to attention."""
    query = torch.zeros((1, 1, 1, 1))
    key = torch.zeros((1, 1, 2, 1))
    value = torch.tensor([[[[2.0], [4.0]]]])
    attn_mask = torch.tensor([[[[False, True]]]])
    attention = LinearAttention(feature_map=torch.ones_like, eps=0.0)

    output, _ = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
    )

    torch.testing.assert_close(output, torch.tensor([[[[2.0]]]]))


def test_linear_attention_rejects_query_dependent_attention_masks() -> None:
    """Checks that LinearAttention rejects query-dependent masks."""
    query = torch.zeros((1, 1, 2, 2))
    attn_mask = torch.zeros((2, 2), dtype=torch.bool)
    attention = LinearAttention()

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


def test_linear_attention_rejects_attention_scores() -> None:
    """Checks that LinearAttention rejects attention score output."""
    with pytest.raises(ValueError) as error:
        LinearAttention(output_attention_scores=True)

    assert str(error.value) == (
        "Linear attention does not support returning attention scores. "
        "Set output_attention_scores=False."
    )
