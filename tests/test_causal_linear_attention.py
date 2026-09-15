import pytest
import torch

from adlers.nlp.linear_attention import LinearAttention
from tests._typing import MakeQKV


def test_causal_linear_attention_is_causal_without_explicit_mask() -> None:
    """Checks causal behavior without an explicit triangular mask."""
    query = torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]).repeat(1, 3, 1, 1)
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]).repeat(1, 3, 1, 1)
    value = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]).repeat(1, 3, 1, 1)
    attention = LinearAttention(is_causal=True, eps=1e-6)

    output, attn_weights = attention(
        query=query,
        key=key,
        value=value,
    )

    expected_output = torch.tensor(
        [[[[0.99999976, 1.99999952], [2.22975826, 3.22975802]]]]
    ).repeat(1, 3, 1, 1)
    assert attn_weights is None
    torch.testing.assert_close(actual=output, expected=expected_output)
    assert output.is_contiguous()


@pytest.mark.parametrize("value_head_dim", [6, 4])
def test_causal_linear_attention_matches_explicit_attention_calculation(
    value_head_dim: int,
    make_qkv: MakeQKV,
) -> None:
    """Checks causal Linear Attention against an explicit score matrix."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=3,
        head_dim=6,
    )
    value = value[..., :value_head_dim]
    eps = 1e-6
    attention = LinearAttention(is_causal=True, eps=eps)

    output, attn_weights = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )

    mapped_query = torch.nn.functional.elu(input=query) + 1
    mapped_key = torch.nn.functional.elu(input=key) + 1
    scores = mapped_query @ mapped_key.transpose(dim0=-2, dim1=-1)
    causal_mask = torch.triu(
        torch.ones(
            size=(query.shape[-2], key.shape[-2]),
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    scores = scores.masked_fill(mask=causal_mask, value=0)
    weights = scores / (scores.sum(dim=-1, keepdim=True) + eps)
    expected_output = weights @ value

    assert attn_weights is None
    torch.testing.assert_close(actual=output, expected=expected_output)


def test_causal_linear_attention_applies_key_padding_mask() -> None:
    """Checks that padded keys do not contribute to causal attention."""
    query = torch.zeros((1, 1, 2, 1))
    key = torch.zeros((1, 1, 2, 1))
    value = torch.tensor([[[[2.0], [4.0]]]])
    attn_mask = torch.tensor([[[[False, True]]]])
    attention = LinearAttention(
        is_causal=True,
        feature_map=torch.ones_like,
        eps=0.0,
    )

    output, attn_weights = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
    )

    expected_output = torch.tensor([[[[2.0], [2.0]]]])
    assert attn_weights is None
    torch.testing.assert_close(actual=output, expected=expected_output)


def test_causal_linear_attention_rejects_query_dependent_attention_masks() -> (
    None
):
    """Checks that causal Linear Attention rejects query-dependent masks."""
    query = torch.zeros((1, 1, 2, 2))
    attn_mask = torch.zeros((2, 2), dtype=torch.bool)
    attention = LinearAttention(is_causal=True)

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


@pytest.mark.parametrize(
    ("tensor_name", "tensor_index"),
    [
        ("query", 0),
        ("key", 1),
        ("value", 2),
    ],
)
def test_causal_linear_attention_rejects_non_four_dimensional_qkv_tensors(
    tensor_name: str,
    tensor_index: int,
    make_qkv: MakeQKV,
) -> None:
    """Checks that query, key, and value tensors include all four axes."""
    tensors = list(
        make_qkv(
            batch_size=2,
            num_heads=1,
            num_queries=3,
            num_keys=3,
            head_dim=6,
        )
    )
    tensors[tensor_index] = tensors[tensor_index].squeeze(dim=1)
    query, key, value = tensors
    attention = LinearAttention(is_causal=True)

    with pytest.raises(ValueError) as error:
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )

    assert str(error.value) == (
        f"{tensor_name.capitalize()} tensor must be 4D "
        "[batch_size, num_heads, sequence_length, head_dim]; "
        f"got shape {tuple(tensors[tensor_index].shape)}."
    )


@pytest.mark.parametrize(
    ("dimension", "expected_error"),
    [
        pytest.param(
            0,
            "Query, key, and value batch sizes must match; "
            "got query batch size 1, key batch size 2, and "
            "value batch size 2. Use the same batch size for all "
            "three tensors.",
            id="batch-size",
        ),
        pytest.param(
            1,
            "Query, key, and value head counts must match; "
            "got query head count 1, key head count 4, and "
            "value head count 4. Use the same number of heads for "
            "all three tensors.",
            id="head-count",
        ),
    ],
)
def test_causal_linear_attention_rejects_query_batch_and_head_mismatches(
    dimension: int,
    expected_error: str,
    make_qkv: MakeQKV,
) -> None:
    """Checks that query, key, and value batches and head counts match."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=3,
        head_dim=6,
    )
    query = query.narrow(
        dim=dimension,
        start=0,
        length=1,
    )
    attention = LinearAttention(is_causal=True)

    with pytest.raises(ValueError) as error:
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )

    assert str(error.value) == expected_error


def test_causal_linear_attention_rejects_mismatched_query_and_key_dimensions(
    make_qkv: MakeQKV,
) -> None:
    """Checks that query and key head dimensions must match."""
    query, _, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=3,
        head_dim=6,
    )
    key = torch.zeros((2, 4, 3, 5))
    attention = LinearAttention(is_causal=True)

    with pytest.raises(ValueError) as error:
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )

    assert str(error.value) == (
        "Query and key head dimensions must match; "
        "got query head dimension 6 and key head dimension 5. "
        "Use the same head dimension for both tensors."
    )


def test_causal_linear_attention_rejects_unequal_key_and_value_lengths(
    make_qkv: MakeQKV,
) -> None:
    """Checks that each key position must have a corresponding value."""
    query, key, _ = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=3,
        head_dim=6,
    )
    value = torch.zeros((2, 4, 4, 6))
    attention = LinearAttention(is_causal=True)

    with pytest.raises(ValueError) as error:
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )

    assert str(error.value) == (
        "Key and value sequence lengths must match; "
        "got key length 3 and value length 4. "
        "Provide one value position for each key position."
    )


def test_causal_linear_attention_rejects_different_query_and_key_lengths(
    make_qkv: MakeQKV,
) -> None:
    """Checks that causal query, key, and value lengths must match."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=5,
        head_dim=6,
    )
    attention = LinearAttention(is_causal=True)

    with pytest.raises(ValueError) as error:
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )

    assert str(error.value) == (
        "Query, key, and value sequence lengths must match for causal "
        "linear attention; got query length 3, key length 5, and "
        "value length 5. Use the same sequence length for all three tensors."
    )
