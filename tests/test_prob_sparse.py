import pytest
import torch

from adlers.time_series.prob_sparse import ProbSparseAttention
from tests._typing import MakeQKV


@pytest.mark.parametrize(
    (
        "is_causal",
        "output_attention_scores",
        "expected_output",
        "expected_weights",
    ),
    [
        pytest.param(
            False,
            False,
            torch.tensor(
                [
                    [
                        [
                            [4.0, 5.0],
                            [4.0, 5.0],
                            [4.2044735, 5.2044735],
                            [2.412292, 3.412292],
                        ]
                    ]
                ]
            ),
            None,
            id="non_causal_without_attention",
        ),
        pytest.param(
            True,
            True,
            torch.tensor(
                [
                    [
                        [
                            [1.0, 2.0],
                            [4.0, 6.0],
                            [3.5104697, 4.5104694],
                            [2.412292, 3.412292],
                        ]
                    ]
                ]
            ),
            torch.tensor(
                [
                    [
                        [
                            [0.25, 0.25, 0.25, 0.25],
                            [0.25, 0.25, 0.25, 0.25],
                            [0.24825509, 0.24825509, 0.50348985, 0.0],
                            [0.61451048, 0.07366338, 0.30299589, 0.00883027],
                        ]
                    ]
                ]
            ),
            id="causal_with_attention",
        ),
    ],
)
def test_prob_sparse_matches_pinned_informer_sparse_query_behavior(
    is_causal: bool,
    output_attention_scores: bool,
    expected_output: torch.Tensor,
    expected_weights: torch.Tensor | None,
) -> None:
    """Checks the pinned Informer outputs for sparse query selection."""
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 2.0]]]])
    value = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]])
    attention = ProbSparseAttention(
        is_causal=is_causal,
        factor=1,
        dropout_rate=0.0,
        output_attention_scores=output_attention_scores,
    )
    torch.manual_seed(seed=66)

    output, weights = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )

    torch.testing.assert_close(output, expected_output)
    if expected_weights is None:
        assert weights is None
    else:
        assert weights is not None
        torch.testing.assert_close(weights, expected_weights)


def test_prob_sparse_returns_weights_when_query_and_key_lengths_differ(
    make_qkv: MakeQKV,
) -> None:
    """Checks non-causal attention with different query and key lengths."""
    batch_size = 2
    num_heads = 4
    num_queries = 3
    num_keys = 5
    query, key, value = make_qkv(
        batch_size=batch_size,
        num_heads=num_heads,
        num_queries=num_queries,
        num_keys=num_keys,
    )
    attention = ProbSparseAttention(
        is_causal=False,
        factor=1,
        output_attention_scores=True,
    )

    output, weights = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )

    assert output.shape == (
        batch_size,
        num_heads,
        num_queries,
        query.shape[-1],
    )
    assert weights is not None
    assert weights.shape == (
        batch_size,
        num_heads,
        num_queries,
        num_keys,
    )


@pytest.mark.parametrize("is_causal", [False, True])
def test_prob_sparse_supports_single_position_sequences(
    is_causal: bool,
    make_qkv: MakeQKV,
) -> None:
    """Checks ProbSparse attention with one query and key position."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=3,
        num_queries=1,
        num_keys=1,
        head_dim=4,
    )
    attention = ProbSparseAttention(
        is_causal=is_causal,
        factor=1,
        output_attention_scores=True,
    )

    output, weights = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )

    assert weights is not None
    torch.testing.assert_close(output, value)
    torch.testing.assert_close(weights, torch.ones_like(weights))


def test_prob_sparse_uses_zero_custom_scale_factor() -> None:
    """Checks that zero overrides the default attention scale."""
    query = torch.tensor([[[[1.0], [2.0]]]])
    key = torch.tensor([[[[1.0], [2.0]]]])
    value = torch.tensor([[[[3.0], [5.0]]]])
    attention = ProbSparseAttention(
        is_causal=False,
        factor=2,
        custom_scale_factor=0.0,
        output_attention_scores=True,
    )

    output, weights = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )

    assert weights is not None
    torch.testing.assert_close(weights, torch.full_like(weights, 0.5))
    expected_output = value.mean(dim=-2, keepdim=True).expand_as(output)
    torch.testing.assert_close(output, expected_output)


def test_prob_sparse_rejects_different_causal_query_value_lengths() -> None:
    """Checks that causal attention requires equal query and value lengths."""
    query = torch.zeros(1, 1, 3, 2)
    key = torch.zeros(1, 1, 4, 2)
    value = torch.zeros(1, 1, 4, 2)
    attention = ProbSparseAttention(
        is_causal=True,
        factor=1,
        dropout_rate=0.0,
        output_attention_scores=False,
    )

    with pytest.raises(ValueError) as error:
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )

    assert str(error.value) == (
        "Causal ProbSparse attention requires query and value tensors to have "
        "the same sequence length; got query length 3 and value length 4."
    )


def test_prob_sparse_rejects_custom_attention_masks() -> None:
    """Checks that ProbSparse rejects unsupported custom attention masks."""
    query = torch.zeros(1, 1, 3, 2)
    key = torch.zeros(1, 1, 3, 2)
    value = torch.zeros(1, 1, 3, 2)
    attn_mask = torch.zeros(3, 3, dtype=torch.bool)
    attention = ProbSparseAttention(
        is_causal=False,
        factor=1,
        dropout_rate=0.0,
        output_attention_scores=False,
    )

    with pytest.raises(ValueError) as error:
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
        )

    assert str(error.value) == (
        "ProbSparse attention does not support custom attention masks; "
        "got shape (3, 3). Pass attn_mask=None."
    )


@pytest.mark.parametrize("factor", [0, -1])
def test_prob_sparse_rejects_non_positive_factors(factor: int) -> None:
    """Checks that the ProbSparse sampling factor is positive."""
    with pytest.raises(ValueError) as error:
        ProbSparseAttention(
            is_causal=False,
            factor=factor,
            dropout_rate=0.0,
            output_attention_scores=False,
        )

    assert str(error.value) == (
        f"ProbSparse factor must be greater than 0; got {factor}."
    )


@pytest.mark.parametrize(
    ("tensor_name", "tensor_index"),
    [
        ("query", 0),
        ("key", 1),
        ("value", 2),
    ],
)
def test_prob_sparse_rejects_non_four_dimensional_qkv_tensors(
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
    attention = ProbSparseAttention(
        is_causal=False,
        factor=1,
        dropout_rate=0.0,
        output_attention_scores=False,
        strict_mode=True,
    )

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


def test_prob_sparse_rejects_nonzero_dropout_rate() -> None:
    """Checks that unsupported ProbSparse dropout is rejected."""
    with pytest.raises(ValueError) as error:
        ProbSparseAttention(
            is_causal=False,
            factor=1,
            dropout_rate=0.1,
            output_attention_scores=False,
        )

    assert str(error.value) == (
        "ProbSparse attention does not support dropout; got dropout_rate 0.1. "
        "Set dropout_rate=0.0."
    )
