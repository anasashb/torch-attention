import pytest
import torch

from adlers.time_series.prob_sparse import ProbSparseAttention


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
