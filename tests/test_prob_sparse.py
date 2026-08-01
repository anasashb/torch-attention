import pytest
import torch

from adlers.time_series.prob_sparse import ProbAttention


@pytest.mark.parametrize(
    (
        "mask_flag",
        "output_attention",
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
                        [[4.0, 5.0]],
                        [[4.0, 5.0]],
                        [[4.2044735, 5.2044735]],
                        [[2.412292, 3.412292]],
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
                        [[1.0, 2.0]],
                        [[4.0, 6.0]],
                        [[3.5104697, 4.5104694]],
                        [[2.412292, 3.412292]],
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
    mask_flag: bool,
    output_attention: bool,
    expected_output: torch.Tensor,
    expected_weights: torch.Tensor | None,
) -> None:
    """Checks the pinned Informer outputs for sparse query selection."""
    queries = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[2.0, -1.0]]]]
    )
    keys = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[-1.0, 2.0]]]]
    )
    values = torch.tensor(
        [[[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]], [[7.0, 8.0]]]]
    )
    attention = ProbAttention(
        mask_flag=mask_flag,
        factor=1,
        attention_dropout=0.0,
        output_attention=output_attention,
    )
    torch.manual_seed(seed=66)

    output, weights = attention(
        queries=queries,
        keys=keys,
        values=values,
        attn_mask=None,
    )

    torch.testing.assert_close(output, expected_output)
    if expected_weights is None:
        assert weights is None
    else:
        assert weights is not None
        torch.testing.assert_close(weights, expected_weights)
