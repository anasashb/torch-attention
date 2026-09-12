from types import SimpleNamespace

import torch

from adlers.nlp.causal_linear_attention import CausalLinearAttention


def test_causal_linear_attention_matches_pinned_fast_transformers_behavior() -> (
    None
):
    """Checks the pinned fast-transformers causal attention output."""
    query = torch.tensor([[[[1.0, -1.0]], [[-1.0, 1.0]]]]).repeat(1, 1, 3, 1)
    key = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]]).repeat(1, 1, 3, 1)
    value = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]).repeat(1, 1, 3, 1)
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
        [
            [
                [
                    [0.99999976, 1.99999952],
                    [0.99999976, 1.99999952],
                    [0.99999976, 1.99999952],
                ],
                [
                    [2.22975826, 3.22975802],
                    [2.22975826, 3.22975802],
                    [2.22975826, 3.22975802],
                ],
            ]
        ]
    )
    torch.testing.assert_close(actual=output, expected=expected_output)
    assert output.is_contiguous()
