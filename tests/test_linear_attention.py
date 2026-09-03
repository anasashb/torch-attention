from types import SimpleNamespace

import torch

from adlers.nlp.linear_attention import LinearAttention


def test_linear_attention_matches_pinned_fast_transformers_behavior() -> None:
    """Checks the pinned fast-transformers unmasked attention output."""
    query = torch.tensor([[[[1.0, -1.0]], [[-1.0, 1.0]]]]).repeat(1, 1, 3, 1)
    key = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]]).repeat(1, 1, 3, 1)
    value = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]]).repeat(1, 1, 3, 1)
    attn_mask = SimpleNamespace(all_ones=True)
    key_lengths = SimpleNamespace(float_matrix=torch.ones((1, 2)))
    attention = LinearAttention(query_dimensions=2, eps=1e-6)

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
                    [1.77024138, 2.77024126],
                    [1.77024138, 2.77024126],
                    [1.77024138, 2.77024126],
                ],
                [
                    [2.22975802, 3.22975779],
                    [2.22975802, 3.22975779],
                    [2.22975802, 3.22975779],
                ],
            ]
        ]
    )
    torch.testing.assert_close(output, expected_output)
