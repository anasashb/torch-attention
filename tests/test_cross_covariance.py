import torch

from adlers.vision.cross_covariance import XCA
from tests._typing import MakeQKV


def test_xca_matches_pinned_xcit_behavior(make_qkv: MakeQKV) -> None:
    """Checks the pinned XCiT cross-covariance attention output."""
    batch_size = 1
    num_heads = 2
    num_tokens = 4
    head_dim = 3
    query, _, _ = make_qkv(
        batch_size=batch_size,
        num_heads=num_heads,
        num_queries=num_tokens,
        num_keys=num_tokens,
        head_dim=head_dim,
    )
    x = query.transpose(dim0=1, dim1=2).reshape(
        batch_size, num_tokens, num_heads * head_dim
    )
    attention = XCA(dim=num_heads * head_dim, num_heads=num_heads)

    with torch.no_grad():
        identity = torch.eye(n=num_heads * head_dim)
        attention.qkv.weight.copy_(identity.repeat(3, 1))
        attention.proj.weight.copy_(identity)
        assert attention.proj.bias is not None
        attention.proj.bias.zero_()
        attention.temperature.copy_(
            torch.tensor([0.5, 1.5]).reshape(num_heads, 1, 1)
        )

    output = attention(x=x)

    expected_output = torch.tensor(
        [
            [
                [
                    -0.02354858,
                    -0.06868488,
                    -0.10348340,
                    1.14431930,
                    1.19752169,
                    0.02320504,
                ],
                [
                    1.06503868,
                    1.03925395,
                    0.99524808,
                    -0.56248677,
                    -0.66829437,
                    0.87732321,
                ],
                [
                    -0.59352618,
                    -0.71019590,
                    -0.64062285,
                    0.53730857,
                    0.79480863,
                    0.56489396,
                ],
                [
                    0.15881109,
                    0.02678001,
                    -0.17744431,
                    1.20716035,
                    1.03630614,
                    0.39902550,
                ],
            ]
        ]
    )
    torch.testing.assert_close(actual=output, expected=expected_output)
