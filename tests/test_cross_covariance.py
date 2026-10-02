import pytest
import torch

from adlers.vision.cross_covariance import CrossCovarianceAttention
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
    attention = CrossCovarianceAttention(
        num_heads=num_heads,
        is_causal=False,
        dropout_rate=0.0,
        strict_mode=True,
    )

    with torch.no_grad():
        attention.temperature.copy_(
            torch.tensor([0.5, 1.5]).reshape(num_heads, 1, 1)
        )

    output = attention(
        query=query,
        key=query,
        value=query,
        attn_mask=None,
    )

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
    expected_output = expected_output.reshape(
        batch_size, num_tokens, num_heads, head_dim
    ).transpose(dim0=1, dim1=2)
    torch.testing.assert_close(actual=output, expected=expected_output)


@pytest.mark.parametrize("strict_mode", [False, True])
def test_cross_covariance_attention_rejects_causal_mode(
    strict_mode: bool,
) -> None:
    """Checks that XCA rejects causal mode regardless of strict mode."""
    with pytest.raises(ValueError) as error:
        CrossCovarianceAttention(
            num_heads=2,
            is_causal=True,
            strict_mode=strict_mode,
        )

    assert str(error.value) == (
        "Cross-covariance attention does not support causal masking; "
        "got is_causal=True. Set is_causal=False."
    )


@pytest.mark.parametrize("strict_mode", [False, True])
def test_cross_covariance_attention_rejects_custom_attention_masks(
    strict_mode: bool,
    make_qkv: MakeQKV,
) -> None:
    """Checks that XCA rejects supplied attention masks."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=3,
        head_dim=6,
    )
    attn_mask = torch.zeros(size=(3, 3), dtype=torch.bool)
    attention = CrossCovarianceAttention(
        num_heads=4,
        strict_mode=strict_mode,
    )

    with pytest.raises(ValueError) as error:
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
        )

    assert str(error.value) == (
        "Cross-covariance attention does not support custom attention masks; "
        "got shape (3, 3). Pass attn_mask=None."
    )


def test_cross_covariance_attention_rejects_mismatched_query_and_key_lengths(
    make_qkv: MakeQKV,
) -> None:
    """Checks that XCA requires matching query and key lengths."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=5,
        head_dim=6,
    )
    attention = CrossCovarianceAttention(num_heads=4)

    with pytest.raises(ValueError) as error:
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )

    assert str(error.value) == (
        "Cross-covariance attention requires matching query and key "
        "sequence lengths; got query length 3 and key length 5. "
        "Use the same sequence length for both tensors."
    )


def test_cross_covariance_attention_rejects_mismatched_key_and_value_head_dimensions(
    make_qkv: MakeQKV,
) -> None:
    """Checks that XCA requires matching key and value head dimensions."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=3,
        head_dim=6,
    )
    value = value[..., :4]
    attention = CrossCovarianceAttention(num_heads=4)

    with pytest.raises(ValueError) as error:
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )

    assert str(error.value) == (
        "Cross-covariance attention requires matching key and value head "
        "dimensions; got key head dimension 6 and value head dimension 4. "
        "Use the same head dimension for both tensors."
    )
