import pytest
import torch

from adlers.shared._attention_base import AttentionBase


@pytest.mark.parametrize(
    ("attn_mask_shape", "expected_shape"),
    [
        ((4, 4), (4, 4)),
        ((2, 4, 4), (2, 1, 4, 4)),
        ((2, 1, 4, 4), (2, 1, 4, 4)),
        ((2, 3, 4, 4), (2, 3, 4, 4)),
    ],
)
def test_attention_base_preserves_broadcastable_mask_shapes(
    attn_mask_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> None:
    """Checks that mask normalization avoids unnecessary expansion."""
    attn_mask = torch.zeros(attn_mask_shape, dtype=torch.bool)

    normalized_mask = AttentionBase._normalize_attn_mask(attn_mask=attn_mask)

    assert normalized_mask.shape == expected_shape
