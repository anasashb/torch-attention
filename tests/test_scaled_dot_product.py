import pytest
import torch
import torch.nn.functional as F

from adlers._typing import AttentionBackend
from adlers.shared import ScaledDotProductAttention
from tests._typing import MakeQKV


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("output_attention_scores", [False, True])
def test_scaled_dot_product(
    is_causal: bool,
    output_attention_scores: bool,
    make_qkv: MakeQKV,
) -> None:
    """Checks default einsum attention shapes and optional attention scores."""
    batch_size = 32
    num_heads = 4
    num_queries = 96
    num_keys = 96
    head_dim = 16

    query, key, value = make_qkv(
        batch_size=batch_size,
        num_heads=num_heads,
        num_queries=num_queries,
        num_keys=num_keys,
        head_dim=head_dim,
    )

    attention = ScaledDotProductAttention(
        is_causal=is_causal,
        dropout_rate=0.0,
        backend="einsum",
        output_attention_scores=output_attention_scores,
        strict_mode=True,
        custom_scale_factor=None,
    )

    if is_causal:
        # let it generate the triangular mask on its own
        out, weights = attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )
        assert torch.isfinite(out).all()
    else:
        out, weights = attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )

    assert out.shape == (batch_size, num_heads, num_queries, head_dim)

    if output_attention_scores:
        assert weights.shape == (batch_size, num_heads, num_queries, num_keys)

        summed_weights = weights.sum(dim=-1)
        assert torch.allclose(
            summed_weights, torch.ones_like(summed_weights), atol=1e-5
        )


@pytest.mark.parametrize("is_causal", [False, True])
def test_sdpa_backend_matches_torch_scaled_dot_product_attention(
    is_causal: bool,
    make_qkv: MakeQKV,
) -> None:
    """Checks that the SDPA backend matches PyTorch's native implementation."""
    query, key, value = make_qkv()

    attention = ScaledDotProductAttention(
        is_causal=is_causal,
        dropout_rate=0.0,
        backend="sdpa",
        output_attention_scores=False,
        strict_mode=True,
        custom_scale_factor=None,
    )

    out, weights = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )
    expected_out = F.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
    )

    assert weights is None
    torch.testing.assert_close(out, expected_out)


@pytest.mark.parametrize("backend", ["einsum", "sdpa"])
def test_scaled_dot_product_disables_dropout_during_evaluation(
    backend: AttentionBackend,
    make_qkv: MakeQKV,
) -> None:
    """Checks that evaluation mode disables dropout for every backend."""
    query, key, value = make_qkv()
    output_attention_scores = backend == "einsum"
    attention_with_dropout = ScaledDotProductAttention(
        dropout_rate=0.5,
        backend=backend,
        output_attention_scores=output_attention_scores,
    )
    attention_without_dropout = ScaledDotProductAttention(
        dropout_rate=0.0,
        backend=backend,
        output_attention_scores=output_attention_scores,
    )
    attention_with_dropout.eval()
    attention_without_dropout.eval()

    output, weights = attention_with_dropout(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )
    expected_output, expected_weights = attention_without_dropout(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )

    torch.testing.assert_close(output, expected_output)
    if output_attention_scores:
        assert weights is not None
        assert expected_weights is not None
        torch.testing.assert_close(weights, expected_weights)
    else:
        assert weights is None
        assert expected_weights is None


@pytest.mark.parametrize("backend", ["einsum", "sdpa"])
def test_scaled_dot_product_supports_cpu_bfloat16_autocast(
    backend: AttentionBackend,
    make_qkv: MakeQKV,
) -> None:
    """Checks CPU bfloat16 autocast compatibility for every backend."""
    query, key, value = make_qkv()
    output_attention_scores = backend == "einsum"
    attention = ScaledDotProductAttention(
        backend=backend,
        output_attention_scores=output_attention_scores,
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output, weights = attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )

    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    if output_attention_scores:
        assert weights is not None
        assert weights.dtype == torch.bfloat16
        assert torch.isfinite(weights).all()
    else:
        assert weights is None


@pytest.mark.parametrize(
    "attn_mask_shape",
    [
        (4, 4),
        (2, 4, 4),
        (2, 1, 4, 4),
        (2, 3, 4, 4),
    ],
)
def test_scaled_dot_product_supports_mask_broadcasting(
    attn_mask_shape: tuple[int, ...],
    make_qkv: MakeQKV,
) -> None:
    """Checks supported mask shapes and semantics across both backends."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=3,
        num_queries=4,
        num_keys=4,
    )
    attn_mask = torch.zeros(attn_mask_shape, dtype=torch.bool)
    attn_mask[..., -1] = True

    einsum_attention = ScaledDotProductAttention(
        is_causal=False,
        dropout_rate=0.0,
        backend="einsum",
        output_attention_scores=True,
        strict_mode=True,
        custom_scale_factor=None,
    )
    sdpa_attention = ScaledDotProductAttention(
        is_causal=False,
        dropout_rate=0.0,
        backend="sdpa",
        output_attention_scores=False,
        strict_mode=True,
        custom_scale_factor=None,
    )

    expected_out, expected_weights = einsum_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
    )
    out, weights = sdpa_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
    )

    assert expected_weights is not None
    assert torch.count_nonzero(expected_weights[..., -1]) == 0
    assert weights is None
    torch.testing.assert_close(out, expected_out)


def test_scaled_dot_product_backends_match_with_different_query_and_key_lengths(
    make_qkv: MakeQKV,
) -> None:
    """Checks cross-attention with different query and key lengths."""
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
    attn_mask = torch.zeros(num_queries, num_keys, dtype=torch.bool)
    attn_mask[:, -1] = True

    einsum_attention = ScaledDotProductAttention(
        is_causal=False,
        dropout_rate=0.0,
        backend="einsum",
        output_attention_scores=True,
        strict_mode=True,
        custom_scale_factor=None,
    )
    sdpa_attention = ScaledDotProductAttention(
        is_causal=False,
        dropout_rate=0.0,
        backend="sdpa",
        output_attention_scores=False,
        strict_mode=True,
        custom_scale_factor=None,
    )

    expected_out, expected_weights = einsum_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
    )
    out, weights = sdpa_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
    )

    assert expected_out.shape == (
        batch_size,
        num_heads,
        num_queries,
        query.shape[-1],
    )
    assert expected_weights is not None
    assert expected_weights.shape == (
        batch_size,
        num_heads,
        num_queries,
        num_keys,
    )
    assert torch.count_nonzero(expected_weights[..., -1]) == 0
    assert weights is None
    torch.testing.assert_close(out, expected_out)


def test_scaled_dot_product_rejects_unequal_key_and_value_lengths(
    make_qkv: MakeQKV,
) -> None:
    """Checks that each key position must have a corresponding value."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=5,
    )
    value = value[..., :-1, :]
    attention = ScaledDotProductAttention(
        backend="einsum",
        strict_mode=True,
    )

    with pytest.raises(
        ValueError,
        match="Key and value sequence lengths must match",
    ):
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )


@pytest.mark.parametrize(
    ("tensor_name", "tensor_index"),
    [
        ("query", 0),
        ("key", 1),
        ("value", 2),
    ],
)
def test_scaled_dot_product_rejects_non_four_dimensional_qkv_tensors(
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
    attention = ScaledDotProductAttention(strict_mode=True)

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
    ("key_shape", "expected_message"),
    [
        pytest.param(
            (1, 4, 5, 6),
            "Query, key, and value batch sizes must match; got query batch "
            "size 2, key batch size 1, and value batch size 2. Use the same "
            "batch size for all three tensors.",
            id="batch-size",
        ),
        pytest.param(
            (2, 3, 5, 6),
            "Query, key, and value head counts must match; got query head "
            "count 4, key head count 3, and value head count 4. Use the same "
            "number of heads for all three tensors.",
            id="head-count",
        ),
        pytest.param(
            (2, 4, 5, 5),
            "Query, key, and value head dimensions must match; got query head "
            "dimension 6, key head dimension 5, and value head dimension 6. "
            "Use the same head dimension for all three tensors.",
            id="head-dimension",
        ),
    ],
)
def test_scaled_dot_product_rejects_qkv_shape_mismatches(
    key_shape: tuple[int, int, int, int],
    expected_message: str,
    make_qkv: MakeQKV,
) -> None:
    """Checks batch size, head count, and head dimension mismatches."""
    query, _, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=5,
        head_dim=6,
    )
    key = torch.randn(key_shape)
    attention = ScaledDotProductAttention(strict_mode=True)

    with pytest.raises(ValueError) as error:
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )

    assert str(error.value) == expected_message


def test_scaled_dot_product_combines_causal_and_explicit_masks(
    make_qkv: MakeQKV,
) -> None:
    """Checks that explicit restrictions are added to causal masking."""
    query, key, value = make_qkv(
        batch_size=1,
        num_heads=1,
        num_queries=4,
        num_keys=4,
    )
    attn_mask = torch.zeros(4, 4, dtype=torch.bool)
    attn_mask[2:, 0] = True
    causal_mask = torch.triu(torch.ones_like(attn_mask), diagonal=1)
    combined_mask = causal_mask | attn_mask
    einsum_attention = ScaledDotProductAttention(
        is_causal=True,
        backend="einsum",
        output_attention_scores=True,
    )
    sdpa_attention = ScaledDotProductAttention(
        is_causal=True,
        backend="sdpa",
        output_attention_scores=False,
    )

    expected_out, expected_weights = einsum_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
    )
    out, weights = sdpa_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
    )

    assert expected_weights is not None
    assert (
        torch.count_nonzero(expected_weights.masked_select(combined_mask)) == 0
    )
    assert weights is None
    torch.testing.assert_close(out, expected_out)


def test_scaled_dot_product_backends_zero_fully_masked_query_rows(
    make_qkv: MakeQKV,
) -> None:
    """Checks that fully masked query rows produce zeros across backends."""
    batch_size = 2
    num_heads = 4
    num_queries = 3
    num_keys = 5
    fully_masked_query_index = 1
    query, key, value = make_qkv(
        batch_size=batch_size,
        num_heads=num_heads,
        num_queries=num_queries,
        num_keys=num_keys,
    )
    attn_mask = torch.zeros(num_queries, num_keys, dtype=torch.bool)
    attn_mask[fully_masked_query_index, :] = True
    einsum_attention = ScaledDotProductAttention(
        backend="einsum",
        output_attention_scores=True,
    )
    sdpa_attention = ScaledDotProductAttention(backend="sdpa")

    einsum_output, einsum_weights = einsum_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
    )
    sdpa_output, sdpa_weights = sdpa_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
    )

    assert einsum_weights is not None
    assert sdpa_weights is None
    torch.testing.assert_close(
        einsum_weights[..., fully_masked_query_index, :],
        torch.zeros_like(einsum_weights[..., fully_masked_query_index, :]),
    )
    torch.testing.assert_close(
        einsum_output[..., fully_masked_query_index, :],
        torch.zeros_like(einsum_output[..., fully_masked_query_index, :]),
    )
    torch.testing.assert_close(
        sdpa_output[..., fully_masked_query_index, :],
        torch.zeros_like(sdpa_output[..., fully_masked_query_index, :]),
    )
    torch.testing.assert_close(einsum_output, sdpa_output)


def test_sdpa_backend_rejects_attention_scores() -> None:
    """Checks that SDPA rejects unsupported attention score output."""
    with pytest.raises(ValueError, match="does not support"):
        ScaledDotProductAttention(
            backend="sdpa",
            output_attention_scores=True,
        )


def test_scaled_dot_product_rejects_invalid_backend() -> None:
    """Checks that unknown attention backends fail fast."""
    with pytest.raises(ValueError, match="Invalid backend"):
        ScaledDotProductAttention(backend="supermaxx")  # type: ignore[arg-type]


def test_scaled_dot_product_requires_boolean_mask(
    make_qkv: MakeQKV,
) -> None:
    """Checks that attention masks must use torch.bool dtype."""
    query, key, value = make_qkv()
    attn_mask = torch.zeros(
        query.shape[-2],
        key.shape[-2],
        dtype=torch.float32,
    )
    attention = ScaledDotProductAttention(
        is_causal=False,
        strict_mode=True,
    )

    with pytest.raises(
        TypeError, match="Only boolean attention masks are supported"
    ):
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
        )


@pytest.mark.parametrize(
    (
        "attn_mask_shape",
        "attn_mask_dtype",
        "expected_error",
        "expected_message",
    ),
    [
        pytest.param(
            (8, 8),
            torch.float32,
            TypeError,
            "Only boolean attention masks are supported",
            id="dtype",
        ),
        pytest.param(
            (8,),
            torch.bool,
            ValueError,
            "Attention mask must be 2D, 3D or 4D",
            id="rank",
        ),
    ],
)
def test_scaled_dot_product_rejects_invalid_masks_without_strict_mode(
    attn_mask_shape: tuple[int, ...],
    attn_mask_dtype: torch.dtype,
    expected_error: type[Exception],
    expected_message: str,
    make_qkv: MakeQKV,
) -> None:
    """Checks that mask dtype and rank validation cannot be disabled."""
    query, key, value = make_qkv()
    attn_mask = torch.zeros(attn_mask_shape, dtype=attn_mask_dtype)
    attention = ScaledDotProductAttention(strict_mode=False)

    with pytest.raises(expected_error, match=expected_message):
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
        )


@pytest.mark.parametrize(
    "attn_mask_shape",
    [
        pytest.param((4, 5), id="2d-query-length"),
        pytest.param((3, 3, 5), id="3d-batch-size"),
        pytest.param((2, 3, 3, 5), id="4d-head-count"),
    ],
)
def test_scaled_dot_product_rejects_invalid_mask_shapes_in_strict_mode(
    attn_mask_shape: tuple[int, ...],
    make_qkv: MakeQKV,
) -> None:
    """Checks exact mask dimensions when strict validation is enabled."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=5,
    )
    attn_mask = torch.zeros(attn_mask_shape, dtype=torch.bool)
    attention = ScaledDotProductAttention(strict_mode=True)

    with pytest.raises(ValueError, match="Invalid mask shape"):
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
        )
