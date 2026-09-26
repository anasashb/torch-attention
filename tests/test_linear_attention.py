import pytest
import torch

from adlers.nlp import LinearAttention
from tests._typing import MakeQKV


def test_linear_attention_matches_pinned_fast_transformers_behavior() -> None:
    """Checks the pinned fast-transformers unmasked attention output."""
    query = torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]).repeat(1, 3, 1, 1)
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]).repeat(1, 3, 1, 1)
    value = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]).repeat(1, 3, 1, 1)
    attention = LinearAttention(eps=1e-6)

    output = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )

    expected_output = torch.tensor(
        [[[[1.77024138, 2.77024126], [2.22975802, 3.22975779]]]]
    ).repeat(1, 3, 1, 1)
    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(output, expected_output)


def test_linear_attention_applies_causal_attention() -> None:
    """Checks that causal mode excludes future key-value positions."""
    query = torch.zeros((1, 1, 2, 1))
    key = torch.zeros((1, 1, 2, 1))
    value = torch.tensor([[[[2.0], [4.0]]]])
    attention = LinearAttention(
        feature_map=torch.ones_like,
        eps=0.0,
        is_causal=True,
    )

    output = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )

    expected_output = torch.tensor([[[[2.0], [3.0]]]])
    torch.testing.assert_close(actual=output, expected=expected_output)


@pytest.mark.parametrize(
    ("num_queries", "num_keys", "value_head_dim"),
    [(3, 3, 6), (3, 5, 4)],
)
def test_linear_attention_matches_quadratic_reference(
    num_queries: int,
    num_keys: int,
    value_head_dim: int,
    make_qkv: MakeQKV,
) -> None:
    """Checks linear attention against the full query-key calculation."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=num_queries,
        num_keys=num_keys,
        head_dim=6,
    )
    value = value[..., :value_head_dim]
    eps = 1e-6
    attention = LinearAttention(eps=eps)

    output = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )

    mapped_query = torch.nn.functional.elu(input=query) + 1
    mapped_key = torch.nn.functional.elu(input=key) + 1
    scores = mapped_query @ mapped_key.transpose(dim0=-2, dim1=-1)
    weights = scores / (scores.sum(dim=-1, keepdim=True) + eps)
    expected_output = weights @ value

    torch.testing.assert_close(actual=output, expected=expected_output)


def test_linear_attention_accepts_tensor_feature_map() -> None:
    """Checks that feature maps operate directly on tensors."""
    query = torch.zeros((1, 1, 2, 1))
    key = torch.zeros((1, 1, 2, 1))
    value = torch.tensor([[[[2.0], [4.0]]]])
    attention = LinearAttention(feature_map=torch.ones_like, eps=0.0)

    output = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )

    torch.testing.assert_close(output, torch.full_like(value, 3.0))


def test_linear_attention_applies_key_padding_mask() -> None:
    """Checks that padded keys do not contribute to attention."""
    query = torch.zeros((1, 1, 1, 1))
    key = torch.zeros((1, 1, 2, 1))
    value = torch.tensor([[[[2.0], [4.0]]]])
    attn_mask = torch.tensor([[[[False, True]]]])
    attention = LinearAttention(feature_map=torch.ones_like, eps=0.0)

    output = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
    )

    torch.testing.assert_close(output, torch.tensor([[[[2.0]]]]))


@pytest.mark.parametrize("is_causal", [False, True])
def test_linear_attention_rejects_query_dependent_attention_masks(
    is_causal: bool,
) -> None:
    """Checks that LinearAttention rejects query-dependent masks."""
    query = torch.zeros((1, 1, 2, 2))
    attn_mask = torch.zeros((2, 2), dtype=torch.bool)
    attention = LinearAttention(is_causal=is_causal)

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


def test_linear_attention_rejects_nonzero_dropout_rate() -> None:
    """Checks that unsupported LinearAttention dropout is rejected."""
    with pytest.raises(ValueError) as error:
        LinearAttention(dropout_rate=0.1)

    assert str(error.value) == (
        "Linear attention does not support dropout; got dropout_rate 0.1. "
        "Set dropout_rate=0.0."
    )


def test_linear_attention_ignores_dropout_outside_strict_mode() -> None:
    """Checks that nonzero dropout is ignored when strict mode is off."""
    query = torch.zeros((1, 1, 2, 1))
    value = torch.tensor([[[[2.0], [4.0]]]])
    attention = LinearAttention(
        feature_map=torch.ones_like,
        eps=0.0,
        dropout_rate=1.0,
        strict_mode=False,
    )

    output = attention(
        query=query,
        key=query,
        value=value,
        attn_mask=None,
    )

    torch.testing.assert_close(output, torch.full_like(value, 3.0))


@pytest.mark.parametrize("is_causal", [False, True])
def test_linear_attention_rejects_non_boolean_key_padding_masks(
    is_causal: bool,
) -> None:
    """Checks that LinearAttention requires boolean key-padding masks."""
    query = torch.zeros(size=(1, 1, 2, 2))
    attn_mask = torch.zeros(size=(1, 1, 1, 2), dtype=torch.float32)
    attention = LinearAttention(is_causal=is_causal)

    with pytest.raises(TypeError) as error:
        attention(
            query=query,
            key=query,
            value=query,
            attn_mask=attn_mask,
        )

    assert str(error.value) == (
        "Only boolean attention masks are supported; "
        "got mask dtype torch.float32. Use a torch.bool mask "
        "with True for positions that should be masked out and "
        "False for positions that can be attended to."
    )


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    ("tensor_name", "tensor_index"),
    [
        ("query", 0),
        ("key", 1),
        ("value", 2),
    ],
)
def test_linear_attention_rejects_non_four_dimensional_qkv_tensors(
    is_causal: bool,
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
    attention = LinearAttention(is_causal=is_causal)

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


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    ("dimension", "dimension_name", "expected_guidance"),
    [
        pytest.param(
            0,
            "batch size",
            "Use the same batch size for all three tensors.",
            id="batch-size",
        ),
        pytest.param(
            1,
            "head count",
            "Use the same number of heads for all three tensors.",
            id="head-count",
        ),
    ],
)
@pytest.mark.parametrize(
    "tensor_index",
    [
        pytest.param(0, id="query"),
        pytest.param(1, id="key"),
        pytest.param(2, id="value"),
    ],
)
def test_linear_attention_rejects_mismatched_qkv_batch_sizes_and_head_counts(
    is_causal: bool,
    tensor_index: int,
    dimension: int,
    dimension_name: str,
    expected_guidance: str,
    make_qkv: MakeQKV,
) -> None:
    """Checks that query, key, and value batch sizes and head counts match."""
    tensors = list(
        make_qkv(
            batch_size=2,
            num_heads=4,
            num_queries=3,
            num_keys=3,
            head_dim=6,
        )
    )
    tensors[tensor_index] = tensors[tensor_index].narrow(
        dim=dimension,
        start=0,
        length=1,
    )
    query, key, value = tensors
    attention = LinearAttention(is_causal=is_causal)

    with pytest.raises(ValueError) as error:
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )

    assert str(error.value) == (
        f"Query, key, and value {dimension_name}s must match; "
        f"got query {dimension_name} {query.shape[dimension]}, "
        f"key {dimension_name} {key.shape[dimension]}, and "
        f"value {dimension_name} {value.shape[dimension]}. "
        f"{expected_guidance}"
    )


@pytest.mark.parametrize("is_causal", [False, True])
def test_linear_attention_rejects_unequal_key_and_value_lengths(
    is_causal: bool,
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
    attention = LinearAttention(is_causal=is_causal)

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


@pytest.mark.parametrize("is_causal", [False, True])
def test_linear_attention_rejects_mismatched_query_and_key_head_dimensions(
    is_causal: bool,
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
    key = torch.zeros(size=(2, 4, 3, 5))
    attention = LinearAttention(is_causal=is_causal)

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
