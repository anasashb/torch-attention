import pytest
import torch

from adlers.nlp import LinearAttention
from adlers.nlp._causal_product import _CausalDotProduct
from tests._typing import MakeQKV


def test_causal_linear_attention_is_causal_without_explicit_mask() -> None:
    """Checks causal behavior without an explicit triangular mask."""
    query = torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]]).repeat(1, 3, 1, 1)
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]).repeat(1, 3, 1, 1)
    value = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]).repeat(1, 3, 1, 1)
    attention = LinearAttention(is_causal=True, eps=1e-6)

    output = attention(
        query=query,
        key=key,
        value=value,
    )

    expected_output = torch.tensor(
        [[[[0.99999976, 1.99999952], [2.22975826, 3.22975802]]]]
    ).repeat(1, 3, 1, 1)
    torch.testing.assert_close(actual=output, expected=expected_output)
    assert output.is_contiguous()


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(torch.device("cpu"), id="cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available",
            ),
            id="cuda",
        ),
    ],
)
@pytest.mark.parametrize("value_head_dim", [6, 4])
def test_causal_linear_attention_matches_explicit_attention_calculation(
    device: torch.device,
    value_head_dim: int,
    make_qkv: MakeQKV,
) -> None:
    """Checks causal Linear Attention against an explicit score matrix."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=3,
        head_dim=6,
    )
    query = query.to(device=device).requires_grad_()
    key = key.to(device=device).requires_grad_()
    value = value[..., :value_head_dim].to(device=device).requires_grad_()
    eps = 1e-6
    attention = LinearAttention(is_causal=True, eps=eps)

    output = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )

    mapped_query = torch.nn.functional.elu(input=query) + 1
    mapped_key = torch.nn.functional.elu(input=key) + 1
    scores = mapped_query @ mapped_key.transpose(dim0=-2, dim1=-1)
    causal_mask = torch.triu(
        torch.ones(
            size=(query.shape[-2], key.shape[-2]),
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )
    scores = scores.masked_fill(mask=causal_mask, value=0)
    weights = scores / (scores.sum(dim=-1, keepdim=True) + eps)
    expected_output = weights @ value
    output_gradient = torch.linspace(
        start=0.1,
        end=1.0,
        steps=output.numel(),
        dtype=output.dtype,
        device=output.device,
    ).reshape_as(output)

    actual_gradients = torch.autograd.grad(
        outputs=output,
        inputs=(query, key, value),
        grad_outputs=output_gradient,
    )
    expected_gradients = torch.autograd.grad(
        outputs=expected_output,
        inputs=(query, key, value),
        grad_outputs=output_gradient,
    )

    torch.testing.assert_close(actual=output, expected=expected_output)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual=actual_gradient,
            expected=expected_gradient,
        )


def test_causal_linear_attention_applies_key_padding_mask() -> None:
    """Checks that padded keys do not contribute to causal attention."""
    query = torch.zeros((1, 1, 2, 1))
    key = torch.zeros((1, 1, 2, 1))
    value = torch.tensor([[[[2.0], [4.0]]]])
    attn_mask = torch.tensor([[[[False, True]]]])
    attention = LinearAttention(
        is_causal=True,
        feature_map=torch.ones_like,
        eps=0.0,
    )

    output = attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
    )

    expected_output = torch.tensor([[[[2.0], [2.0]]]])
    torch.testing.assert_close(actual=output, expected=expected_output)


def test_causal_linear_attention_rejects_different_query_and_key_lengths(
    make_qkv: MakeQKV,
) -> None:
    """Checks that causal query, key, and value lengths must match."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=5,
        head_dim=6,
    )
    attention = LinearAttention(is_causal=True)

    with pytest.raises(ValueError) as error:
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )

    assert str(error.value) == (
        "Query, key, and value sequence lengths must match for causal "
        "linear attention; got query length 3, key length 5, and "
        "value length 5. Use the same sequence length for all three tensors."
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float64],
)
def test_causal_linear_attention_rejects_non_float32_tensors(
    dtype: torch.dtype,
) -> None:
    """Checks that the compiled causal product requires float32 tensors."""
    query = torch.zeros(size=(1, 1, 2, 2), dtype=dtype)
    attention = LinearAttention(is_causal=True)

    with pytest.raises(TypeError) as error:
        attention(
            query=query,
            key=query,
            value=query,
            attn_mask=None,
        )

    assert str(error.value) == (
        "Causal Linear Attention only supports torch.float32 tensors; "
        f"got query dtype {dtype}, key dtype {dtype}, and value dtype {dtype}."
    )


def test_causal_linear_attention_preserves_query_dtype() -> None:
    """Checks that native output allocation follows the query dtype."""
    query = torch.zeros(
        size=(1, 1, 2, 2),
        dtype=torch.float32,
    )
    attention = LinearAttention(is_causal=True)
    original_default_dtype = torch.get_default_dtype()

    try:
        torch.set_default_dtype(torch.float64)
        output = attention(
            query=query,
            key=query,
            value=query,
            attn_mask=None,
        )
    finally:
        torch.set_default_dtype(original_default_dtype)

    assert output.dtype == query.dtype


def test_causal_linear_attention_reports_unavailable_compiled_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks the error shown when a compiled operation is unavailable."""
    monkeypatch.setitem(_CausalDotProduct.dot, "cpu", None)
    query = torch.zeros(size=(1, 1, 2, 2))
    attention = LinearAttention(is_causal=True)

    with pytest.raises(RuntimeError) as error:
        attention(
            query=query,
            key=query,
            value=query,
            attn_mask=None,
        )

    assert str(error.value) == (
        "Causal Linear Attention cannot run on device type 'cpu' because its "
        "compiled extension is unavailable. Reinstall ADLERS with support for "
        "that device or use another supported device."
    )
