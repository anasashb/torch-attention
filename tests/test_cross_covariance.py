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
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cross_covariance_attention_preserves_device_and_dtype_during_lazy_initialization(
    device: torch.device,
    dtype: torch.dtype,
    make_qkv: MakeQKV,
) -> None:
    """Checks that lazy temperature keeps the module's device and dtype."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=3,
        head_dim=6,
    )
    query = query.to(device=device, dtype=dtype)
    key = key.to(device=device, dtype=dtype)
    value = value.to(device=device, dtype=dtype)
    attention = CrossCovarianceAttention().to(device=device, dtype=dtype)

    attention(query=query, key=key, value=value, attn_mask=None)

    torch.testing.assert_close(
        actual=attention.temperature,
        expected=torch.ones(size=(4, 1, 1), device=device, dtype=dtype),
    )


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


@pytest.mark.parametrize(
    ("num_heads", "input_num_heads"),
    [(4, 1), (1, 4)],
)
def test_cross_covariance_attention_rejects_mismatched_configured_head_count(
    num_heads: int,
    input_num_heads: int,
    make_qkv: MakeQKV,
) -> None:
    """Checks that input head counts match the configured num_heads."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=input_num_heads,
        num_queries=3,
        num_keys=3,
        head_dim=6,
    )
    attention = CrossCovarianceAttention(num_heads=num_heads)

    with pytest.raises(ValueError) as error:
        attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )

    assert str(error.value) == (
        f"Cross-covariance attention was configured with num_heads={num_heads}; "
        f"got query head count {input_num_heads}. "
        "Set num_heads to match the input tensors."
    )


def test_cross_covariance_attention_infers_head_count_from_input(
    make_qkv: MakeQKV,
) -> None:
    """Checks that XCA initializes one temperature per input head."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=3,
        head_dim=6,
    )
    attention = CrossCovarianceAttention()

    attention(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )

    torch.testing.assert_close(
        actual=attention.temperature,
        expected=torch.ones(size=(4, 1, 1)),
    )


def test_cross_covariance_attention_rejects_changed_inferred_head_count(
    make_qkv: MakeQKV,
) -> None:
    """Checks that the inferred head count stays fixed after initialization."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=3,
        head_dim=6,
    )
    attention = CrossCovarianceAttention()
    attention(query=query, key=key, value=value, attn_mask=None)

    with pytest.raises(ValueError) as error:
        attention(
            query=query[:, :1],
            key=key[:, :1],
            value=value[:, :1],
            attn_mask=None,
        )

    assert str(error.value) == (
        "Cross-covariance attention was configured with num_heads=4; "
        "got query head count 1. "
        "Set num_heads to match the input tensors."
    )


def test_cross_covariance_attention_loads_temperature_before_lazy_initialization(
    make_qkv: MakeQKV,
) -> None:
    """Checks that loading a checkpoint preserves per-head temperatures."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=3,
        head_dim=6,
    )
    original = CrossCovarianceAttention(num_heads=4)

    with torch.no_grad():
        original.temperature.copy_(
            torch.tensor([0.5, 1.0, 1.5, 2.0]).reshape(4, 1, 1)
        )

    expected_output = original(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )

    restored = CrossCovarianceAttention()
    restored.load_state_dict(state_dict=original.state_dict())
    output = restored(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
    )

    torch.testing.assert_close(
        actual=restored.temperature,
        expected=original.temperature,
    )
    torch.testing.assert_close(actual=output, expected=expected_output)


@pytest.mark.parametrize(
    "num_heads",
    [
        pytest.param(None, id="lazy"),
        pytest.param(4, id="explicit"),
    ],
)
def test_cross_covariance_attention_updates_temperature_during_training(
    num_heads: int | None,
    make_qkv: MakeQKV,
) -> None:
    """Checks that per-head temperatures receive gradients and updates."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=3,
        head_dim=6,
    )
    attention = CrossCovarianceAttention(num_heads=num_heads)
    output = attention(query=query, key=key, value=value, attn_mask=None)

    initial_temperature = attention.temperature.detach().clone()
    optimizer = torch.optim.SGD(params=attention.parameters(), lr=0.1)

    output.square().mean().backward()

    assert attention.temperature.grad is not None
    assert torch.isfinite(input=attention.temperature.grad).all()

    optimizer.step()

    assert not torch.equal(
        input=attention.temperature,
        other=initial_temperature,
    )


@pytest.mark.parametrize("training", [True, False])
def test_cross_covariance_attention_applies_dropout_only_during_training(
    training: bool,
    make_qkv: MakeQKV,
) -> None:
    """Checks that configured dropout applies in training but not evaluation."""
    query, key, value = make_qkv(
        batch_size=2,
        num_heads=4,
        num_queries=3,
        num_keys=3,
        head_dim=6,
    )
    attention = CrossCovarianceAttention(dropout_rate=1.0)
    attention.train(mode=training)

    output = attention(query=query, key=key, value=value, attn_mask=None)

    if training:
        expected_output = torch.zeros_like(input=output)
    else:
        attention_without_dropout = CrossCovarianceAttention(dropout_rate=0.0)
        attention_without_dropout.eval()
        expected_output = attention_without_dropout(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )

    torch.testing.assert_close(actual=output, expected=expected_output)
