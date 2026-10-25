import importlib
from types import ModuleType

import numpy as np
import pytest
import torch

from adlers.vision import local_attention

pytestmark = pytest.mark.tf_reference


@pytest.fixture(scope="module")
def tf() -> ModuleType:
    """Returns TensorFlow with GPUs hidden, skipping when it is missing."""
    tensorflow = pytest.importorskip(modname="tensorflow")
    tensorflow.config.set_visible_devices(devices=[], device_type="GPU")
    return tensorflow


@pytest.fixture(scope="module")
def common_attention(tf: ModuleType) -> ModuleType:
    """Returns the Tensor2Tensor reference, after TensorFlow is set up."""
    return importlib.import_module(
        name="tests.reference.tensor2tensor.common_attention"
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """Returns a NumPy generator for inputs shared by both frameworks."""
    return np.random.default_rng(seed=66)


@pytest.mark.parametrize(
    argnames="shape",
    argvalues=[
        pytest.param((), id="scalar"),
        pytest.param((5,), id="vector"),
        pytest.param((2, 3, 5, 4), id="sequence"),
        pytest.param((2, 3, 2, 4, 6), id="blocked"),
    ],
)
def test_shape_list_matches_reference(
    tf: ModuleType,
    common_attention: ModuleType,
    shape: tuple[int, ...],
) -> None:
    """Checks shape_list against the Tensor2Tensor reference."""
    x = np.zeros(shape=shape, dtype=np.float32)

    expected = common_attention.shape_list(x=tf.constant(value=x))
    actual = local_attention.shape_list(x=torch.from_numpy(x))

    assert actual == expected


@pytest.mark.parametrize(
    argnames="shape",
    argvalues=[
        pytest.param((2, 5, 4), id="sequence"),
        pytest.param((2, 3, 2, 4, 6), id="blocked"),
    ],
)
def test_embedding_to_padding_matches_reference(
    tf: ModuleType,
    common_attention: ModuleType,
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> None:
    """Checks embedding_to_padding against the Tensor2Tensor reference."""
    emb = rng.standard_normal(size=shape).astype(dtype=np.float32)
    emb[..., -2:, :] = 0.0

    expected = common_attention.embedding_to_padding(emb=tf.constant(value=emb))
    actual = local_attention.embedding_to_padding(emb=torch.from_numpy(emb))

    torch.testing.assert_close(
        actual=actual,
        expected=torch.from_numpy(expected.numpy()),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    argnames=("rows", "cols", "num_lower", "num_upper", "out_shape"),
    argvalues=[
        pytest.param(4, 4, -1, -1, None, id="unlimited"),
        pytest.param(4, 4, -1, 0, None, id="lower-triangle"),
        pytest.param(3, 5, 1, 2, None, id="non-square"),
        pytest.param(4, 8, -1, 4, None, id="masked-1d-block"),
        pytest.param(4, 4, 1, 1, [1, 1, 4, 4], id="reshaped"),
    ],
)
@pytest.mark.parametrize(
    argnames="as_tensor",
    argvalues=[
        pytest.param(False, id="int"),
        pytest.param(True, id="tensor"),
    ],
)
def test_ones_matrix_band_part_matches_reference(
    tf: ModuleType,
    common_attention: ModuleType,
    rows: int,
    cols: int,
    num_lower: int,
    num_upper: int,
    out_shape: list[int] | None,
    as_tensor: bool,
) -> None:
    """Checks ones_matrix_band_part against the Tensor2Tensor reference."""
    arguments = {
        "rows": rows,
        "cols": cols,
        "num_lower": num_lower,
        "num_upper": num_upper,
    }
    tf_arguments = {
        name: tf.constant(value=value) if as_tensor else value
        for name, value in arguments.items()
    }
    torch_arguments = {
        name: torch.tensor(data=value) if as_tensor else value
        for name, value in arguments.items()
    }

    expected = common_attention.ones_matrix_band_part(
        **tf_arguments, out_shape=out_shape
    )
    actual = local_attention.ones_matrix_band_part(
        **torch_arguments, out_shape=out_shape
    )

    torch.testing.assert_close(
        actual=actual,
        expected=torch.from_numpy(expected.numpy()),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    argnames=("length", "max_backward", "max_forward"),
    argvalues=[
        pytest.param(4, -1, 0, id="causal"),
        pytest.param(5, 1, 1, id="window"),
        pytest.param(5, -1, -1, id="unlimited"),
        pytest.param(1, 0, 0, id="single"),
    ],
)
@pytest.mark.parametrize(
    argnames="length_type",
    argvalues=["int", "numpy", "tensor"],
)
def test_attention_bias_local_matches_reference(
    tf: ModuleType,
    common_attention: ModuleType,
    length: int,
    max_backward: int,
    max_forward: int,
    length_type: str,
) -> None:
    """Checks attention_bias_local against the Tensor2Tensor reference."""
    tf_lengths = {
        "int": length,
        "numpy": np.int64(length),
        "tensor": tf.constant(value=length),
    }
    torch_lengths = {
        "int": length,
        "numpy": np.int64(length),
        "tensor": torch.tensor(data=length),
    }

    expected = common_attention.attention_bias_local(
        length=tf_lengths[length_type],
        max_backward=max_backward,
        max_forward=max_forward,
    )
    actual = local_attention.attention_bias_local(
        length=torch_lengths[length_type],
        max_backward=max_backward,
        max_forward=max_forward,
    )

    torch.testing.assert_close(
        actual=actual,
        expected=torch.from_numpy(expected.numpy()),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(argnames="length", argvalues=[1, 4])
@pytest.mark.parametrize(
    argnames="length_type",
    argvalues=["int", "numpy", "tensor"],
)
def test_attention_bias_lower_triangle_matches_reference(
    tf: ModuleType,
    common_attention: ModuleType,
    length: int,
    length_type: str,
) -> None:
    """Checks attention_bias_lower_triangle against Tensor2Tensor."""
    tf_lengths = {
        "int": length,
        "numpy": np.int64(length),
        "tensor": tf.constant(value=length),
    }
    torch_lengths = {
        "int": length,
        "numpy": np.int64(length),
        "tensor": torch.tensor(data=length),
    }

    expected = common_attention.attention_bias_lower_triangle(
        length=tf_lengths[length_type]
    )
    actual = local_attention.attention_bias_lower_triangle(
        length=torch_lengths[length_type]
    )

    torch.testing.assert_close(
        actual=actual,
        expected=torch.from_numpy(expected.numpy()),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    argnames=("query_shape", "key_shape", "value_shape", "bias_shape"),
    argvalues=[
        pytest.param(
            (2, 3, 5, 4),
            (2, 3, 7, 4),
            (2, 3, 7, 6),
            (1, 1, 5, 7),
            id="shared-bias",
        ),
        pytest.param(
            (2, 3, 2, 4, 4),
            (2, 3, 2, 8, 4),
            (2, 3, 2, 8, 6),
            (2, 3, 2, 1, 8),
            id="blocked",
        ),
    ],
)
def test_dot_product_attention_matches_reference(
    tf: ModuleType,
    common_attention: ModuleType,
    rng: np.random.Generator,
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    bias_shape: tuple[int, ...],
) -> None:
    """Checks dot_product_attention against the Tensor2Tensor reference."""
    q = rng.standard_normal(size=query_shape).astype(dtype=np.float32)
    k = rng.standard_normal(size=key_shape).astype(dtype=np.float32)
    v = rng.standard_normal(size=value_shape).astype(dtype=np.float32)
    bias = np.where(rng.random(size=bias_shape) < 0.25, -1e9, 0.0).astype(
        dtype=np.float32
    )

    expected = common_attention.dot_product_attention(
        q=tf.constant(value=q),
        k=tf.constant(value=k),
        v=tf.constant(value=v),
        bias=tf.constant(value=bias),
        dropout_rate=0.0,
        make_image_summary=False,
    )
    actual = local_attention.dot_product_attention(
        q=torch.from_numpy(q),
        k=torch.from_numpy(k),
        v=torch.from_numpy(v),
        bias=torch.from_numpy(bias),
        dropout_rate=0.0,
    )

    torch.testing.assert_close(
        actual=actual,
        expected=torch.from_numpy(expected.numpy()),
        rtol=1e-5,
        atol=1e-5,
    )
