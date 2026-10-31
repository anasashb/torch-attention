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


@pytest.mark.parametrize(
    argnames="shape",
    argvalues=[
        pytest.param((2, 3, 1, 4, 5), id="single-block"),
        pytest.param((2, 3, 2, 4, 5), id="two-blocks"),
        pytest.param((2, 3, 4, 4, 5), id="four-blocks"),
    ],
)
def test_make_local_block_matches_reference(
    tf: ModuleType,
    common_attention: ModuleType,
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> None:
    """Checks _make_local_block against the Tensor2Tensor reference."""
    x = rng.standard_normal(size=shape).astype(dtype=np.float32)
    batch, heads, num_blocks, block_length, depth = shape
    arguments = {
        "depth": depth,
        "batch": batch,
        "heads": heads,
        "num_blocks": num_blocks,
        "block_length": block_length,
    }

    expected = common_attention._make_local_block(
        x=tf.constant(value=x), **arguments
    )
    actual = local_attention._make_local_block(
        x=torch.from_numpy(x), **arguments
    )

    torch.testing.assert_close(
        actual=actual,
        expected=torch.from_numpy(expected.numpy()),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    argnames=("shape", "memory_block_size"),
    argvalues=[
        pytest.param((2, 3, 8, 5), 4, id="two-blocks"),
        pytest.param((2, 3, 8, 5), 8, id="one-block"),
        pytest.param((2, 3, 6, 5), 1, id="unit-blocks"),
    ],
)
def test_reshape_by_blocks_matches_reference(
    tf: ModuleType,
    common_attention: ModuleType,
    rng: np.random.Generator,
    shape: tuple[int, ...],
    memory_block_size: int,
) -> None:
    """Checks reshape_by_blocks against the Tensor2Tensor reference."""
    x = rng.standard_normal(size=shape).astype(dtype=np.float32)

    expected = common_attention.reshape_by_blocks(
        x=tf.constant(value=x),
        x_shape=list(shape),
        memory_block_size=memory_block_size,
    )
    actual = local_attention.reshape_by_blocks(
        x=torch.from_numpy(x),
        x_shape=list(shape),
        memory_block_size=memory_block_size,
    )

    torch.testing.assert_close(
        actual=actual,
        expected=torch.from_numpy(expected.numpy()),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    argnames=("length", "block_length"),
    argvalues=[
        pytest.param(10, 4, id="padded-blocks"),
        pytest.param(12, 4, id="whole-blocks"),
        pytest.param(8, 4, id="two-block-boundary"),
        pytest.param(5, 4, id="short-single-block"),
    ],
)
@pytest.mark.parametrize(
    argnames="block_length_type",
    argvalues=["int", "tensor"],
)
def test_masked_local_attention_1d_matches_reference(
    tf: ModuleType,
    common_attention: ModuleType,
    rng: np.random.Generator,
    length: int,
    block_length: int,
    block_length_type: str,
) -> None:
    """Checks masked_local_attention_1d against the Tensor2Tensor reference."""
    q = rng.standard_normal(size=(2, 3, length, 4)).astype(dtype=np.float32)
    k = rng.standard_normal(size=(2, 3, length, 4)).astype(dtype=np.float32)
    v = rng.standard_normal(size=(2, 3, length, 6)).astype(dtype=np.float32)
    tf_block_lengths = {
        "int": block_length,
        "tensor": tf.constant(value=block_length),
    }
    torch_block_lengths = {
        "int": block_length,
        "tensor": torch.tensor(data=block_length),
    }

    expected = common_attention.masked_local_attention_1d(
        q=tf.constant(value=q),
        k=tf.constant(value=k),
        v=tf.constant(value=v),
        block_length=tf_block_lengths[block_length_type],
        dropout_rate=0.0,
    )
    actual = local_attention.masked_local_attention_1d(
        q=torch.from_numpy(q),
        k=torch.from_numpy(k),
        v=torch.from_numpy(v),
        block_length=torch_block_lengths[block_length_type],
        dropout_rate=0.0,
    )

    torch.testing.assert_close(
        actual=actual,
        expected=torch.from_numpy(expected.numpy()),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    argnames=("length", "block_length", "filter_width"),
    argvalues=[
        pytest.param(12, 4, 3, id="partial-filter"),
        pytest.param(10, 4, 3, id="padded-partial-filter"),
        pytest.param(12, 4, 4, id="whole-block-filter"),
        pytest.param(12, 4, 5, id="filter-past-one-block"),
        pytest.param(3, 4, 3, id="short-single-block"),
    ],
)
def test_local_attention_1d_matches_reference(
    tf: ModuleType,
    common_attention: ModuleType,
    rng: np.random.Generator,
    length: int,
    block_length: int,
    filter_width: int,
) -> None:
    """Checks local_attention_1d against the Tensor2Tensor reference."""
    q = rng.standard_normal(size=(2, 3, length, 4)).astype(dtype=np.float32)
    k = rng.standard_normal(size=(2, 3, length, 4)).astype(dtype=np.float32)
    v = rng.standard_normal(size=(2, 3, length, 6)).astype(dtype=np.float32)

    expected = common_attention.local_attention_1d(
        q=tf.constant(value=q),
        k=tf.constant(value=k),
        v=tf.constant(value=v),
        block_length=block_length,
        filter_width=filter_width,
    )
    actual = local_attention.local_attention_1d(
        q=torch.from_numpy(q),
        k=torch.from_numpy(k),
        v=torch.from_numpy(v),
        block_length=block_length,
        filter_width=filter_width,
    )

    torch.testing.assert_close(
        actual=actual,
        expected=torch.from_numpy(expected.numpy()),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    argnames=("shape", "block_shape"),
    argvalues=[
        pytest.param((2, 5, 6, 3), (2, 4), id="4d-padded"),
        pytest.param((2, 4, 8, 3), (2, 4), id="4d-whole"),
        pytest.param((2, 3, 5, 6, 4), (2, 4), id="5d-padded"),
        pytest.param((2, 3, 6, 8, 4), (3, 4), id="5d-whole"),
    ],
)
def test_pad_to_multiple_2d_matches_reference(
    tf: ModuleType,
    common_attention: ModuleType,
    rng: np.random.Generator,
    shape: tuple[int, ...],
    block_shape: tuple[int, int],
) -> None:
    """Checks pad_to_multiple_2d against the Tensor2Tensor reference."""
    x = rng.standard_normal(size=shape).astype(dtype=np.float32)

    expected = common_attention.pad_to_multiple_2d(
        x=tf.constant(value=x), block_shape=block_shape
    )
    actual = local_attention.pad_to_multiple_2d(
        x=torch.from_numpy(x), block_shape=block_shape
    )

    torch.testing.assert_close(
        actual=actual,
        expected=torch.from_numpy(expected.numpy()),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    argnames=("shape", "i", "j", "new_shape"),
    argvalues=[
        pytest.param((2, 3, 4, 5, 6), 2, 4, [20], id="merge"),
        pytest.param((6, 6), 0, 1, [2, 3, 1], id="split"),
    ],
)
def test_reshape_range_matches_reference(
    tf: ModuleType,
    common_attention: ModuleType,
    rng: np.random.Generator,
    shape: tuple[int, ...],
    i: int,
    j: int,
    new_shape: list[int],
) -> None:
    """Checks reshape_range against the Tensor2Tensor reference."""
    tensor = rng.standard_normal(size=shape).astype(dtype=np.float32)

    expected = common_attention.reshape_range(
        tensor=tf.constant(value=tensor), i=i, j=j, shape=new_shape
    )
    actual = local_attention.reshape_range(
        tensor=torch.from_numpy(tensor), i=i, j=j, shape=new_shape
    )

    torch.testing.assert_close(
        actual=actual,
        expected=torch.from_numpy(expected.numpy()),
        rtol=0.0,
        atol=0.0,
    )
