import numpy as np
import pytest
import torch

from adlers.vision import local_attention

tf = pytest.importorskip(modname="tensorflow")
tf.config.set_visible_devices(devices=[], device_type="GPU")

from tests.reference.tensor2tensor import common_attention  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    """Returns a NumPy generator for inputs shared by both frameworks."""
    return np.random.default_rng(seed=66)


@pytest.mark.parametrize(
    argnames="shape",
    argvalues=[
        pytest.param((2, 5, 4), id="sequence"),
        pytest.param((2, 3, 2, 4, 6), id="blocked"),
    ],
)
def test_embedding_to_padding_matches_reference(
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
