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
