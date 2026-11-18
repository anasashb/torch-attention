import math

import pytest
import torch

from adlers.vision import local_attention


@pytest.mark.parametrize(
    argnames=("length", "block_length"),
    argvalues=[
        pytest.param(10, 4, id="padded-blocks"),
        pytest.param(5, 4, id="short-single-block"),
    ],
)
def test_masked_local_attention_1d_ignores_later_positions(
    length: int,
    block_length: int,
) -> None:
    """Checks that no 1D output depends on a later position."""
    generator = torch.Generator().manual_seed(66)
    shape = (2, 3, length, 4)
    q, k, v = (torch.randn(size=shape, generator=generator) for _ in range(3))
    output = local_attention.masked_local_attention_1d(
        q=q, k=k, v=v, block_length=block_length
    )

    for position in range(length):
        changed_q, changed_k, changed_v = (
            tensor.clone() for tensor in (q, k, v)
        )
        for changed in (changed_q, changed_k, changed_v):
            changed[:, :, position] = torch.randn(
                size=(2, 3, 4), generator=generator
            )
        changed_output = local_attention.masked_local_attention_1d(
            q=changed_q, k=changed_k, v=changed_v, block_length=block_length
        )

        torch.testing.assert_close(
            actual=changed_output[:, :, :position],
            expected=output[:, :, :position],
            rtol=0.0,
            atol=0.0,
        )
        assert not torch.equal(
            input=changed_output[:, :, position],
            other=output[:, :, position],
        )


@pytest.mark.parametrize(
    argnames="memory_flange",
    argvalues=[
        pytest.param((1, 2), id="top-and-left"),
        pytest.param((0, 0), id="no-flange"),
    ],
)
@pytest.mark.parametrize(
    argnames="grid_shape",
    argvalues=[
        pytest.param((4, 6), id="whole-grid"),
        pytest.param((3, 5), id="padded-grid"),
    ],
)
def test_masked_local_attention_2d_ignores_later_positions(
    grid_shape: tuple[int, int],
    memory_flange: tuple[int, int],
) -> None:
    """Checks that no 2D output depends on a later position.

    Positions are generated block by block in raster order, and in raster
    order within each block, so "later" follows that order rather than plain
    raster order over the whole grid.
    """
    query_height, query_width = 2, 3
    height, width = grid_shape
    generator = torch.Generator().manual_seed(66)
    shape = (2, 3, height, width, 4)
    q, k, v = (torch.randn(size=shape, generator=generator) for _ in range(3))
    output = local_attention.masked_local_attention_2d(
        q=q,
        k=k,
        v=v,
        query_shape=(query_height, query_width),
        memory_flange=memory_flange,
    )

    blocks_per_row = math.ceil(width / query_width)
    rows = torch.arange(end=height)[:, None]
    cols = torch.arange(end=width)[None, :]
    block_index = (rows // query_height) * blocks_per_row + cols // query_width
    index_in_block = (rows % query_height) * query_width + cols % query_width
    generation_rank = block_index * query_height * query_width + index_in_block

    for row in range(height):
        for col in range(width):
            changed_q, changed_k, changed_v = (
                tensor.clone() for tensor in (q, k, v)
            )
            for changed in (changed_q, changed_k, changed_v):
                changed[:, :, row, col] = torch.randn(
                    size=(2, 3, 4), generator=generator
                )
            changed_output = local_attention.masked_local_attention_2d(
                q=changed_q,
                k=changed_k,
                v=changed_v,
                query_shape=(query_height, query_width),
                memory_flange=memory_flange,
            )
            earlier = generation_rank < generation_rank[row, col]

            torch.testing.assert_close(
                actual=changed_output[:, :, earlier],
                expected=output[:, :, earlier],
                rtol=0.0,
                atol=0.0,
            )
            assert not torch.equal(
                input=changed_output[:, :, row, col],
                other=output[:, :, row, col],
            )
