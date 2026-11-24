import math

import pytest
import torch

from adlers.vision import local_attention
from tests._typing import MakeQKV


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


def test_masked_local_attention_1d_matches_pinned_tensor2tensor_behavior(
    make_qkv: MakeQKV,
) -> None:
    """Checks masked_local_attention_1d against pinned Tensor2Tensor output."""
    query, key, value = make_qkv(
        batch_size=1, num_heads=1, num_queries=7, num_keys=7, head_dim=2
    )

    output = local_attention.masked_local_attention_1d(
        q=query, k=key, v=value, block_length=2
    )

    expected_output = torch.tensor(
        data=[
            [
                [
                    [-0.21554717, -0.08586854],
                    [-0.30605689, -0.15226135],
                    [-0.09319781, -0.62043130],
                    [-0.20452517, -0.69853967],
                    [0.23387688, -0.81826800],
                    [0.52573204, -0.27424172],
                    [0.03592094, -0.03462873],
                ]
            ]
        ]
    )
    torch.testing.assert_close(actual=output, expected=expected_output)


def test_local_attention_1d_matches_pinned_tensor2tensor_behavior(
    make_qkv: MakeQKV,
) -> None:
    """Checks local_attention_1d against pinned Tensor2Tensor output."""
    query, key, value = make_qkv(
        batch_size=1, num_heads=1, num_queries=7, num_keys=7, head_dim=2
    )

    output = local_attention.local_attention_1d(
        q=query, k=key, v=value, block_length=2, filter_width=1
    )

    expected_output = torch.tensor(
        data=[
            [
                [
                    [-0.24441026, -0.37608388],
                    [-0.08625253, -0.62670010],
                    [0.06609430, -0.63146859],
                    [-0.01086021, -0.69332010],
                    [-0.19183487, -0.18216173],
                    [0.51338041, -0.05235806],
                    [-0.58736897, 0.00645295],
                ]
            ]
        ]
    )
    torch.testing.assert_close(actual=output, expected=expected_output)


def test_masked_local_attention_2d_matches_pinned_tensor2tensor_behavior(
    make_qkv: MakeQKV,
) -> None:
    """Checks masked_local_attention_2d against pinned Tensor2Tensor output."""
    query, key, value = (
        torch.reshape(input=tensor, shape=(1, 1, 3, 4, 2))
        for tensor in make_qkv(
            batch_size=1, num_heads=1, num_queries=12, num_keys=12, head_dim=2
        )
    )

    output = local_attention.masked_local_attention_2d(
        q=query, k=key, v=value, query_shape=(2, 2), memory_flange=(1, 1)
    )

    expected_output = torch.tensor(
        data=[
            [
                [
                    [
                        [0.83920944, -0.24991827],
                        [0.39961624, -0.65879899],
                        [-0.62210333, 0.27947396],
                        [1.33683729, -0.68064010],
                    ],
                    [
                        [0.10802013, -0.49167177],
                        [-0.47792137, 0.04784281],
                        [-0.83967602, 0.31098750],
                        [0.40267393, -0.38121152],
                    ],
                    [
                        [-0.89572525, 0.34429777],
                        [-1.02716184, 0.10172251],
                        [-0.90844190, 0.08513299],
                        [-0.33734003, 0.31594053],
                    ],
                ]
            ]
        ]
    )
    torch.testing.assert_close(actual=output, expected=expected_output)


def test_local_attention_2d_matches_pinned_tensor2tensor_behavior(
    make_qkv: MakeQKV,
) -> None:
    """Checks local_attention_2d against pinned Tensor2Tensor output."""
    query, key, value = (
        torch.reshape(input=tensor, shape=(1, 1, 3, 4, 2))
        for tensor in make_qkv(
            batch_size=1, num_heads=1, num_queries=12, num_keys=12, head_dim=2
        )
    )

    output = local_attention.local_attention_2d(
        q=query, k=key, v=value, query_shape=(2, 2), memory_flange=(1, 1)
    )

    expected_output = torch.tensor(
        data=[
            [
                [
                    [
                        [-0.22318561, -0.17221767],
                        [-0.18481880, -0.31228179],
                        [-0.61605394, 0.27874219],
                        [0.76880759, -0.20558973],
                    ],
                    [
                        [-0.13864893, -0.34366229],
                        [0.21209238, -0.26084015],
                        [-0.80136800, 0.23914815],
                        [0.33868724, -0.30874729],
                    ],
                    [
                        [-0.06904912, -0.29286635],
                        [-0.80501366, 0.00466618],
                        [-0.83411455, 0.14890090],
                        [-0.33734000, 0.31594047],
                    ],
                ]
            ]
        ]
    )
    torch.testing.assert_close(actual=output, expected=expected_output)
