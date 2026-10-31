# Adapted from Tensor2Tensor:
# https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/layers/common_attention.py
# https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/layers/common_layers.py
#
# Licensed under the Apache License, Version 2.0.
# This file has been modified for ADLERS.
# See LICENSES/Apache-2.0.txt and NOTICE.
#
# coding=utf-8
# Copyright 2023 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for attention."""

import functools
import operator

import numpy as np
import torch
from torch.nn import functional as F


def shape_list(x):
    """Return list of dims, statically where possible."""
    return list(x.shape)


def ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
    """Matrix band part of ones.

    Args:
      rows: int determining number of rows in output
      cols: int
      num_lower: int, maximum distance backward. Negative values indicate
        unlimited.
      num_upper: int, maximum distance forward. Negative values indicate
        unlimited.
      out_shape: shape to reshape output by.

    Returns:
      Tensor of size rows * cols reshaped into shape out_shape.
    """
    if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
        # Needed info is constant, so we construct in numpy
        if num_lower < 0:
            num_lower = rows - 1
        if num_upper < 0:
            num_upper = cols - 1
        lower_mask = np.tri(cols, rows, num_lower).T
        upper_mask = np.tri(rows, cols, num_upper)
        band = np.ones((rows, cols)) * lower_mask * upper_mask
        if out_shape:
            band = band.reshape(out_shape)
        band = torch.tensor(data=band, dtype=torch.float32)
    else:
        row_index = torch.arange(end=rows)[:, None]
        col_index = torch.arange(end=cols)[None, :]
        in_lower = (num_lower < 0) | (row_index - col_index <= num_lower)
        in_upper = (num_upper < 0) | (col_index - row_index <= num_upper)
        band = (in_lower & in_upper).to(dtype=torch.float32)
        if out_shape:
            band = torch.reshape(input=band, shape=out_shape)

    return band


def embedding_to_padding(emb):
    """Calculates the padding mask based on which embeddings are all zero.

    We have hacked symbol_modality to return all-zero embeddings for padding.

    Args:
      emb: a Tensor with shape [..., depth].

    Returns:
      a float Tensor with shape [...]. Each element is 1 if its corresponding
      embedding vector is all zero, and is 0 otherwise.
    """
    emb_sum = torch.sum(input=torch.abs(input=emb), dim=-1)
    return torch.eq(input=emb_sum, other=0.0).to(dtype=torch.float32)


def attention_bias_local(length, max_backward, max_forward):
    """Create an bias tensor to be added to attention logits.

    A position may attend to positions at most max_distance from it,
    forward and backwards.

    This does not actually save any computation.

    Args:
      length: int
      max_backward: int, maximum distance backward to attend. Negative values
        indicate unlimited.
      max_forward: int, maximum distance forward to attend. Negative values
        indicate unlimited.

    Returns:
      a `Tensor` with shape [1, 1, length, length].
    """
    band = ones_matrix_band_part(
        rows=length,
        cols=length,
        num_lower=max_backward,
        num_upper=max_forward,
        out_shape=[1, 1, length, length],
    )
    return -1e9 * (1.0 - band)


def attention_bias_lower_triangle(length):
    """Create an bias tensor to be added to attention logits.

    Allows a query to attend to all positions up to and including its own.

    Args:
     length: a Scalar.

    Returns:
      a `Tensor` with shape [1, 1, length, length].
    """
    return attention_bias_local(length=length, max_backward=-1, max_forward=0)


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.

    The first of these two dimensions is n.

    Args:
      x: a Tensor with shape [..., m]
      n: an integer.

    Returns:
      a Tensor with shape [..., n, m/n]
    """
    x_shape = common_layers.shape_list(x)
    m = x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
        assert m % n == 0
    return tf.reshape(x, x_shape[:-1] + [n, m // n])


def dot_product_attention(
    q,
    k,
    v,
    bias,
    dropout_rate=0.0,
):
    """Dot-product attention.

    Args:
      q: Tensor with shape [..., length_q, depth_k].
      k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
        match with q.
      v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
        match with q.
      bias: bias Tensor (see attention_bias())
      dropout_rate: a float.

    Returns:
      Tensor with shape [..., length_q, depth_v].
    """
    logits = torch.matmul(input=q, other=k.mT)  # [..., length_q, length_kv]
    if bias is not None:
        bias = bias.to(dtype=logits.dtype)
        logits = logits + bias
    weights = torch.softmax(input=logits, dim=-1)
    weights = weights.to(dtype=q.dtype)
    # Drop out attention links for each head.
    weights = F.dropout(input=weights, p=dropout_rate)
    return torch.matmul(input=weights, other=v)


def masked_local_attention_1d(
    q,
    k,
    v,
    block_length=128,
    dropout_rate=0.0,
):
    """Attention to the source position and a neighborhood to the left of it.

    The sequence is divided into blocks of length block_length. Attention for a
    given query position can only see memory positions less than or equal to the
    query position, in the corresponding block and the previous block.

    Args:
      q: a Tensor with shape [batch, heads, length, depth_k]
      k: a Tensor with shape [batch, heads, length, depth_k]
      v: a Tensor with shape [batch, heads, length, depth_v]
      block_length: an integer
      dropout_rate: Dropout rate for attention dropout

    Returns:
      a Tensor of shape [batch, heads, length, depth_v]
    """
    batch, heads, length, depth_k = shape_list(q)
    depth_v = shape_list(v)[-1]
    if isinstance(block_length, torch.Tensor):
        block_length = int(block_length)
    # If (length < 2 * block_length), then we use only one block.
    block_length = length if length < block_length * 2 else block_length

    # Pad query, key, value to ensure multiple of block length.
    original_length = length
    padding_size = -length % block_length
    length += padding_size
    padding = (0, 0, 0, padding_size)
    q = F.pad(input=q, pad=padding)
    k = F.pad(input=k, pad=padding)
    v = F.pad(input=v, pad=padding)

    num_blocks = length // block_length

    # Compute attention for the first query block.
    first_q = q[:, :, :block_length]
    first_k = k[:, :, :block_length]
    first_v = v[:, :, :block_length]

    first_output = dot_product_attention(
        q=first_q,
        k=first_k,
        v=first_v,
        bias=attention_bias_lower_triangle(length=block_length),
        dropout_rate=dropout_rate,
    )

    # Compute attention for all subsequent query blocks.
    q = torch.reshape(
        input=q, shape=[batch, heads, num_blocks, block_length, depth_k]
    )
    k = torch.reshape(
        input=k, shape=[batch, heads, num_blocks, block_length, depth_k]
    )
    v = torch.reshape(
        input=v, shape=[batch, heads, num_blocks, block_length, depth_v]
    )

    local_k = _make_local_block(
        x=k,
        depth=depth_k,
        batch=batch,
        heads=heads,
        num_blocks=num_blocks,
        block_length=block_length,
    )
    local_v = _make_local_block(
        x=v,
        depth=depth_v,
        batch=batch,
        heads=heads,
        num_blocks=num_blocks,
        block_length=block_length,
    )
    tail_q = q[:, :, 1:]
    tail_q = torch.reshape(
        input=tail_q,
        shape=[batch, heads, num_blocks - 1, block_length, depth_k],
    )
    local_length = shape_list(local_k)[3]

    # make sure source_pos <= target_pos
    good_part = ones_matrix_band_part(
        rows=block_length,
        cols=local_length,
        num_lower=-1,
        num_upper=block_length,
        out_shape=[1, 1, 1, block_length, local_length],
    )
    bias = (1.0 - good_part) * -1e9
    # TODO(noam): figure out how to show a summary for the remaining blocks.
    # The naive way currently causes errors due to empty tensors.
    # output: [batch, heads, num_blocks-1, block_length, depth_v]
    tail_output = dot_product_attention(
        q=tail_q,
        k=local_k,
        v=local_v,
        bias=bias,
        dropout_rate=dropout_rate,
    )
    tail_output = torch.reshape(
        input=tail_output,
        shape=[batch, heads, (num_blocks - 1) * block_length, depth_v],
    )
    output = torch.cat(tensors=[first_output, tail_output], dim=2)

    # Remove the padding if introduced.
    output = output[:, :, :original_length]
    output = torch.reshape(
        input=output, shape=[batch, heads, original_length, depth_v]
    )
    return output


def _make_local_block(x, depth, batch, heads, num_blocks, block_length):
    """Helper function to create a local version of the keys or values for 1d."""
    prev_block = x[:, :, : num_blocks - 1]
    cur_block = x[:, :, 1:]
    local_block = torch.cat(tensors=[prev_block, cur_block], dim=3)
    return torch.reshape(
        input=local_block,
        shape=[batch, heads, num_blocks - 1, block_length * 2, depth],
    )


def local_attention_1d(q, k, v, block_length=128, filter_width=100):
    """Strided block local self-attention.

    The sequence is divided into blocks of length block_length. Attention for a
    given query position can see all memory positions in the corresponding block
    and filter_width many positions to the left and right of the block.

    Args:
      q: a Tensor with shape [batch, heads, length, depth_k]
      k: a Tensor with shape [batch, heads, length, depth_k]
      v: a Tensor with shape [batch, heads, length, depth_v]
      block_length: an integer
      filter_width: an integer indicating how much to look left and right of the
        block.

    Returns:
      a Tensor of shape [batch, heads, length, depth_v]
    """
    # Check that q, k, v have the same shape except in their depth dimension.
    if q.shape[:-1] != k.shape[:-1]:
        raise ValueError(
            f"Shapes {tuple(q.shape[:-1])} and {tuple(k.shape[:-1])} are"
            " incompatible"
        )
    if q.shape[:-1] != v.shape[:-1]:
        raise ValueError(
            f"Shapes {tuple(q.shape[:-1])} and {tuple(v.shape[:-1])} are"
            " incompatible"
        )

    batch_size, num_heads, original_length, _ = shape_list(q)

    # Pad query, key, value to ensure multiple of corresponding lengths.
    def pad_to_multiple(x, pad_length):
        x_length = shape_list(x)[2]
        return F.pad(input=x, pad=(0, 0, 0, -x_length % pad_length))

    def pad_l_and_r(x, pad_length):
        return F.pad(input=x, pad=(0, 0, pad_length, pad_length))

    # Set up query blocks.
    # [batch, heads, blocks_q, block_length, depth_k]
    q = pad_to_multiple(x=q, pad_length=block_length)
    q = reshape_by_blocks(
        x=q, x_shape=shape_list(q), memory_block_size=block_length
    )
    total_query_blocks = shape_list(q)[2]

    # Set up key and value blocks.
    # [batch, heads, blocks_k, block_length, depth_k]
    blocks_per_filter_width = filter_width // block_length
    remaining_items = filter_width % block_length
    k = pad_to_multiple(x=k, pad_length=block_length)
    v = pad_to_multiple(x=v, pad_length=block_length)
    k = pad_l_and_r(
        x=k, pad_length=filter_width + block_length - remaining_items
    )
    v = pad_l_and_r(
        x=v, pad_length=filter_width + block_length - remaining_items
    )
    k = reshape_by_blocks(
        x=k, x_shape=shape_list(k), memory_block_size=block_length
    )
    v = reshape_by_blocks(
        x=v, x_shape=shape_list(v), memory_block_size=block_length
    )

    total_kv_blocks = shape_list(k)[2]

    slices = []
    # prepare the left-most and right-most partial blocks if needed
    if remaining_items:
        first_partial_block_k = k[
            :, :, :total_query_blocks, block_length - remaining_items :
        ]
        first_partial_block_v = v[
            :, :, :total_query_blocks, block_length - remaining_items :
        ]
        last_partial_block_k = k[
            :, :, total_kv_blocks - total_query_blocks :, :remaining_items
        ]
        last_partial_block_v = v[
            :, :, total_kv_blocks - total_query_blocks :, :remaining_items
        ]
        slices.append((first_partial_block_k, first_partial_block_v))
        slices.append((last_partial_block_k, last_partial_block_v))

    # Prepare the rest of the blocks
    first_block_index = 1 if remaining_items else 0
    attention_blocks = 2 * blocks_per_filter_width + 1
    for i in range(first_block_index, attention_blocks + first_block_index):
        block_k = k[:, :, i : i + total_query_blocks]
        block_v = v[:, :, i : i + total_query_blocks]
        slices.append((block_k, block_v))
    # [batch, heads, blocks_q, block_length + 2 * filter_width, depth_k]
    k = torch.cat(tensors=[s[0] for s in slices], dim=3)
    v = torch.cat(tensors=[s[1] for s in slices], dim=3)

    attention_bias = torch.unsqueeze(
        input=embedding_to_padding(emb=k) * -1e9, dim=-2
    )
    depth_v = shape_list(v)[-1]

    output = dot_product_attention(
        q=q,
        k=k,
        v=v,
        bias=attention_bias,
        dropout_rate=0.0,
    )
    output = torch.reshape(
        input=output, shape=[batch_size, num_heads, -1, depth_v]
    )

    # Remove the padding if introduced.
    output = output[:, :, :original_length]
    return output


def reshape_by_blocks(x, x_shape, memory_block_size):
    """Reshapes input by splitting its length over blocks of memory_block_size.

    Args:
      x: a Tensor with shape [batch, heads, length, depth]
      x_shape: tf.TensorShape of x.
      memory_block_size: Integer which divides length.

    Returns:
      Tensor with shape
      [batch, heads, length // memory_block_size, memory_block_size, depth].
    """
    x = torch.reshape(
        input=x,
        shape=[
            x_shape[0],
            x_shape[1],
            x_shape[2] // memory_block_size,
            memory_block_size,
            x_shape[3],
        ],
    )
    return x


def local_attention_2d(q, k, v, query_shape=(8, 16), memory_flange=(8, 16)):
    """Strided block local self-attention.

    The 2-D sequence is divided into 2-D blocks of shape query_shape. Attention
    for a given query position can only see memory positions less than or equal to
    the query position. The memory positions are the corresponding block with
    memory_flange many positions to add to the height and width of the block
    (namely, left, top, and right).

    Args:
      q: a Tensor with shape [batch, heads, h, w, depth_k]
      k: a Tensor with shape [batch, heads, h, w, depth_k]
      v: a Tensor with shape [batch, heads, h, w, depth_v]. In the current
        implementation, depth_v must be equal to depth_k.
      query_shape: an tuple indicating the height and width of each query block.
      memory_flange: an integer indicating how much to look in height and width
        from each query block.

    Returns:
      a Tensor of shape [batch, heads, h, w, depth_v]
    """
    v_shape = common_layers.shape_list(v)

    # Pad query, key, value to ensure multiple of corresponding lengths.
    q = pad_to_multiple_2d(q, query_shape)
    k = pad_to_multiple_2d(k, query_shape)
    v = pad_to_multiple_2d(v, query_shape)
    paddings = [
        [0, 0],
        [0, 0],
        [memory_flange[0], memory_flange[1]],
        [memory_flange[0], memory_flange[1]],
        [0, 0],
    ]
    k = tf.pad(k, paddings)
    v = tf.pad(v, paddings)

    # Set up query blocks.
    q_indices = gather_indices_2d(q, query_shape, query_shape)
    q_new = gather_blocks_2d(q, q_indices)

    # Set up key and value blocks.
    memory_shape = (
        query_shape[0] + 2 * memory_flange[0],
        query_shape[1] + 2 * memory_flange[1],
    )
    k_and_v_indices = gather_indices_2d(k, memory_shape, query_shape)
    k_new = gather_blocks_2d(k, k_and_v_indices)
    v_new = gather_blocks_2d(v, k_and_v_indices)

    attention_bias = tf.expand_dims(
        tf.to_float(embedding_to_padding(k_new)) * -1e9, axis=-2
    )
    output = dot_product_attention(
        q_new,
        k_new,
        v_new,
        attention_bias,
        dropout_rate=0.0,
    )
    # Put representations back into original shapes.
    padded_q_shape = common_layers.shape_list(q)
    output = scatter_blocks_2d(output, q_indices, padded_q_shape)

    # Remove the padding if introduced.
    output = tf.slice(
        output, [0, 0, 0, 0, 0], [-1, -1, v_shape[2], v_shape[3], -1]
    )
    return output


def pad_to_multiple_2d(x, block_shape):
    """Making sure x is a multiple of shape.

    Args:
      x: a [batch, heads, h, w, depth] or [batch, h, w, depth] tensor
      block_shape: a 2-d list of integer shapes

    Returns:
      padded_x: a [batch, heads, h, w, depth] or [batch, h, w, depth] tensor
    """
    old_shape = list(x.shape)
    if len(old_shape) == 4:
        height_padding = -shape_list(x)[1] % block_shape[0]
        width_padding = -shape_list(x)[2] % block_shape[1]
        paddings = (0, 0, 0, width_padding, 0, height_padding, 0, 0)
    elif len(old_shape) == 5:
        height_padding = -shape_list(x)[2] % block_shape[0]
        width_padding = -shape_list(x)[3] % block_shape[1]
        paddings = (0, 0, 0, width_padding, 0, height_padding, 0, 0, 0, 0)

    padded_x = F.pad(input=x, pad=paddings)
    return padded_x


def reshape_range(tensor, i, j, shape):
    """Reshapes a tensor between dimensions i and j."""
    t_shape = shape_list(x=tensor)
    target_shape = t_shape[:i] + shape + t_shape[j:]
    return torch.reshape(input=tensor, shape=target_shape)


def gather_blocks_2d(x, indices):
    """Gathers flattened blocks from x."""
    x_shape = common_layers.shape_list(x)
    x = reshape_range(x, 2, 4, [tf.reduce_prod(x_shape[2:4])])
    # [length, batch, heads, dim]
    x_t = tf.transpose(x, [2, 0, 1, 3])
    x_new = tf.gather(x_t, indices)
    # returns [batch, heads, num_blocks, block_length ** 2, dim]
    return tf.transpose(x_new, [2, 3, 0, 1, 4])


def scatter_blocks_2d(x, indices, shape):
    """scatters blocks from x into shape with indices."""
    x_shape = common_layers.shape_list(x)
    # [length, batch, heads, dim]
    x_t = tf.transpose(
        tf.reshape(x, [x_shape[0], x_shape[1], -1, x_shape[-1]]), [2, 0, 1, 3]
    )
    x_t_shape = common_layers.shape_list(x_t)
    indices = tf.reshape(indices, [-1, 1])
    scattered_x = tf.scatter_nd(indices, x_t, x_t_shape)
    scattered_x = tf.transpose(scattered_x, [1, 2, 0, 3])
    return tf.reshape(scattered_x, shape)


def gather_indices_2d(x, block_shape, block_stride):
    """Getting gather indices."""
    # making an identity matrix kernel
    kernel = tf.eye(block_shape[0] * block_shape[1])
    kernel = reshape_range(kernel, 0, 1, [block_shape[0], block_shape[1], 1])
    # making indices [1, h, w, 1] to appy convs
    x_shape = common_layers.shape_list(x)
    indices = tf.range(x_shape[2] * x_shape[3])
    indices = tf.reshape(indices, [1, x_shape[2], x_shape[3], 1])
    indices = tf.nn.conv2d(
        tf.cast(indices, tf.float32),
        kernel,
        strides=[1, block_stride[0], block_stride[1], 1],
        padding="VALID",
    )
    # making indices [num_blocks, dim] to gather
    dims = common_layers.shape_list(indices)[:3]
    if all([isinstance(dim, int) for dim in dims]):
        num_blocks = functools.reduce(operator.mul, dims, 1)
    else:
        num_blocks = tf.reduce_prod(dims)
    indices = tf.reshape(indices, [num_blocks, -1])
    return tf.cast(indices, tf.int32)


def get_memory_region(x, query_block_shape, memory_flange, q_indices):
    """Get the memory regions that surround a 2d query.

      The memory regions will be the left and top right.

    Args:
      x: A tensor with shape [batch, heads, height, width, depth]
      query_block_shape: a 2-d tuple of integers
      memory_flange: a 2-d tuple of integers
      q_indices: a tensor of indices for each of the center blocks.
        [num_blocks, block_length]
    Returns:
      x_flange: A tensor of shape [batch, heads, #blocks, block_length, depth]
    """
    # Padding x to be multiple of query_shape and then
    # extracting the memory blocks from the same regions as the query blocks
    x_query_padded = pad_to_multiple_2d(x, query_block_shape)
    x_center = gather_blocks_2d(x_query_padded, q_indices)
    # Then padding the flange region
    paddings = [
        [0, 0],
        [0, 0],
        [memory_flange[0], 0],
        [memory_flange[1], memory_flange[1]],
        [0, 0],
    ]
    x_memory_padded = tf.pad(x_query_padded, paddings)
    left_x = None
    top_x = None
    # Extracting the memory regions around the query block. left_x_region extends
    # to the left and the top_x_region is the combination of top left, top, and
    # top right of the query block
    # if no left region
    if memory_flange[1] > 0:
        left_x_region = x_memory_padded[
            :,
            :,
            memory_flange[0] :,
            : -(query_block_shape[1] + memory_flange[1]),
            :,
        ]
        left_memory_shape = (query_block_shape[0], memory_flange[1])
        left_indices = gather_indices_2d(
            left_x_region, left_memory_shape, query_block_shape
        )
        left_x = gather_blocks_2d(left_x_region, left_indices)
    # if no top region
    if memory_flange[0] > 0:
        top_x_region = x_memory_padded[:, :, : -query_block_shape[0], :, :]

        top_memory_shape = (
            memory_flange[0],
            query_block_shape[1] + 2 * memory_flange[1],
        )

        top_indices = gather_indices_2d(
            top_x_region, top_memory_shape, query_block_shape
        )

        top_x = gather_blocks_2d(top_x_region, top_indices)
    x_flange = None
    if top_x is not None and left_x is not None:
        x_flange = tf.concat([top_x, left_x], axis=3)
    else:
        x_flange = top_x if top_x is not None else left_x
    return x_flange, x_center


def masked_local_attention_2d(
    q, k, v, query_shape=(8, 16), memory_flange=(8, 16)
):
    """Strided block local self-attention.

    Each position in a query block can attend to all the generated queries in
    the query block, which are generated in raster scan, and positions that are
    generated to the left and top. The shapes are specified by query shape and
    memory flange. Note that if you're using this function, you do not need to
    right shift. Right shifting happens inside this function separately for each
    block.

    Args:
      q: a Tensor with shape [batch, heads, h, w, depth_k]
      k: a Tensor with shape [batch, heads, h, w, depth_k]
      v: a Tensor with shape [batch, heads, h, w, depth_v]. In the current
        implementation, depth_v must be equal to depth_k.
      query_shape: an tuple indicating the height and width of each query block.
        query_shape = block_shape
      memory_flange: an integer indicating how much to look in height and width
        from each query block.
        memory shape = query_shape + (block_flange[0], 2*block_flange[1])

    Returns:
      a Tensor of shape [batch, heads, h, w, depth_v]
    """
    v_shape = common_layers.shape_list(v)

    # Pad query to ensure multiple of corresponding lengths.
    q = pad_to_multiple_2d(q, query_shape)

    # Set up query blocks.
    q_indices = gather_indices_2d(q, query_shape, query_shape)
    q_new = gather_blocks_2d(q, q_indices)

    # Set up key and value blocks.
    k_flange, k_center = get_memory_region(
        k, query_shape, memory_flange, q_indices
    )
    v_flange, v_center = get_memory_region(
        v, query_shape, memory_flange, q_indices
    )
    if k_flange is not None:
        k_new = tf.concat([k_flange, k_center], axis=3)
        v_new = tf.concat([v_flange, v_center], axis=3)
    else:
        k_new = k_center
        v_new = v_center

    # Set up the masks.
    query_elements = np.prod(query_shape)
    padding_mask = None
    if k_flange is not None:
        padding_mask = tf.expand_dims(
            embedding_to_padding(k_flange) * -1e9, axis=-2
        )
        padding_mask = tf.tile(padding_mask, [1, 1, 1, query_elements, 1])

    center_attention_bias = attention_bias_lower_triangle(
        np.prod(query_elements)
    )
    center_attention_bias = tf.reshape(
        center_attention_bias, [1, 1, 1, query_elements, query_elements]
    )
    v_center_shape = common_layers.shape_list(v_center)
    center_attention_bias = tf.tile(
        center_attention_bias,
        [v_center_shape[0], v_center_shape[1], v_center_shape[2], 1, 1],
    )
    if padding_mask is not None:
        # Combine the mask for padding and visible region.
        attention_bias = tf.concat(
            [padding_mask, center_attention_bias], axis=4
        )
    else:
        attention_bias = center_attention_bias

    output = dot_product_attention(
        q_new,
        k_new,
        v_new,
        attention_bias,
        dropout_rate=0.0,
    )
    # Put representations back into original shapes.
    padded_q_shape = common_layers.shape_list(q)
    output = scatter_blocks_2d(output, q_indices, padded_q_shape)

    # Remove the padding if introduced.
    output = tf.slice(
        output, [0, 0, 0, 0, 0], [-1, -1, v_shape[2], v_shape[3], -1]
    )
    return output
