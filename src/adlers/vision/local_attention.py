# Adapted from Tensor2Tensor:
# https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/layers/common_attention.py
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


def mixed_precision_is_enabled(
    activation_dtype=None, weight_dtype=None, hparams=None
):
    assert not (
        hparams and (activation_dtype or weight_dtype)
    ), "Provide only hparams or activation_dtype and weight_dtype"
    if (
        hparams
        and hasattr(hparams, "activation_dtype")
        and hasattr(hparams, "weight_dtype")
    ):
        activation_dtype = hparams.activation_dtype
        weight_dtype = hparams.weight_dtype
    return activation_dtype == tf.float16 and weight_dtype == tf.float32


def maybe_upcast(
    logits, activation_dtype=None, weight_dtype=None, hparams=None
):
    if mixed_precision_is_enabled(activation_dtype, weight_dtype, hparams):
        return tf.cast(logits, tf.float32)
    return logits


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
    band = common_layers.ones_matrix_band_part(
        length,
        length,
        max_backward,
        max_forward,
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
    return attention_bias_local(length, -1, 0)


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


def attention_image_summary(attn, image_shapes=None):
    """Compute color image summary.

    Args:
      attn: a Tensor with shape [batch, num_heads, query_length, memory_length]
      image_shapes: optional tuple of integer scalars.
        If the query positions and memory positions represent the
        pixels of flattened images, then pass in their dimensions:
          (query_rows, query_cols, memory_rows, memory_cols).
        If the query positions and memory positions represent the
        pixels x channels of flattened images, then pass in their dimensions:
          (query_rows, query_cols, query_channels,
           memory_rows, memory_cols, memory_channels).
    """
    attn = tf.cast(attn, tf.float32)
    num_heads = common_layers.shape_list(attn)[1]
    # [batch, query_length, memory_length, num_heads]
    image = tf.transpose(attn, [0, 2, 3, 1])
    image = tf.pow(image, 0.2)  # for high-dynamic-range
    # Each head will correspond to one of RGB.
    # pad the heads to be a multiple of 3
    image = tf.pad(image, [[0, 0], [0, 0], [0, 0], [0, tf.mod(-num_heads, 3)]])
    image = split_last_dimension(image, 3)
    image = tf.reduce_max(image, 4)
    if image_shapes is not None:
        if len(image_shapes) == 4:
            q_rows, q_cols, m_rows, m_cols = list(image_shapes)
            image = tf.reshape(image, [-1, q_rows, q_cols, m_rows, m_cols, 3])
            image = tf.transpose(image, [0, 1, 3, 2, 4, 5])
            image = tf.reshape(image, [-1, q_rows * m_rows, q_cols * m_cols, 3])
        else:
            assert len(image_shapes) == 6
            q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels = list(
                image_shapes
            )
            image = tf.reshape(
                image,
                [
                    -1,
                    q_rows,
                    q_cols,
                    q_channnels,
                    m_rows,
                    m_cols,
                    m_channels,
                    3,
                ],
            )
            image = tf.transpose(image, [0, 1, 4, 3, 2, 5, 6, 7])
            image = tf.reshape(
                image,
                [
                    -1,
                    q_rows * m_rows * q_channnels,
                    q_cols * m_cols * m_channels,
                    3,
                ],
            )
    tf.summary.image("attention", image, max_outputs=1)


def harden_attention_weights(weights, k, gumbel_noise_weight):
    """Make attention weights non-0 only on the top k ones."""
    if gumbel_noise_weight > 0.0:
        gumbel_noise = -tf.log(
            -tf.log(
                tf.random_uniform(
                    tf.shape(weights), minval=1e-5, maxval=1 - 1e-5
                )
            )
        )
        weights += gumbel_noise * gumbel_noise_weight

    # Subtract the top-kth weight and zero-out all lower ones.
    # Note that currently in case of numerical ties it will retain more
    # than k elements. In the future, we may want to avoid this.
    weights -= common_layers.top_kth_iterative(weights, k)
    weights = tf.nn.relu(weights)
    # Re-normalize the weights.
    weights_sum = tf.reduce_sum(weights, axis=-1, keep_dims=True)
    weights_sum = tf.maximum(weights_sum, 1e-6)  # Avoid division by 0.
    weights /= weights_sum
    return weights


def dot_product_attention(
    q,
    k,
    v,
    bias,
    dropout_rate=0.0,
    image_shapes=None,
    name=None,
    make_image_summary=True,
    save_weights_to=None,
    dropout_broadcast_dims=None,
    activation_dtype=None,
    weight_dtype=None,
    hard_attention_k=0,
    gumbel_noise_weight=0.0,
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
      image_shapes: optional tuple of integer scalars.
        see comments for attention_image_summary()
      name: an optional string
      make_image_summary: True if you want an image summary.
      save_weights_to: an optional dictionary to capture attention weights
        for visualization; the weights tensor will be appended there under
        a string key created from the variable scope (including name).
      dropout_broadcast_dims: an optional list of integers less than rank of q.
        Specifies in which dimensions to broadcast the dropout decisions.
      activation_dtype: Used to define function activation dtype when using
        mixed precision.
      weight_dtype: The dtype weights are stored in when using mixed precision
      hard_attention_k: integer, if > 0 triggers hard attention (picking top-k)
      gumbel_noise_weight: if > 0, apply Gumbel noise with weight
        `gumbel_noise_weight` before picking top-k. This is a no op if
        hard_attention_k <= 0.

    Returns:
      Tensor with shape [..., length_q, depth_v].
    """
    with tf.variable_scope(
        name, default_name="dot_product_attention", values=[q, k, v]
    ) as scope:
        logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
        if bias is not None:
            bias = common_layers.cast_like(bias, logits)
            logits += bias
        # If logits are fp16, upcast before softmax
        logits = maybe_upcast(logits, activation_dtype, weight_dtype)
        weights = tf.nn.softmax(logits, name="attention_weights")
        if hard_attention_k > 0:
            weights = harden_attention_weights(
                weights, hard_attention_k, gumbel_noise_weight
            )
        weights = common_layers.cast_like(weights, q)
        if save_weights_to is not None:
            save_weights_to[scope.name] = weights
            save_weights_to[scope.name + "/logits"] = logits
        # Drop out attention links for each head.
        weights = common_layers.dropout_with_broadcast_dims(
            weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims
        )
        if common_layers.should_generate_summaries() and make_image_summary:
            attention_image_summary(weights, image_shapes)
        return tf.matmul(weights, v)


def masked_local_attention_1d(
    q,
    k,
    v,
    block_length=128,
    make_image_summary=False,
    dropout_rate=0.0,
    name=None,
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
      make_image_summary: a boolean, whether to make an attention image summary.
      dropout_rate: Dropout rate for attention dropout
      name: an optional string

    Returns:
      a Tensor of shape [batch, heads, length, depth_v]
    """
    with tf.variable_scope(
        name, default_name="local_attention_1d", values=[q, k, v]
    ):
        batch, heads, length, depth_k = common_layers.shape_list(q)
        depth_v = common_layers.shape_list(v)[-1]
        if isinstance(block_length, tf.Tensor):
            const = contrib.util().constant_value(block_length)
            if const is not None:
                block_length = int(const)
        # If (length < 2 * block_length), then we use only one block.
        if isinstance(length, int) and isinstance(block_length, int):
            block_length = length if length < block_length * 2 else block_length
        else:
            block_length = tf.where(
                tf.less(length, block_length * 2), length, block_length
            )

        # Pad query, key, value to ensure multiple of block length.
        original_length = length
        padding_size = tf.mod(-length, block_length)
        length += padding_size
        padding = [[0, 0], [0, 0], [0, padding_size], [0, 0]]
        q = tf.pad(q, padding)
        k = tf.pad(k, padding)
        v = tf.pad(v, padding)

        if isinstance(length, int) and isinstance(block_length, int):
            num_blocks = length // block_length
        else:
            num_blocks = tf.div(length, block_length)

        # Compute attention for the first query block.
        first_q = tf.slice(q, [0, 0, 0, 0], [-1, -1, block_length, -1])
        first_k = tf.slice(k, [0, 0, 0, 0], [-1, -1, block_length, -1])
        first_v = tf.slice(v, [0, 0, 0, 0], [-1, -1, block_length, -1])

        first_output = dot_product_attention(
            first_q,
            first_k,
            first_v,
            attention_bias_lower_triangle(block_length),
            dropout_rate=dropout_rate,
            make_image_summary=make_image_summary,
            name="first_block",
        )

        # Compute attention for all subsequent query blocks.
        q = tf.reshape(q, [batch, heads, num_blocks, block_length, depth_k])
        k = tf.reshape(k, [batch, heads, num_blocks, block_length, depth_k])
        v = tf.reshape(v, [batch, heads, num_blocks, block_length, depth_v])

        local_k = _make_local_block(
            k, depth_k, batch, heads, num_blocks, block_length
        )
        local_v = _make_local_block(
            v, depth_v, batch, heads, num_blocks, block_length
        )
        tail_q = tf.slice(q, [0, 0, 1, 0, 0], [-1, -1, -1, -1, -1])
        tail_q = tf.reshape(
            tail_q, [batch, heads, num_blocks - 1, block_length, depth_k]
        )
        local_length = common_layers.shape_list(local_k)[3]

        # make sure source_pos <= target_pos
        good_part = common_layers.ones_matrix_band_part(
            block_length,
            local_length,
            -1,
            block_length,
            out_shape=[1, 1, 1, block_length, local_length],
        )
        bias = (1.0 - good_part) * -1e9
        # TODO(noam): figure out how to show a summary for the remaining blocks.
        # The naive way currently causes errors due to empty tensors.
        # output: [batch, heads, num_blocks-1, block_length, depth_v]
        tail_output = dot_product_attention(
            tail_q,
            local_k,
            local_v,
            bias,
            dropout_rate=dropout_rate,
            make_image_summary=False,
            name="tail_block",
        )
        tail_output = tf.reshape(
            tail_output,
            [batch, heads, (num_blocks - 1) * block_length, depth_v],
        )
        output = tf.concat([first_output, tail_output], axis=2)

        # Remove the padding if introduced.
        output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
        output = tf.reshape(output, [batch, heads, original_length, depth_v])
        return output


def _make_local_block(x, depth, batch, heads, num_blocks, block_length):
    """Helper function to create a local version of the keys or values for 1d."""
    prev_block = tf.slice(x, [0, 0, 0, 0, 0], [-1, -1, num_blocks - 1, -1, -1])
    cur_block = tf.slice(x, [0, 0, 1, 0, 0], [-1, -1, -1, -1, -1])
    local_block = tf.concat([prev_block, cur_block], 3)
    return tf.reshape(
        local_block, [batch, heads, num_blocks - 1, block_length * 2, depth]
    )


def local_attention_1d(q, k, v, block_length=128, filter_width=100, name=None):
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
      name: an optional string

    Returns:
      a Tensor of shape [batch, heads, length, depth_v]
    """
    with tf.variable_scope(
        name, default_name="local_self_attention_1d", values=[q, k, v]
    ):
        # Check that q, k, v have the same shape except in their depth dimension.
        q.get_shape()[:-1].assert_is_compatible_with(k.get_shape()[:-1])
        q.get_shape()[:-1].assert_is_compatible_with(v.get_shape()[:-1])

        batch_size, num_heads, original_length, _ = common_layers.shape_list(q)

        # Pad query, key, value to ensure multiple of corresponding lengths.
        def pad_to_multiple(x, pad_length):
            x_length = common_layers.shape_list(x)[2]
            return tf.pad(
                x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]]
            )

        def pad_l_and_r(x, pad_length):
            return tf.pad(x, [[0, 0], [0, 0], [pad_length, pad_length], [0, 0]])

        # Set up query blocks.
        # [batch, heads, blocks_q, block_length, depth_k]
        q = pad_to_multiple(q, block_length)
        q = reshape_by_blocks(q, common_layers.shape_list(q), block_length)
        total_query_blocks = common_layers.shape_list(q)[2]

        # Set up key and value blocks.
        # [batch, heads, blocks_k, block_length, depth_k]
        blocks_per_filter_width = filter_width // block_length
        remaining_items = filter_width % block_length
        k = pad_to_multiple(k, block_length)
        v = pad_to_multiple(v, block_length)
        k = pad_l_and_r(k, filter_width + block_length - remaining_items)
        v = pad_l_and_r(v, filter_width + block_length - remaining_items)
        k = reshape_by_blocks(k, common_layers.shape_list(k), block_length)
        v = reshape_by_blocks(v, common_layers.shape_list(v), block_length)

        total_kv_blocks = common_layers.shape_list(k)[2]

        slices = []
        # prepare the left-most and right-most partial blocks if needed
        if remaining_items:
            first_partial_block_k = tf.slice(
                k,
                [0, 0, 0, block_length - remaining_items, 0],
                [-1, -1, total_query_blocks, -1, -1],
            )
            first_partial_block_v = tf.slice(
                v,
                [0, 0, 0, block_length - remaining_items, 0],
                [-1, -1, total_query_blocks, -1, -1],
            )
            last_partial_block_k = tf.slice(
                k,
                [0, 0, total_kv_blocks - total_query_blocks, 0, 0],
                [-1, -1, -1, remaining_items, -1],
            )
            last_partial_block_v = tf.slice(
                v,
                [0, 0, total_kv_blocks - total_query_blocks, 0, 0],
                [-1, -1, -1, remaining_items, -1],
            )
            slices.append((first_partial_block_k, first_partial_block_v))
            slices.append((last_partial_block_k, last_partial_block_v))

        # Prepare the rest of the blocks
        first_block_index = 1 if remaining_items else 0
        attention_blocks = 2 * blocks_per_filter_width + 1
        for i in range(first_block_index, attention_blocks + first_block_index):
            block_k = tf.slice(
                k, [0, 0, i, 0, 0], [-1, -1, total_query_blocks, -1, -1]
            )
            block_v = tf.slice(
                v, [0, 0, i, 0, 0], [-1, -1, total_query_blocks, -1, -1]
            )
            slices.append((block_k, block_v))
        # [batch, heads, blocks_q, block_length + 2 * filter_width, depth_k]
        k = tf.concat([s[0] for s in slices], axis=3)
        v = tf.concat([s[1] for s in slices], axis=3)

        attention_bias = tf.expand_dims(embedding_to_padding(k) * -1e9, axis=-2)
        depth_v = common_layers.shape_list(v)[-1]

        output = dot_product_attention(
            q,
            k,
            v,
            attention_bias,
            dropout_rate=0.0,
            name="local_1d",
            make_image_summary=False,
        )
        output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])

        # Remove the padding if introduced.
        output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
        output.set_shape(
            [
                None if isinstance(dim, tf.Tensor) else dim
                for dim in (batch_size, num_heads, original_length, depth_v)
            ]
        )
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
    x = tf.reshape(
        x,
        [
            x_shape[0],
            x_shape[1],
            x_shape[2] // memory_block_size,
            memory_block_size,
            x_shape[3],
        ],
    )
    return x


def local_attention_2d(
    q, k, v, query_shape=(8, 16), memory_flange=(8, 16), name=None
):
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
      name: an optional string

    Returns:
      a Tensor of shape [batch, heads, h, w, depth_v]
    """
    with tf.variable_scope(
        name, default_name="local_self_attention_2d", values=[q, k, v]
    ):
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
            name="local_2d",
            make_image_summary=False,
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
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    if len(old_shape) == 4:
        height_padding = -common_layers.shape_list(x)[1] % block_shape[0]
        width_padding = -common_layers.shape_list(x)[2] % block_shape[1]
        paddings = [[0, 0], [0, height_padding], [0, width_padding], [0, 0]]
    elif len(old_shape) == 5:
        height_padding = -common_layers.shape_list(x)[2] % block_shape[0]
        width_padding = -common_layers.shape_list(x)[3] % block_shape[1]
        paddings = [
            [0, 0],
            [0, 0],
            [0, height_padding],
            [0, width_padding],
            [0, 0],
        ]

    padded_x = tf.pad(x, paddings)
    padded_shape = padded_x.get_shape().as_list()
    padded_shape = padded_shape[:-1] + [last]
    padded_x.set_shape(padded_shape)
    return padded_x


def reshape_range(tensor, i, j, shape):
    """Reshapes a tensor between dimensions i and j."""
    t_shape = common_layers.shape_list(tensor)
    target_shape = t_shape[:i] + shape + t_shape[j:]
    return tf.reshape(tensor, target_shape)


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
    q, k, v, query_shape=(8, 16), memory_flange=(8, 16), name=None
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
      name: an optional string

    Returns:
      a Tensor of shape [batch, heads, h, w, depth_v]
    """
    with tf.variable_scope(
        name, default_name="local_masked_self_attention_2d", values=[q, k, v]
    ):
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
            name="masked_local_2d",
            make_image_summary=False,
        )
        # Put representations back into original shapes.
        padded_q_shape = common_layers.shape_list(q)
        output = scatter_blocks_2d(output, q_indices, padded_q_shape)

        # Remove the padding if introduced.
        output = tf.slice(
            output, [0, 0, 0, 0, 0], [-1, -1, v_shape[2], v_shape[3], -1]
        )
        return output
