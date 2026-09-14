# Adapted from fast-transformers:
# https://github.com/idiap/fast-transformers/blob/2ad36b97e64cb93862937bd21fcc9568d989561f/fast_transformers/attention/causal_linear_attention.py
# https://github.com/idiap/fast-transformers/blob/2ad36b97e64cb93862937bd21fcc9568d989561f/fast_transformers/causal_product/__init__.py
#
# Licensed under the MIT License.
# This file has been modified for ADLERS.
# See LICENSES/fast-transformers-MIT.txt and NOTICE.
#
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#
# Paper:
# Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
# https://proceedings.mlr.press/v119/katharopoulos20a.html
#
# Equation references below refer to this paper.
#

"""Implement causally masked linear attention."""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from adlers.shared._attention_base import AttentionBase

from .causal_product_cpu import causal_dot_backward as causal_dot_backward_cpu
from .causal_product_cpu import causal_dot_product as causal_dot_product_cpu
from .linear_attention import _elu_feature_map

try:
    from .causal_product_cuda import (
        causal_dot_backward as causal_dot_backward_cuda,
    )
    from .causal_product_cuda import (
        causal_dot_product as causal_dot_product_cuda,
    )
except ImportError:
    causal_dot_product_cuda = causal_dot_backward_cuda = None


class CausalDotProduct(torch.autograd.Function):
    """
    Computes the unnormalized weighted sum of values for causal Linear Attention.

    For each position, the compiled CPU or CUDA function applies the mapped
    query to the prefix key-value sum from Equation 10. Its forward and
    backward computations follow Algorithm 1 of *Transformers are RNNs*.
    """

    dot = {"cpu": causal_dot_product_cpu, "cuda": causal_dot_product_cuda}
    dot_backward = {
        "cpu": causal_dot_backward_cpu,
        "cuda": causal_dot_backward_cuda,
    }

    @staticmethod
    def forward(
        ctx: Any,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        # Save the inputs for the gradient computation
        ctx.save_for_backward(query, key, value)

        # Create the output tensor
        device = query.device
        batch_size, num_heads, num_queries, _ = query.shape
        value_head_dim = value.shape[-1]
        attn_output = torch.zeros(
            (batch_size, num_heads, num_queries, value_head_dim),
            device=device,
        )

        # Actually perform the dot product
        CausalDotProduct.dot[device.type](
            query.data,
            key.data,
            value.data,
            attn_output,
        )

        return attn_output

    @staticmethod
    def backward(
        ctx: Any,
        output_gradient: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Extract the saved tensors
        query, key, value = ctx.saved_tensors

        # Allocate memory for the gradients
        query_gradient = torch.zeros_like(query)
        key_gradient = torch.zeros_like(key)
        value_gradient = torch.zeros_like(value)

        # Actually compute the gradients
        CausalDotProduct.dot_backward[query.device.type](
            query.data,
            key.data,
            value.data,
            output_gradient,
            query_gradient,
            key_gradient,
            value_gradient,
        )

        return query_gradient, key_gradient, value_gradient


# Alias the autograd functions to python style snake case naming
causal_dot_product = CausalDotProduct.apply


def causal_linear(
    mapped_query: Tensor,
    mapped_key: Tensor,
    value: Tensor,
) -> Tensor:
    """
    Computes unnormalized weighted value sums for causal Linear Attention.

    Args:
        mapped_query (Tensor): Feature-mapped query tensor.
        mapped_key (Tensor): Feature-mapped key tensor.
        value (Tensor): Value tensor.

    Returns:
        Tensor: The unnormalized weighted value sums.
    """
    return causal_dot_product(
        mapped_query.contiguous(),
        mapped_key.contiguous(),
        value.contiguous(),
    )


class CausalLinearAttention(Module):
    """
    Implements causal Linear Attention from *Transformers are RNNs*.

    Cumulative key and key-value sums restrict each query to its current
    position and preceding positions, following Equations 9-12.

    Args:
        feature_map (Callable[[Tensor], Tensor] | None): Function applied to
            query and key tensors. When None, defaults to ELU(x) + 1 from
            Equation 7.
        eps (float): Small value added to the normalization denominator for
            numerical stability.

    Attributes:
        feature_map (Callable[[Tensor], Tensor]): Configured feature-map
            function.
        eps (float): Value added to the normalization denominator.
    """

    def __init__(
        self,
        feature_map: Callable[[Tensor], Tensor] | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.feature_map = (
            feature_map if feature_map is not None else _elu_feature_map
        )
        self.eps = eps

    def _make_sizes_compatible(
        self,
        mapped_query: Tensor,
        mapped_key: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Either slice or pad K in case that the sizes do not match between Q
        and K."""
        batch_size, num_heads, num_queries, head_dim = mapped_query.shape
        _, _, num_keys, _ = mapped_key.shape
        if num_queries == num_keys:
            return mapped_query, mapped_key

        if num_queries < num_keys:
            return mapped_query, mapped_key[:, :, :num_queries, :]

        return mapped_query, torch.cat(
            [
                mapped_key,
                mapped_key.new_zeros(
                    batch_size,
                    num_heads,
                    num_queries - num_keys,
                    head_dim,
                ),
            ],
            dim=2,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        AttentionBase._validate_qkv_rank(
            query=query,
            key=key,
            value=value,
        )
        AttentionBase._validate_qkv_batch_sizes(
            query=query,
            key=key,
            value=value,
        )
        AttentionBase._validate_qkv_head_counts(
            query=query,
            key=key,
            value=value,
        )
        AttentionBase._validate_qk_head_dimensions(
            query=query,
            key=key,
        )
        AttentionBase._validate_kv_sequence_lengths(
            key=key,
            value=value,
        )
        num_queries = query.shape[-2]
        num_keys = key.shape[-2]
        num_values = value.shape[-2]
        if not (num_queries == num_keys == num_values):
            raise ValueError(
                "Query, key, and value sequence lengths must match for causal "
                "linear attention; "
                f"got query length {num_queries}, key length {num_keys}, and "
                f"value length {num_values}. Use the same sequence length for "
                "all three tensors."
            )

        # Apply the feature map to the queries and keys (Equation 7)
        mapped_query = self.feature_map(query)
        mapped_key = self.feature_map(key)

        if attn_mask is not None:
            AttentionBase._validate_attn_mask_dtype(attn_mask=attn_mask)
            expected_mask_shape = (
                query.shape[0],
                1,
                1,
                key.shape[-2],
            )
            if attn_mask.shape != expected_mask_shape:
                raise ValueError(
                    "Linear attention only supports key-padding masks shaped "
                    "[batch_size, 1, 1, num_keys]; "
                    f"got shape {tuple(attn_mask.shape)}."
                )

            key_padding_mask = attn_mask.squeeze(dim=-2).unsqueeze(dim=-1)
            mapped_key = mapped_key.masked_fill(key_padding_mask, 0)

        # Ensure that Q and K have compatible sizes for the following
        # computations, namely L == S
        mapped_query, mapped_key = self._make_sizes_compatible(
            mapped_query,
            mapped_key,
        )

        # Invert the denominator from Equation 12 for the final multiplication
        normalization_factor = 1 / (
            torch.einsum(
                "bhld,bhld->bhl",
                mapped_query,
                mapped_key.cumsum(dim=-2),
            )
            + self.eps
        )

        # Compute the numerator from Equation 12
        unnormalized_attn_output = causal_linear(
            mapped_query,
            mapped_key,
            value,
        )

        return unnormalized_attn_output * normalization_factor[:, :, :, None]
