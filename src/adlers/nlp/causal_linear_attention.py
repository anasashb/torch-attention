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

"""Implement causally masked linear attention."""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

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
    """Compute the weighted sum of values but attending only to previous
    values."""

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
    mapped_query = mapped_query.permute(0, 2, 1, 3).contiguous()
    mapped_key = mapped_key.permute(0, 2, 1, 3).contiguous()
    value = value.permute(0, 2, 1, 3).contiguous()
    unnormalized_attn_output = causal_dot_product(
        mapped_query,
        mapped_key,
        value,
    )
    return unnormalized_attn_output.permute(0, 2, 1, 3).contiguous()


class CausalLinearAttention(Module):
    """Implement causally masked attention using dot product of feature maps in
    O(N D^2) complexity.

    See fast_transformers.attention.linear_attention.LinearAttention for the
    general concept of replacing the softmax with feature maps. In addition to
    that, we also make use of the fact that causal masking is a triangular mask
    which allows us to apply the masking and still compute the attention in O(N
    D^2) complexity.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
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
        batch_size, num_queries, num_heads, head_dim = mapped_query.shape
        _, num_keys, _, _ = mapped_key.shape
        if num_queries == num_keys:
            return mapped_query, mapped_key

        if num_queries < num_keys:
            return mapped_query, mapped_key[:, :num_queries, :, :]

        return mapped_query, torch.cat(
            [
                mapped_key,
                mapped_key.new_zeros(
                    batch_size,
                    num_queries - num_keys,
                    num_heads,
                    head_dim,
                ),
            ],
            dim=1,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Any,
        query_lengths: Any,
        key_lengths: Any,
    ) -> Tensor:
        # Apply the feature map to the queries and keys
        mapped_query = self.feature_map(query)
        mapped_key = self.feature_map(key)

        # Apply the key padding mask and make sure the attn_mask is a
        # lower triangular causal mask
        if not attn_mask.lower_triangular:
            raise RuntimeError(
                "CausalLinearAttention only supports full "
                "lower triangular masks"
            )
        mapped_key = mapped_key * key_lengths.float_matrix[:, :, None, None]

        # Ensure that Q and K have compatible sizes for the following
        # computations, namely L == S
        mapped_query, mapped_key = self._make_sizes_compatible(
            mapped_query,
            mapped_key,
        )

        # TODO: Shall we divide the Q and K with a relatively large number to
        #       avoid numerical instabilities in computing the denominator?
        #       We used to divide each with the max norm of all q and k but
        #       that seems relatively costly for a simple normalization.

        # Compute the normalizers
        normalization_factor = 1 / (
            torch.einsum(
                "nlhi,nlhi->nlh",
                mapped_query,
                mapped_key.cumsum(1),
            )
            + self.eps
        )

        # Compute the unnormalized result
        unnormalized_attn_output = causal_linear(
            mapped_query,
            mapped_key,
            value,
        )

        return unnormalized_attn_output * normalization_factor[:, :, :, None]
