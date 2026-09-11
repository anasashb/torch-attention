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

import torch
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
    def forward(ctx, query, key, value):
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
    def backward(ctx, output_gradient):
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


def causal_linear(Q, K, V):
    Q = Q.permute(0, 2, 1, 3).contiguous()
    K = K.permute(0, 2, 1, 3).contiguous()
    V = V.permute(0, 2, 1, 3).contiguous()
    V_new = causal_dot_product(Q, K, V)
    return V_new.permute(0, 2, 1, 3).contiguous()


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

    def __init__(self, feature_map=None, eps=1e-6):
        super().__init__()
        self.feature_map = (
            feature_map if feature_map is not None else _elu_feature_map
        )
        self.eps = eps

    def _make_sizes_compatible(self, Q, K):
        """Either slice or pad K in case that the sizes do not match between Q
        and K."""
        N, L, H, E = Q.shape
        _, S, _, _ = K.shape
        if L == S:
            return Q, K

        if L < S:
            return Q, K[:, :L, :, :]

        if L > S:
            return Q, torch.cat([K, K.new_zeros(N, L - S, H, E)], dim=1)

    def forward(
        self, queries, keys, values, attn_mask, query_lengths, key_lengths
    ):
        # Apply the feature map to the queries and keys
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # Apply the key padding mask and make sure the attn_mask is a
        # lower triangular causal mask
        if not attn_mask.lower_triangular:
            raise RuntimeError(
                "CausalLinearAttention only supports full "
                "lower triangular masks"
            )
        K = K * key_lengths.float_matrix[:, :, None, None]

        # Ensure that Q and K have compatible sizes for the following
        # computations, namely L == S
        Q, K = self._make_sizes_compatible(Q, K)

        # TODO: Shall we divide the Q and K with a relatively large number to
        #       avoid numerical instabilities in computing the denominator?
        #       We used to divide each with the max norm of all q and k but
        #       that seems relatively costly for a simple normalization.

        # Compute the normalizers
        Z = 1 / (torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + self.eps)

        # Compute the unnormalized result
        V = causal_linear(Q, K, values)

        return V * Z[:, :, :, None]
