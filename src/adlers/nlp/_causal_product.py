# Adapted from fast-transformers:
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

"""Provide the compiled causal product used by Linear Attention."""

from typing import Any

import torch
from torch import Tensor

from adlers.nlp.causal_product_cpu import (
    causal_dot_backward as causal_dot_backward_cpu,
)
from adlers.nlp.causal_product_cpu import (
    causal_dot_product as causal_dot_product_cpu,
)

try:
    from adlers.nlp.causal_product_cuda import (
        causal_dot_backward as causal_dot_backward_cuda,
    )
    from adlers.nlp.causal_product_cuda import (
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
