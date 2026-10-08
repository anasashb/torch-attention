# Adapted from XCiT:
# https://github.com/facebookresearch/xcit/blob/82f5291f412604970c39a912586e008ec009cdca/xcit.py
#
# Licensed under the Apache License, Version 2.0.
# This file has been modified for ADLERS.
# See LICENSES/Apache-2.0.txt and NOTICE.
#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# Paper:
# XCiT: Cross-Covariance Image Transformers
# https://arxiv.org/abs/2106.09681v2
#
# Equation and algorithm references below refer to this paper.
"""Cross-covariance attention from the implementation of Cross-Covariance Image Transformer (XCiT).

Based on the timm and DeiT code bases:
https://github.com/rwightman/pytorch-image-models/tree/master/timm
https://github.com/facebookresearch/deit/
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.lazy import LazyModuleMixin

from adlers.shared._attention_base import AttentionBase


class CrossCovarianceAttention(LazyModuleMixin, AttentionBase):
    """
    Implements XCiT's Cross-Covariance Attention (XCA) mechanism.

    L2-normalized queries and keys are used to compute attention weights
    between channels, as described in Section 3.2 of the XCiT paper.
    These weights mix value channels within each token.

    Args:
        num_heads (int | None): Number of attention heads. When None,
            inferred on the first forward call and fixed for later calls.
        is_causal (bool): Only False is supported.
        dropout_rate (float): Dropout rate applied to attention weights
            during training.
        strict_mode (bool): Whether input shapes are validated on every call.
        learnable_temperature (bool): Whether per-head temperatures are
            learned. Defaults to True, matching the original implementation.
            When False, uses a fixed temperature of 1.0.

    Attributes:
        num_heads (int | None): Configured or inferred head count.
        temperature (Tensor): Per-head score multipliers of shape
            [num_heads, 1, 1], initialized to 1.0. Stored as a parameter
            when learned, or a buffer otherwise.
        is_causal (bool): Whether causal masking is enabled. Always False.
        strict_mode (bool): Whether shape validation is enabled.
    """

    def __init__(
        self,
        num_heads: int | None = None,
        is_causal: bool = False,
        dropout_rate: float = 0.0,
        strict_mode: bool = True,
        learnable_temperature: bool = True,
    ) -> None:
        if is_causal:
            raise ValueError(
                "Cross-covariance attention does not support causal masking; "
                "got is_causal=True. Set is_causal=False."
            )

        super().__init__(
            is_causal=is_causal,
            dropout_rate=dropout_rate,
            strict_mode=strict_mode,
        )
        self.num_heads = num_heads

        self.temperature: Tensor
        # original XCA behavior
        if learnable_temperature:
            self.temperature = (
                nn.UninitializedParameter()
                if num_heads is None
                else nn.Parameter(torch.ones(num_heads, 1, 1))
            )
        # fixed temperature constant (1) here as an extra (convenience)
        # extension for ADLERS
        else:
            self.register_buffer(
                name="temperature",
                tensor=(
                    nn.UninitializedBuffer()
                    if num_heads is None
                    else torch.ones(num_heads, 1, 1)
                ),
            )

    def initialize_parameters(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
    ) -> None:
        """
        Initializes per-head temperatures from the input head count.

        Called automatically before the first forward call. Uninitialized
        temperatures are set to 1.0; existing temperatures are left unchanged.
        """
        if isinstance(
            self.temperature,
            (nn.UninitializedParameter, nn.UninitializedBuffer),
        ):
            if self.strict_mode:
                self._validate_shapes(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=attn_mask,
                )

            with torch.no_grad():
                self.temperature.materialize(shape=(query.shape[1], 1, 1))
                nn.init.ones_(tensor=self.temperature)

        self.num_heads = self.temperature.shape[0]

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Computes cross-covariance attention.

        Args:
            query (Tensor): Query tensor of shape [batch_size, num_heads,
                num_queries, head_dim].
            key (Tensor): Key tensor of shape [batch_size, num_heads,
                num_keys, head_dim].
            value (Tensor): Value tensor of shape [batch_size, num_heads,
                num_keys, value_head_dim].
            attn_mask (Tensor | None): Must be None. Custom attention masks
                are not supported.

        Returns:
            Tensor: Attention output of shape [batch_size, num_heads,
                num_queries, value_head_dim].

        Raises:
            ValueError: If a custom attention mask is supplied or an input shape
                is invalid.
        """
        if attn_mask is not None:
            raise ValueError(
                "Cross-covariance attention does not support custom attention masks; "
                f"got shape {tuple(attn_mask.shape)}. Pass attn_mask=None."
            )

        return super().forward(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
        )

    def _validate_shapes(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None,
    ) -> None:
        """Validates tensor shapes for cross-covariance attention."""
        super()._validate_shapes(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
        )

        input_num_heads = query.shape[1]
        if self.num_heads is not None and input_num_heads != self.num_heads:
            raise ValueError(
                "Cross-covariance attention was configured with "
                f"num_heads={self.num_heads}; got query head count {input_num_heads}. "
                "Set num_heads to match the input tensors."
            )

        num_queries = query.shape[-2]
        num_keys = key.shape[-2]
        # Lq, Lk need to match becuase cross-covariance matmuls sum over
        # token positions
        if num_queries != num_keys:
            raise ValueError(
                "Cross-covariance attention requires matching query and key "
                f"sequence lengths; got query length {num_queries} and key "
                f"length {num_keys}. Use the same sequence length for both tensors."
            )

        key_head_dim = key.shape[-1]
        value_head_dim = value.shape[-1]
        # Dhk, Dhv need to match because attn_weights @ value sums over
        # channels
        if key_head_dim != value_head_dim:
            raise ValueError(
                "Cross-covariance attention requires matching key and value head "
                f"dimensions; got key head dimension {key_head_dim} and value head "
                f"dimension {value_head_dim}. "
                "Use the same head dimension for both tensors."
            )

    def _attend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None,
    ) -> Tensor:
        """Computes cross-covariance attention for the supplied tensors."""
        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        value = value.transpose(-2, -1)

        # L2-normalize each query and key channel across tokens (Section 3.2).
        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        # Compute per-head channel scores, shaped [B, H, D, D] (Section 3.2).
        # NOTE: The XCiT authors' original implementation uses in code (see
        # below) uses query @ key.T, whereas Algorithm 1 in the appendix of the
        # XCiT paper uses key @ query.T. These are not generally equivalent
        # after softmax for the same inputs, but here I preserve the original
        # code as was.
        scores = (query @ key.transpose(-2, -1)) * self.temperature
        attn_weights = scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        return (attn_weights @ value).transpose(dim0=-2, dim1=-1)
