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
    """Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(
        self,
        num_heads: int | None = None,
        is_causal: bool = False,
        dropout_rate: float = 0.0,
        strict_mode: bool = True,
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
        self.temperature: nn.Parameter | nn.UninitializedParameter = (
            nn.UninitializedParameter()
            if num_heads is None
            else nn.Parameter(torch.ones(num_heads, 1, 1))
        )

    def initialize_parameters(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
    ) -> None:
        """Initializes per-head temperatures from the input head count."""
        if isinstance(self.temperature, nn.UninitializedParameter):
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
        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        value = value.transpose(-2, -1)

        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        scores = (query @ key.transpose(-2, -1)) * self.temperature
        attn_weights = scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        return (attn_weights @ value).transpose(dim0=-2, dim1=-1)
