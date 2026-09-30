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


class CrossCovarianceAttention(nn.Module):
    """Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """

    def __init__(
        self,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        value = value.transpose(-2, -1)

        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        scores = (query @ key.transpose(-2, -1)) * self.temperature
        attn_weights = scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        return (attn_weights @ value).transpose(dim0=-2, dim1=-1)
