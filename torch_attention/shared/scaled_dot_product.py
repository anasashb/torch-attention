from math import sqrt
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from torch_attention.shared._attention_base import AttentionBase


class ScaledDotProductAttention(AttentionBase):
    # TODO docstring
    def _attend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale_factor: float,
        mask: Optional[Tensor],
    ):

        # Short-hand notation for shapes
        Bq, Hq, Lq, _ = query.shape
        _, _, Lk, _ = key.shape

        # Get raw scores
        scores = torch.einsum("bhle,bhse->bhls", query, key)

        # Apply mask if needed
        if self.use_mask:
            if mask is None:
                mask = torch.triu(
                    torch.ones(Lq, Lk, dtype=torch.bool, device=query.device),
                    diagonal=1,
                )
                mask = mask.unsqueeze(0).unsqueeze(0).expand(Bq, Hq, Lq, Lk)

            scores.masked_fill_(mask, float("-inf"))

        # Get attention scores
        attn_weights = self.dropout(
            torch.softmax(scale_factor * scores, dim=-1)
        )

        # TODO if dropout handling somewhere here

        # Get attention outputs
        attn_outputs = torch.einsum("bhls,bhsd->bhld", attn_weights, value)

        return attn_outputs, attn_weights
