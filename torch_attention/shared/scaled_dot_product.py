from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_attention.shared._attention_base import AttentionBase


class ScaledDotProductAttention(AttentionBase):
    """
    Conventional scaled dot-product attention. Inherits .forward() and instance
    arguments from AttentionBase, implements custom ._attend() method called
    in .forward().

    Args:
        use_mask (bool): Whether forward() should expect and (even if not
            provided) apply an attention mask.
        dropout_rate (float): Dropout rate.
        output_attention_scores (bool): Whether forward() should return
            attention weights.
        strict_mode (bool): Whether to explicitly validate tensor shapes
            at each forward call.
        scale_factor (Optional[float]): Custom attention scaling factor.
    """

    def _attend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale_factor: float,
        mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Attention computations of the conventional scaled dot-product. This
        method will be called by the .forward() method inherited from the
        AttentionBase class.

        Args:
            query (Tensor): Query tensor of shape [batch_size, num_heads,
                num_queries, head_dim].
            key (Tensor): Key tensor of shape [batch_size, num_heads,
                num_keys, head_dim].
            value (Tensor): Value tensor of shape [batch_size, num_heads,
                num_values, head_dim].
            scale_factor (float): Scale factor to multiply raw scores by.
            mask (Tensor): Attention mask tensor of shape [batch_size,
                num_heads, num_queries, num_keys].

        Returns:
            attn_output (Tensor): Attention output tensor of shape [batch_size,
                num_heads, num_queries, head_dim].
            attn_weights (Tensor): Attention weights tensor of shape
                [batch_size, num_heads, num_queries, num_keys].
        """

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

        # Get attention outputs
        attn_outputs = torch.einsum("bhls,bhsd->bhld", attn_weights, value)

        return attn_outputs, attn_weights
