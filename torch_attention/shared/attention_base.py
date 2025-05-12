from typing import Optional, Tuple

from torch import Tensor, nn


class AttentionBase(nn.Module):
    """
    Base class for all attention modules in this package.

    Args:
        use_mask (bool): Whether forward() should expect and (even if not
            provided) apply an attention mask.
        dropout_rate (float): Dropout rate.
        output_attention (bool): Whether forward() should return attention
            weights.
        strict_mode (bool): Whether to explicitly validate tensor shapes
            at each forward call.
        scale_factor (Optional[float]): Custom attention scaling factor.

    Child classes must implement forward().
    """

    def __init__(
        self,
        use_mask: bool = False,
        dropout_rate: float = 0.0,
        output_attention: bool = False,
        strict_mode: bool = True,
        scale_factor: Optional[float] = None,
    ):
        super().__init__()
        self.use_mask = use_mask
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.output_attention = output_attention
        self.strict_mode = strict_mode
        self.scale_factor = scale_factor

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass of the attention mechanism.

        Args:
            query (Tensor): Query tensor of shape [batch_size, num_heads,
                num_queries, head_dim].
            key (Tensor): Key tensor of shape [batch_size, num_heads,
                num_keys, head_dim].
            value (Tensor): Value tensor of shape [batch_size, num_heads,
                num_values, head_dim].
            mask (Tensor): Attention mask tensor of either
                a 3D shape of [batch_size, num_queries, num_keys] or
                a 4D shape of [batch_size, num_heads, num_queries, num_keys].

        Returns:
            y (Tensor): Attention output tensor of shape [batch_size, num_heads,
                num_queries, head_dim].
            attn_weights (Optional[Tensor]): Attention weights tensor of shape
                [batch_size, num_heads, num_queries, num_keys].
        """
        raise NotImplementedError(
            "Forward must be implemented in a child class"
        )

    def _validate_shapes(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor],
    ) -> None:
        """
        Validates shapes of the Query, Key, Value and optional attention mask
        tensors. This will only be called if strict_mode = True at init.

        Args:
            query (Tensor): Query tensor of shape [batch_size, num_heads,
                num_queries, head_dim].
            key (Tensor): Key tensor of shape [batch_size, num_heads,
                num_keys, head_dim].
            value (Tensor): Value tensor of shape [batch_size, num_heads,
                num_values, head_dim].
            mask (Tensor): Attention mask tensor of either
                a 3D shape of [batch_size, num_queries, num_keys] or
                a 4D shape of [batch_size, num_heads, num_queries, num_keys].
        Returns:
            None.
        """
        # Short-hand notations shapes
        bq, hq, lq, dh = query.shape
        bk, hk, lk, dh2 = key.shape
        bv, hv, _, dh3 = value.shape

        if not (bq == bk == bv):
            raise ValueError(
                "Batch size mismatch between Queries, Keys, Values tensors."
            )
        if not (hq == hk == hv):
            raise ValueError(
                "Attention heads count mismatch between Queries, Keys, Values "
                "tensors."
            )
        if not (dh == dh2 == dh3):
            raise ValueError(
                "Attention heads dimension mismatch between Queries, Keys, "
                "Values, tensors."
            )

        if mask is not None and mask.shape not in [
            (bq, lq, lk),
            (bq, hq, lq, lk),
        ]:
            raise ValueError(
                f"Invalid mask shape {mask.shape}, expected (batch_size, "
                "num_queries, num_keys) or (batch_size, num_heads, "
                "num_queries, num_keys)."
            )
