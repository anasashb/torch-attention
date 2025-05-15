from abc import ABC, abstractmethod
from typing import Optional, Tuple

from torch import Tensor, nn


class AttentionBase(nn.Module, ABC):
    """
    Base class for all attention modules in this package.

    Args:
        use_mask (bool): Whether forward() should expect and (even if not
            provided) apply an attention mask.
        dropout_rate (float): Dropout rate.
        output_attention_scores (bool): Whether forward() should return
            attention weights.
        strict_mode (bool): Whether to explicitly validate tensor shapes
            at each forward call.
        scale_factor (Optional[float]): Custom attention scaling factor.

    Child classes must implement _attend().
    """

    def __init__(
        self,
        use_mask: bool = False,
        dropout_rate: float = 0.0,
        output_attention_scores: bool = False,
        strict_mode: bool = True,
        scale_factor: Optional[float] = None,
    ):
        super().__init__()
        self.use_mask = use_mask
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.output_attention_scores = output_attention_scores
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
        Forward method to pass down to child classes. Includes shared logic
        such as mask shape normalization (adjustment), Q, K, V, mask shape
        validation. Attention computation is then delegated to the abstract
        _attend method.

        Args:
            query (Tensor): Query tensor of shape [batch_size, num_heads,
                num_queries, head_dim].
            key (Tensor): Key tensor of shape [batch_size, num_heads,
                num_keys, head_dim].
            value (Tensor): Value tensor of shape [batch_size, num_heads,
                num_values, head_dim].
            mask (Tensor): Attention mask tensor of either:
                a 2D shape of [num_queries, num_keys],
                a 3D shape of [batch_size, num_queries, num_keys], or
                a 4D shape of [batch_size, num_heads, num_queries, num_keys].

        Returns:
            attn_output (Tensor): Attention output tensor of shape [batch_size,
                num_heads, num_queries, head_dim].
            attn_weights (Optional[Tensor]): Attention weights tensor of shape
                [batch_size, num_heads, num_queries, num_keys].
        """
        # Adjust mask shape if mask should be used and is provided
        if self.use_mask and mask is not None:
            mask = self._normalize_mask(
                mask=mask, batch_size=query.shape[0], num_heads=query.shape[1]
            )
        # Validate input shapes if using strict mode
        if self.strict_mode:
            self._validate_shapes(query=query, key=key, value=value, mask=mask)

        # Core computations
        attn_output, attn_weights = self._attend(
            query=query, key=key, value=value, mask=mask
        )

        return (
            (attn_output, attn_weights)
            if self.output_attention_scores
            else (attn_output, None)
        )

    @abstractmethod
    def _attend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Core attention method that will be overriden in subclasses.

        Returns:
            attn_output (Tensor): Attention output tensor of shape [batch_size,
                num_heads, num_queries, head_dim].
            attn_weights (Tensor): Attention weights tensor of shape
                [batch_size, num_heads, num_queries, num_keys].
        """
        raise NotImplementedError("Subclasses must implement _attend()")

    # NOTE can be turned into @staticmethod too
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
            mask (Tensor): Attention mask tensor of either:
                a 2D shape of [num_queries, num_keys],
                a 3D shape of [batch_size, num_queries, num_keys], or
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
            (lq, lk),
            (bq, lq, lk),
            (bq, hq, lq, lk),
        ]:
            raise ValueError(
                f"Invalid mask shape {mask.shape}, expected (num_queries, "
                "num_keys), (batch_size, num_queries, num_keys), or "
                "(batch_size, num_heads, num_queries, num_keys)."
            )

    # NOTE can be turned into @staticmethod too
    def _normalize_mask(
        self,
        mask: Tensor,
        batch_size: int,
        num_heads: int,
    ) -> Tensor:
        """
        Adjusts mask shape from either 2D to 4D or 3D to 4D. Expands to
        num_heads dimension on a 4D mask if needed.

        If given mask dimension == 2, assumes it is [num_queries, num_keys].
        First it unsqueezes to [1, 1, num_queries, num_keys] and then expands
        to [batch_size, num_heads, num_queries, num_keys] to match the expected
        shape of attention scores.

        If given mask dimension == 3, assumes it is [batch_size, num_queries,
        num_keys]. First it unsqueezes to [batch_size, 1, num_queries,
        num_keys] and then expands to [batch_size, num_heads, num_queries,
        num_keys] to match the expected shape of attention scores.

        If given mask dimension == 4, and mask's second dimension == 1 but
        num_heads != 1, assumes mask is provided as [batch_size, 1, num_queries,
        num_keys] and expands it to [batch_size, num_heads, num_queries,
        num_keys].

        Args:
            mask (Tensor): Attention mask tensor of either:
                a 2D shape of [num_queries, num_keys],
                a 3D shape of [batch_size, num_queries, num_keys], or
                a 4D shape of [batch_size, num_heads, num_queries, num_keys].
            batch_size (int): Batch size to expand the 2D masks to.
            num_heads (int): Number of attention heads to expand the 2D / 3D /
                4D masks to.

        Returns:
            mask (Tensor): Attention mask of shape [batch_size, num_heads,
                num_queries, num_keys].

        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.expand(batch_size, num_heads, -1, -1)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
            mask = mask.expand(-1, num_heads, -1, -1)
        elif mask.dim() == 4:
            if mask.shape[1] == 1 and num_heads != 1:
                mask = mask.expand(-1, num_heads, -1, -1)
            else:
                pass
        else:
            raise ValueError(f"Mask must be 2D, 3D or 4D; got {mask.dim()}D")

        return mask
