from abc import ABC, abstractmethod
from math import sqrt

import torch
from torch import Tensor, nn


class AttentionBase(nn.Module, ABC):
    """
    Base class for all attention modules in this package.

    Args:
        is_causal (bool): Whether to prevent queries from attending to future
            key positions.
        dropout_rate (float): Dropout rate.
        output_attention_scores (bool): Whether forward() should return
            attention weights.
        strict_mode (bool): Whether to explicitly validate tensor shapes
            at each forward call.
        custom_scale_factor (Optional[float]): Custom attention scaling factor.

    Child classes must implement _attend().
    """

    def __init__(
        self,
        is_causal: bool = False,
        dropout_rate: float = 0.0,
        output_attention_scores: bool = False,
        strict_mode: bool = True,
        custom_scale_factor: float | None = None,
    ) -> None:
        super().__init__()
        self.is_causal = is_causal
        self.dropout = (
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        self.output_attention_scores = output_attention_scores
        self.strict_mode = strict_mode
        self.custom_scale_factor = custom_scale_factor

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Forward method inherited by all child classes of AttentionBase.
        Includes shared logic such as mask shape normalization (adjustment),
        Q, K, V, mask shape validation. Attention computation is then
        delegated to the abstract ._attend() method.

        Args:
            query (Tensor): Query tensor of shape [batch_size, num_heads,
                num_queries, head_dim].
            key (Tensor): Key tensor of shape [batch_size, num_heads,
                num_keys, head_dim].
            value (Tensor): Value tensor of shape [batch_size, num_heads,
                num_values, head_dim].
            attn_mask (Tensor): Boolean attention mask tensor of either:
                a 2D shape of [num_queries, num_keys],
                a 3D shape of [batch_size, num_queries, num_keys], or
                a 4D shape of [batch_size, num_heads, num_queries, num_keys].
                Only torch.bool masks are supported. True marks positions that
                should be masked out, and False marks positions that can be
                attended to.

        Returns:
            attn_output (Tensor): Attention output tensor of shape [batch_size,
                num_heads, num_queries, head_dim].
            attn_weights (Optional[Tensor]): Attention weights tensor of shape
                [batch_size, num_heads, num_queries, num_keys].
        """
        if attn_mask is not None:
            self._validate_attn_mask_dtype(attn_mask=attn_mask)
            attn_mask = self._normalize_attn_mask(
                attn_mask=attn_mask,
                batch_size=query.shape[0],
                num_heads=query.shape[1],
            )

        # Generate scale factor if not provided
        if self.custom_scale_factor is not None:
            scale_factor = self.custom_scale_factor
        else:
            _, _, _, head_dim = key.shape
            scale_factor = 1.0 / sqrt(head_dim)

        # Validate input shapes if using strict mode
        if self.strict_mode:
            self._validate_shapes(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
            )

        # Core computations
        attn_output, attn_weights = self._attend(
            query=query,
            key=key,
            value=value,
            scale_factor=scale_factor,
            attn_mask=attn_mask,
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
        scale_factor: float,
        attn_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Core attention method that will be overriden in subclasses.

        Args:
            query (Tensor): Query tensor of shape [batch_size, num_heads,
                num_queries, head_dim].
            key (Tensor): Key tensor of shape [batch_size, num_heads,
                num_keys, head_dim].
            value (Tensor): Value tensor of shape [batch_size, num_heads,
                num_values, head_dim].
            scale_factor (float): Scale factor to multiply raw scores by.
            attn_mask (Tensor): Boolean attention mask tensor of shape
                [batch_size, num_heads, num_queries, num_keys]. True marks
                positions that should be masked out, and False marks positions
                that can be attended to.

        Returns:
            attn_output (Tensor): Attention output tensor of shape [batch_size,
                num_heads, num_queries, head_dim].
            attn_weights (Optional[Tensor]): Attention weights tensor of shape
                [batch_size, num_heads, num_queries, num_keys].
        """
        raise NotImplementedError("Subclasses must implement _attend()")

    @staticmethod
    def _validate_shapes(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None,
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
            attn_mask (Tensor): Boolean attention mask tensor of either:
                a 2D shape of [num_queries, num_keys],
                a 3D shape of [batch_size, num_queries, num_keys], or
                a 4D shape of [batch_size, num_heads, num_queries, num_keys].
                Only torch.bool masks are supported. True marks positions that
                should be masked out, and False marks positions that can be
                attended to.
        Returns:
            None.
        """
        # Short-hand notations for shapes
        Bq, Hq, Lq, Dhq = query.shape
        Bk, Hk, Lk, Dhk = key.shape
        Bv, Hv, _, Dhv = value.shape

        if not (Bq == Bk == Bv):
            raise ValueError(
                "Batch size mismatch between Queries, Keys, Values tensors."
            )
        if not (Hq == Hk == Hv):
            raise ValueError(
                "Attention heads count mismatch between Queries, Keys, Values "
                "tensors."
            )
        if not (Dhq == Dhk == Dhv):
            raise ValueError(
                "Attention heads dimension mismatch between Queries, Keys, "
                "Values, tensors."
            )

        if attn_mask is not None and attn_mask.shape not in [
            (Lq, Lk),
            (Bq, Lq, Lk),
            (Bq, Hq, Lq, Lk),
        ]:
            raise ValueError(
                f"Invalid mask shape {attn_mask.shape}, expected "
                "(num_queries, num_keys), (batch_size, num_queries, "
                "num_keys), or (batch_size, num_heads, num_queries, "
                "num_keys)."
            )

    @staticmethod
    def _validate_attn_mask_dtype(attn_mask: Tensor) -> None:
        """Validates that an attention mask follows ADLERS semantics."""
        if attn_mask.dtype != torch.bool:
            raise TypeError(
                "Only boolean attention masks are supported; "
                f"got mask dtype {attn_mask.dtype}. Use a torch.bool mask "
                "with True for positions that should be masked out and "
                "False for positions that can be attended to."
            )

    @staticmethod
    def _normalize_attn_mask(
        attn_mask: Tensor,
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
            attn_mask (Tensor): Boolean attention mask tensor of either:
                a 2D shape of [num_queries, num_keys],
                a 3D shape of [batch_size, num_queries, num_keys], or
                a 4D shape of [batch_size, num_heads, num_queries, num_keys].
                Only torch.bool masks are supported. True marks positions that
                should be masked out, and False marks positions that can be
                attended to.
            batch_size (int): Batch size to expand the 2D masks to.
            num_heads (int): Number of attention heads to expand the 2D / 3D /
                4D masks to.

        Returns:
            attn_mask (Tensor): Attention mask of shape [batch_size,
                num_heads, num_queries, num_keys].

        """
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_mask = attn_mask.expand(batch_size, num_heads, -1, -1)
        elif attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)
            attn_mask = attn_mask.expand(-1, num_heads, -1, -1)
        elif attn_mask.dim() == 4:
            if attn_mask.shape[1] == 1 and num_heads != 1:
                attn_mask = attn_mask.expand(-1, num_heads, -1, -1)
            else:
                pass
        else:
            raise ValueError(
                "Attention mask must be 2D, 3D or 4D; "
                f"got {attn_mask.dim()}D."
            )

        return attn_mask

    def _combine_with_causal_mask(
        self,
        query: Tensor,
        key: Tensor,
        attn_mask: Tensor | None,
    ) -> Tensor | None:
        """Combines an explicit attention mask with causal restrictions."""
        if not self.is_causal:
            return attn_mask

        causal_mask = self._make_causal_mask(query=query, key=key)
        if attn_mask is None:
            return causal_mask

        return causal_mask | attn_mask

    @staticmethod
    def _make_causal_mask(query: Tensor, key: Tensor) -> Tensor:
        """Creates a causal mask for the query and key sequence lengths."""
        return torch.triu(
            torch.ones(
                query.shape[-2],
                key.shape[-2],
                dtype=torch.bool,
                device=query.device,
            ),
            diagonal=1,
        )
