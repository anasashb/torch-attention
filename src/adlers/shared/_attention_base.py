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
            attn_mask (Optional[Tensor]): Boolean attention mask tensor of
                either:
                a 2D shape of [num_queries, num_keys],
                a 3D shape of [batch_size, num_queries, num_keys], or
                a 4D shape of [batch_size, 1 or num_heads, num_queries,
                num_keys]. Only torch.bool masks are supported. True marks
                positions that should be masked out, and False marks positions
                that can be attended to.

        Returns:
            attn_output (Tensor): Attention output tensor of shape [batch_size,
                num_heads, num_queries, head_dim].
            attn_weights (Optional[Tensor]): Attention weights tensor of shape
                [batch_size, num_heads, num_queries, num_keys].

        Raises:
            TypeError: If attn_mask is not a torch.bool tensor.
            ValueError: If the mask rank is unsupported, or if strict mode is
                enabled and an input shape is invalid.
        """
        if attn_mask is not None:
            self._validate_attn_mask_dtype(attn_mask=attn_mask)
            attn_mask = self._normalize_attn_mask(attn_mask=attn_mask)

        # Validate input shapes if using strict mode
        if self.strict_mode:
            self._validate_shapes(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
            )

        # Generate scale factor if not provided
        if self.custom_scale_factor is not None:
            scale_factor = self.custom_scale_factor
        else:
            _, _, _, head_dim = key.shape
            scale_factor = 1.0 / sqrt(head_dim)

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
        Core attention method that will be overridden in subclasses.

        Args:
            query (Tensor): Query tensor of shape [batch_size, num_heads,
                num_queries, head_dim].
            key (Tensor): Key tensor of shape [batch_size, num_heads,
                num_keys, head_dim].
            value (Tensor): Value tensor of shape [batch_size, num_heads,
                num_values, head_dim].
            scale_factor (float): Scale factor to multiply raw scores by.
            attn_mask (Optional[Tensor]): Boolean mask broadcastable to
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
            attn_mask (Optional[Tensor]): Boolean attention mask tensor of
                either:
                a 2D shape of [num_queries, num_keys],
                a 3D shape of [batch_size, num_queries, num_keys], or
                a 4D shape of [batch_size, 1 or num_heads, num_queries,
                num_keys]. Only torch.bool masks are supported. True marks
                positions that should be masked out, and False marks positions
                that can be attended to.
        Returns:
            None.
        """
        AttentionBase._validate_qkv_rank(
            query=query,
            key=key,
            value=value,
        )
        AttentionBase._validate_qkv_batch_sizes(
            query=query,
            key=key,
            value=value,
        )

        # Short-hand notations for shapes
        Bq, Hq, Lq, Dhq = query.shape
        _, Hk, Lk, Dhk = key.shape
        _, Hv, Lv, Dhv = value.shape

        if not (Hq == Hk == Hv):
            raise ValueError(
                "Query, key, and value head counts must match; "
                f"got query head count {Hq}, key head count {Hk}, and "
                f"value head count {Hv}. Use the same number of heads for "
                "all three tensors."
            )
        if not (Dhq == Dhk == Dhv):
            raise ValueError(
                "Query, key, and value head dimensions must match; "
                f"got query head dimension {Dhq}, key head dimension {Dhk}, "
                f"and value head dimension {Dhv}. Use the same head dimension "
                "for all three tensors."
            )
        if Lk != Lv:
            raise ValueError(
                "Key and value sequence lengths must match; "
                f"got key length {Lk} and value length {Lv}. "
                "Provide one value position for each key position."
            )

        if attn_mask is not None and attn_mask.shape not in [
            (Lq, Lk),
            (Bq, 1, Lq, Lk),
            (Bq, Hq, Lq, Lk),
        ]:
            raise ValueError(
                f"Invalid mask shape {attn_mask.shape}, expected "
                "(num_queries, num_keys), (batch_size, 1, num_queries, "
                "num_keys), or (batch_size, num_heads, num_queries, "
                "num_keys)."
            )

    @staticmethod
    def _validate_qkv_rank(
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> None:
        """Validates that query, key, and value tensors are four-dimensional."""
        for tensor_name, tensor in (
            ("Query", query),
            ("Key", key),
            ("Value", value),
        ):
            if tensor.ndim != 4:
                raise ValueError(
                    f"{tensor_name} tensor must be 4D "
                    "[batch_size, num_heads, sequence_length, head_dim]; "
                    f"got shape {tuple(tensor.shape)}."
                )

    @staticmethod
    def _validate_qkv_batch_sizes(
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> None:
        """Validates that query, key, and value batch sizes match."""
        Bq = query.shape[0]
        Bk = key.shape[0]
        Bv = value.shape[0]

        if not (Bq == Bk == Bv):
            raise ValueError(
                "Query, key, and value batch sizes must match; "
                f"got query batch size {Bq}, key batch size {Bk}, and "
                f"value batch size {Bv}. Use the same batch size for all "
                "three tensors."
            )

    @staticmethod
    def _validate_attn_mask_dtype(attn_mask: Tensor) -> None:
        """Validates that an attention mask follows ADLERS semantics."""
        # NOTE: for now only allowing boolean masks, may extend support to
        # float masks once/if repo is (ever) more mature
        if attn_mask.dtype != torch.bool:
            raise TypeError(
                "Only boolean attention masks are supported; "
                f"got mask dtype {attn_mask.dtype}. Use a torch.bool mask "
                "with True for positions that should be masked out and "
                "False for positions that can be attended to."
            )

    @staticmethod
    def _normalize_attn_mask(attn_mask: Tensor) -> Tensor:
        """
        Preserves broadcastable mask shapes for attention operations.

        A 2D mask of shape [num_queries, num_keys] is kept unchanged and
        broadcasts across batches and heads.

        A 3D mask of shape [batch_size, num_queries, num_keys] gains a
        singleton head dimension so it broadcasts across heads.

        A 4D mask is kept unchanged, whether it has a singleton head dimension
        or a separate mask for each head.

        Args:
            attn_mask (Tensor): Boolean attention mask tensor of either:
                a 2D shape of [num_queries, num_keys],
                a 3D shape of [batch_size, num_queries, num_keys], or
                a 4D shape of [batch_size, 1 or num_heads, num_queries,
                num_keys].
                Only torch.bool masks are supported. True marks positions that
                should be masked out, and False marks positions that can be
                attended to.

        Returns:
            attn_mask (Tensor): Attention mask with a shape broadcastable to
                [batch_size, num_heads, num_queries, num_keys].
        """
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)
        elif attn_mask.dim() not in (2, 4):
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
