import torch
from torch import Tensor
from torch.nn import functional as F

from torch_attention._typing import AttentionBackend
from torch_attention.shared._attention_base import AttentionBase


class ScaledDotProductAttention(AttentionBase):
    """
    Scaled dot-product attention with selectable computation backends.

    The "einsum" backend computes attention explicitly with PyTorch tensor
    operations and can return attention weights. The "sdpa" backend delegates
    to torch.nn.functional.scaled_dot_product_attention, which can use
    optimized PyTorch kernels depending on the device, dtype, mask, and runtime
    configuration. Use "sdpa" when attention weights are not needed.

    Attention masks must be torch.bool tensors. True marks positions that
    should be masked out, and False marks positions that can be attended to.

    Args:
        use_mask (bool): Whether forward() should expect and (even if not
            provided) apply an attention mask.
        dropout_rate (float): Dropout rate.
        output_attention_scores (bool): Whether forward() should return
            attention weights.
        strict_mode (bool): Whether to explicitly validate tensor shapes
            at each forward call.
        custom_scale_factor (Optional[float]): Custom attention scaling factor.
        backend (AttentionBackend): Attention implementation to use. The
            "einsum" backend returns attention outputs and optionally attention
            weights. The "sdpa" backend delegates to PyTorch's native
            scaled dot-product attention implementation and cannot return
            attention weights.
    """

    def __init__(
        self,
        use_mask: bool = False,
        dropout_rate: float = 0.0,
        output_attention_scores: bool = False,
        strict_mode: bool = True,
        custom_scale_factor: float | None = None,
        backend: AttentionBackend = "einsum",
    ) -> None:
        if backend not in ("einsum", "sdpa"):
            raise ValueError(
                "Invalid backend. Expected 'einsum' or 'sdpa', "
                f"got {backend!r}."
            )
        if backend == "sdpa" and output_attention_scores:
            raise ValueError(
                "The 'sdpa' backend does not support returning attention "
                "scores. Use backend='einsum' or set "
                "output_attention_scores=False."
            )

        super().__init__(
            use_mask=use_mask,
            dropout_rate=dropout_rate,
            output_attention_scores=output_attention_scores,
            strict_mode=strict_mode,
            custom_scale_factor=custom_scale_factor,
        )
        self.backend = backend

    def _attend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale_factor: float,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Routes scaled dot-product attention to the configured backend.

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
            attn_weights (Optional[Tensor]): Attention weights tensor of shape
                [batch_size, num_heads, num_queries, num_keys].
        """
        if self.backend == "einsum":
            return self._attend_einsum(
                query=query,
                key=key,
                value=value,
                scale_factor=scale_factor,
                mask=mask,
            )

        return self._attend_sdpa(
            query=query,
            key=key,
            value=value,
            scale_factor=scale_factor,
            mask=mask,
        )

    def _attend_einsum(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale_factor: float,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """
        Computes attention with explicit einsum operations.

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

    def _attend_sdpa(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale_factor: float,
        mask: Tensor | None,
    ) -> tuple[Tensor, None]:
        """
        Computes attention with PyTorch's native SDPA implementation.

        Boolean masks are inverted before being passed to PyTorch SDPA because
        here we use True for masked positions, while PyTorch SDPA uses
        True for positions that can be attended to.

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
            None: SDPA does not return attention weights.
        """
        attn_mask = None
        is_causal = self.use_mask

        if self.use_mask and mask is not None:
            attn_mask = ~mask if mask.dtype == torch.bool else mask
            is_causal = False

        dropout_p = (
            self.dropout.p
            if isinstance(self.dropout, torch.nn.Dropout)
            else 0.0
        )
        attn_outputs = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p if self.training else 0.0,
            is_causal=is_causal,
            scale=scale_factor,
        )

        return attn_outputs, None
