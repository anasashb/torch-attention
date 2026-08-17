# Adapted from Informer2020:
# https://github.com/zhouhaoyi/Informer2020/blob/29f2a739226a509202a092b464163da81fa74960/models/attn.py
# https://github.com/zhouhaoyi/Informer2020/blob/29f2a739226a509202a092b464163da81fa74960/utils/masking.py
#
# Licensed under the Apache License, Version 2.0.
# This file has been modified for ADLERS.
# See LICENSES/Apache-2.0.txt and NOTICE.
#
# Paper:
# Informer: Beyond Efficient Transformer for Long Sequence Time-Series
# Forecasting
# https://arxiv.org/abs/2012.07436v3
#
# Equation and algorithm references below refer to this paper.

from math import sqrt

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from adlers.shared._attention_base import AttentionBase


class ProbMask:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        num_queries: int,
        query_indices: Tensor,
        scores: Tensor,
        device: torch.device | str = "cpu",
    ) -> None:
        _mask = (
            torch.ones(num_queries, scores.shape[-1], dtype=torch.bool)
            .to(device)
            .triu(1)
        )
        _mask_ex = _mask[None, None, :].expand(
            batch_size,
            num_heads,
            num_queries,
            scores.shape[-1],
        )
        indicator = _mask_ex[
            torch.arange(batch_size)[:, None, None],
            torch.arange(num_heads)[None, :, None],
            query_indices,
            :,
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self) -> Tensor:
        return self._mask


class ProbSparseAttention(nn.Module):
    def __init__(
        self,
        is_causal=True,
        factor=5,
        custom_scale_factor=None,
        dropout_rate=0.1,
        output_attention_scores=False,
        strict_mode=True,
    ):
        if factor <= 0:
            raise ValueError(
                f"ProbSparse factor must be greater than 0; got {factor}."
            )

        super().__init__()
        self.factor = factor
        self.custom_scale_factor = custom_scale_factor
        self.is_causal = is_causal
        self.output_attention_scores = output_attention_scores
        self.strict_mode = strict_mode
        self.dropout = nn.Dropout(dropout_rate)

    def _compute_top_query_scores(
        self,
        query: Tensor,
        key: Tensor,
        num_sampled_keys: int,
        num_top_queries: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Computes scores for queries selected by the ProbSparse attention
        mechanism.

        Keys are sampled for each query to calculate `sampled_query_key_scores`
        (`S_bar` in Algorithm 1) and `query_sparsity_measurements` (`M_bar` in
        Equation 4). The `top_queries` selected under these measurements
        (`Q_bar` in Algorithm 1) are then scored against every key.

        Args:
            query (Tensor): Query tensor of shape [batch_size, num_heads,
                num_queries, head_dim].
            key (Tensor): Key tensor of shape [batch_size, num_heads,
                num_keys, head_dim].
            num_sampled_keys (int): Number of keys sampled for each query.
            num_top_queries (int): Number of queries selected using the
                approximate sparsity measurements.

        Returns:
            top_query_scores (Tensor): Scores of shape [batch_size, num_heads,
                num_top_queries, num_keys].
            top_query_indices (Tensor): Selected query indices of shape
                [batch_size, num_heads, num_top_queries].
        """
        batch_size, num_heads, num_keys, head_dim = key.shape
        _, _, num_queries, _ = query.shape

        # Add a query axis so each query can gather its own sampled keys
        expanded_key = key.unsqueeze(-3).expand(
            batch_size,
            num_heads,
            num_queries,
            num_keys,
            head_dim,
        )

        # One row of sampled key positions per query
        sampled_key_indices = torch.randint(
            num_keys,
            (num_queries, num_sampled_keys),
        )

        # Pair each query with its row of sampled key positions
        sampled_keys = expanded_key[
            :,
            :,
            torch.arange(num_queries).unsqueeze(1),
            sampled_key_indices,
            :,
        ]

        # Sampled query-key scores (S_bar in Algorithm 1 of Informer paper)
        sampled_query_key_scores = torch.matmul(
            query.unsqueeze(-2), sampled_keys.transpose(-2, -1)
        ).squeeze(-2)

        # Approximate query sparsity measurement (M_bar in Equation 4 of the
        # Informer paper)
        query_sparsity_measurements = sampled_query_key_scores.max(-1)[
            0
        ] - torch.div(sampled_query_key_scores.sum(-1), num_keys)

        # Select Top-u queries under M_bar (steps 4-5 of Algorithm 1 in the
        # Informer paper)
        top_query_indices = query_sparsity_measurements.topk(
            num_top_queries,
            sorted=False,
        )[1]

        # Top-u query vectors (Q_bar in Algorithm 1 of the Informer paper)
        top_queries = query[
            torch.arange(batch_size)[:, None, None],
            torch.arange(num_heads)[None, :, None],
            top_query_indices,
            :,
        ]

        # Scores for Q_bar against every key (used to compute S1 in step 6 of
        # Algorithm 1 in the Informer paper)
        top_query_scores = torch.matmul(
            top_queries,
            key.transpose(-2, -1),
        )

        return top_query_scores, top_query_indices

    def _make_default_context(
        self,
        value: Tensor,
        num_queries: int,
    ) -> Tensor:
        """
        Creates the default context for all queries.

        The default context (`S0` in Algorithm 1 of the Informer paper) gives
        queries a cheap initial output before attention is calculated for the
        selected queries. Unselected queries keep this output.

        For non-causal attention, each query starts with the mean value vector.
        For causal attention, each query starts with the cumulative sum of the
        values available up to its position.

        Args:
            value (Tensor): Value tensor of shape [batch_size, num_heads,
                num_values, head_dim].
            num_queries (int): Number of query positions.

        Returns:
            Tensor: Default context of shape [batch_size, num_heads,
                num_queries, head_dim].

        Raises:
            ValueError: If causal attention is used with different query and
                value sequence lengths.
        """
        batch_size, num_heads, num_values, head_dim = value.shape

        if not self.is_causal:
            mean_value = value.mean(dim=-2)
            context = (
                mean_value.unsqueeze(-2)
                .expand(batch_size, num_heads, num_queries, head_dim)
                .clone()
            )

        else:
            if num_queries != num_values:
                raise ValueError(
                    "Causal ProbSparse attention requires query and value "
                    "tensors to have the same sequence length; "
                    f"got query length {num_queries} and value length "
                    f"{num_values}."
                )

            context = value.cumsum(dim=-2)

        return context

    def _update_context_with_selected_queries(
        self,
        context: Tensor,
        value: Tensor,
        top_query_scores: Tensor,
        top_query_indices: Tensor,
        num_queries: int,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Updates the default context using the selected queries.

        The selected query scores are converted to attention weights and
        combined with the values to produce the selected-query context (`S1` in
        step 6 of Algorithm 1 in the Informer paper). These rows replace their
        corresponding rows in the default context (`S0`), producing the final
        context (`S` in step 8 of the algorithm).

        In causal attention, scores for future key positions are masked before
        the attention weights are calculated. When attention weights are
        requested, unselected queries retain uniform rows and selected queries
        receive their calculated weights.

        Args:
            context (Tensor): Default context tensor of shape [batch_size,
                num_heads, num_queries, head_dim]. Selected query rows are
                replaced in place.
            value (Tensor): Value tensor of shape [batch_size, num_heads,
                num_values, head_dim].
            top_query_scores (Tensor): Scores for the selected queries with
                shape [batch_size, num_heads, num_top_queries, num_values].
                Future positions are masked in place during causal attention.
            top_query_indices (Tensor): Selected query indices of shape
                [batch_size, num_heads, num_top_queries].
            num_queries (int): Number of query positions.

        Returns:
            context (Tensor): Final context tensor of shape [batch_size,
                num_heads, num_queries, head_dim].
            attn_weights (Tensor | None): Approximate dense attention weights,
                or None when attention weights are not requested.
        """
        batch_size, num_heads, num_values, _ = value.shape

        if self.is_causal:
            selected_query_causal_mask = ProbMask(
                batch_size,
                num_heads,
                num_queries,
                top_query_indices,
                top_query_scores,
                device=value.device,
            )

            top_query_scores.masked_fill_(
                selected_query_causal_mask.mask,
                -np.inf,
            )

        # Attention weights used to compute the selected-query context
        # (S1 in step 6 of Algorithm 1 in the Informer paper)
        top_query_weights = torch.softmax(top_query_scores, dim=-1)

        # Insert S1 into S0 at the selected rows to produce S
        # (step 8 of Algorithm 1 in the Informer paper)
        context[
            torch.arange(batch_size)[:, None, None],
            torch.arange(num_heads)[None, :, None],
            top_query_indices,
            :,
        ] = torch.matmul(top_query_weights, value).type_as(context)

        if self.output_attention_scores:
            attn_weights = (
                (
                    torch.ones([batch_size, num_heads, num_values, num_values])
                    / num_values
                )
                .type_as(top_query_weights)
                .to(top_query_weights.device)
            )

            attn_weights[
                torch.arange(batch_size)[:, None, None],
                torch.arange(num_heads)[None, :, None],
                top_query_indices,
                :,
            ] = top_query_weights

            return (context, attn_weights)

        else:
            return (context, None)

    def forward(self, query, key, value, attn_mask):
        if attn_mask is not None:
            raise ValueError(
                "ProbSparse attention does not support custom attention masks; "
                f"got shape {tuple(attn_mask.shape)}. Pass attn_mask=None."
            )

        # borrowing _validate_shapes from AttentionBase w/o inheriting yet
        if self.strict_mode:
            AttentionBase._validate_shapes(
                query=query,
                key=key,
                value=value,
                attn_mask=None,
            )

        batch_size, num_heads, num_queries, head_dim = query.shape
        _, _, num_keys, _ = key.shape

        num_sampled_keys = (
            self.factor * np.ceil(np.log(num_keys)).astype("int").item()
        )  # c*ln(L_k)
        num_top_queries = (
            self.factor * np.ceil(np.log(num_queries)).astype("int").item()
        )  # c*ln(L_q)

        num_sampled_keys = (
            num_sampled_keys if num_sampled_keys < num_keys else num_keys
        )
        num_top_queries = (
            num_top_queries if num_top_queries < num_queries else num_queries
        )

        top_query_scores, top_query_indices = self._compute_top_query_scores(
            query=query,
            key=key,
            num_sampled_keys=num_sampled_keys,
            num_top_queries=num_top_queries,
        )

        # add scale factor
        scale_factor = self.custom_scale_factor or 1.0 / sqrt(head_dim)
        if scale_factor is not None:
            top_query_scores = top_query_scores * scale_factor
        attn_output = self._make_default_context(
            value=value,
            num_queries=num_queries,
        )
        # update the context with selected top_k queries
        attn_output, attn_weights = self._update_context_with_selected_queries(
            attn_output,
            value,
            top_query_scores,
            top_query_indices,
            num_queries,
        )

        return attn_output, attn_weights
