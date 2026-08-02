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


class ProbAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

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

        if not self.mask_flag:
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
        context,
        value,
        top_query_scores,
        top_query_indices,
        num_queries,
        attn_mask,
    ):
        batch_size, num_heads, num_values, head_dim = value.shape

        if self.mask_flag:
            attn_mask = ProbMask(
                batch_size,
                num_heads,
                num_queries,
                top_query_indices,
                top_query_scores,
                device=value.device,
            )
            top_query_scores.masked_fill_(attn_mask.mask, -np.inf)

        top_query_weights = torch.softmax(
            top_query_scores, dim=-1
        )  # nn.Softmax(dim=-1)(scores)

        context[
            torch.arange(batch_size)[:, None, None],
            torch.arange(num_heads)[None, :, None],
            top_query_indices,
            :,
        ] = torch.matmul(top_query_weights, value).type_as(context)
        if self.output_attention:
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

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = (
            self.factor * np.ceil(np.log(L_K)).astype("int").item()
        )  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype("int").item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._compute_top_query_scores(
            query=queries,
            key=keys,
            num_sampled_keys=U_part,
            num_top_queries=u,
        )

        # add scale factor
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._make_default_context(
            value=values,
            num_queries=L_Q,
        )
        # update the context with selected top_k queries
        context, attn = self._update_context_with_selected_queries(
            context, values, scores_top, index, L_Q, attn_mask
        )

        return context.transpose(2, 1).contiguous(), attn
