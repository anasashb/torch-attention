# Adapted from fast-transformers:
# https://github.com/idiap/fast-transformers/blob/2ad36b97e64cb93862937bd21fcc9568d989561f/fast_transformers/attention/linear_attention.py
#
# Licensed under the MIT License.
# This file has been modified for ADLERS.
# See LICENSES/fast-transformers-MIT.txt and NOTICE.
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement unmasked linear attention."""

from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import Module

from adlers.shared._attention_base import AttentionBase


def _elu_feature_map(tensor: Tensor) -> Tensor:
    return torch.nn.functional.elu(tensor) + 1


class LinearAttention(Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.

    Given the queries, keys and values as Q, K, V instead of computing

        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),

    we make use of a feature map function Φ(.) and perform the following
    computation

        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).

    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """

    def __init__(
        self,
        feature_map: Callable[[Tensor], Tensor] | None = None,
        eps: float = 1e-6,
        dropout_rate: float = 0.0,
        output_attention_scores: bool = False,
    ) -> None:
        if dropout_rate != 0.0:
            raise ValueError(
                "Linear attention does not support dropout; "
                f"got dropout_rate {dropout_rate}. Set dropout_rate=0.0."
            )

        if output_attention_scores:
            raise ValueError(
                "Linear attention does not support returning attention scores. "
                "Set output_attention_scores=False."
            )

        super().__init__()
        self.feature_map = (
            feature_map if feature_map is not None else _elu_feature_map
        )
        self.eps = eps

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, None]:
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
        AttentionBase._validate_qkv_head_counts(
            query=query,
            key=key,
            value=value,
        )
        AttentionBase._validate_kv_sequence_lengths(
            key=key,
            value=value,
        )

        # Apply the feature map to the query and key
        mapped_query = self.feature_map(query)
        mapped_key = self.feature_map(key)

        if attn_mask is not None:
            AttentionBase._validate_attn_mask_dtype(attn_mask=attn_mask)
            expected_mask_shape = (
                query.shape[0],
                1,
                1,
                key.shape[-2],
            )
            if attn_mask.shape != expected_mask_shape:
                raise ValueError(
                    "Linear attention only supports key-padding masks shaped "
                    "[batch_size, 1, 1, num_keys]; "
                    f"got shape {tuple(attn_mask.shape)}."
                )

            key_padding_mask = attn_mask.squeeze(dim=-2).unsqueeze(dim=-1)
            mapped_key = mapped_key.masked_fill(key_padding_mask, 0)

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        key_value_product = torch.einsum(
            "bhsd,bhsm->bhmd",
            mapped_key,
            value,
        )

        # Compute the normalizer
        normalization_factor = 1 / (
            torch.einsum(
                "bhld,bhd->bhl",
                mapped_query,
                mapped_key.sum(dim=-2),
            )
            + self.eps
        )

        # Finally compute and return the new values
        attn_output = torch.einsum(
            "bhld,bhmd,bhl->bhlm",
            mapped_query,
            key_value_product,
            normalization_factor,
        )

        return attn_output.contiguous(), None
