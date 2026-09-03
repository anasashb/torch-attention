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

from typing import Any

import torch
from torch import Tensor
from torch.nn import Module


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

    # Any preserves temporary fast-transformers types until API adaptation.
    def __init__(
        self,
        query_dimensions: int,
        feature_map: Any | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else _elu_feature_map
        )
        self.eps = eps

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Any,
        query_lengths: Any,
        key_lengths: Any,
    ) -> Tensor:
        # Apply the feature map to the query and key
        mapped_query = self.feature_map(query)
        mapped_key = self.feature_map(key)

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones
        if not attn_mask.all_ones:
            raise RuntimeError(
                "LinearAttention does not support arbitrary attention masks"
            )
        mapped_key = mapped_key * key_lengths.float_matrix[:, :, None, None]

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        key_value_product = torch.einsum(
            "nshd,nshm->nhmd",
            mapped_key,
            value,
        )

        # Compute the normalizer
        normalization_factor = 1 / (
            torch.einsum(
                "nlhd,nhd->nlh",
                mapped_query,
                mapped_key.sum(dim=1),
            )
            + self.eps
        )

        # Finally compute and return the new values
        attn_output = torch.einsum(
            "nlhd,nhmd,nlh->nlhm",
            mapped_query,
            key_value_product,
            normalization_factor,
        )

        return attn_output.contiguous()
