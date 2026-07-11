"""Internal typing helpers for torch-attention."""

from typing import Literal, TypeAlias

AttentionBackend: TypeAlias = Literal["einsum", "sdpa"]
