"""Internal typing helpers for ADLERS."""

from typing import Literal, TypeAlias

AttentionBackend: TypeAlias = Literal["einsum", "sdpa"]
