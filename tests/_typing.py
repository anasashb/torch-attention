from collections.abc import Callable
from typing import TypeAlias

import torch

QKVTensors: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
MakeQKV: TypeAlias = Callable[..., QKVTensors]
