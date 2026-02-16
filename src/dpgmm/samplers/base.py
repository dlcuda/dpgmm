from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Set, TypedDict

import torch


class BaseSamplerFitResult(TypedDict):
    cluster_params: Dict[str, List[torch.Tensor]]
    cluster_assignment: List[Set[int]]
    alpha: float


class BaseSampler(Protocol):
    def fit(
        self, iterations_num: int, data: torch.Tensor, out_dir: Optional[str] = None
    ) -> BaseSamplerFitResult: ...

    def to(self, device: torch.device) -> BaseSampler: ...
