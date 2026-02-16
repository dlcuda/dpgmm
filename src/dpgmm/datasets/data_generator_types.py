from typing import List, Set, TypedDict

import numpy as np


class SyntheticDataset(TypedDict):
    data: np.ndarray  # Shape (N, dim)
    centers: np.ndarray  # Shape (K, dim)
    assignment: List[Set[int]]
