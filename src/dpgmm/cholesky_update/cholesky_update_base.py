from abc import ABC, abstractmethod

import torch
from torch import Tensor


class CholeskyUpdateBase(ABC):
    r"""
    Abstract base class for performing rank-1 Cholesky updates.

    This class defines the interface for updating a Cholesky factorization $L$
    such that $L'L'^T = LL^T + m \cdot v v^T$, where $L$ is a lower triangular matrix,
    $v$ is an update vector, and $m$ is a scalar multiplier.
    """

    def __init__(self, device: torch.device):
        """
        Initializes the Cholesky updater.
        """
        self.device = device

    @abstractmethod
    def cholesky_update(
        self,
        chol_batch: Tensor,  # [B, D, D]
        update_vectors: Tensor,  # [B, D]
        multiplier: Tensor,  # [B]
    ) -> Tensor:
        """
        Abstract method to compute the updated Cholesky factors.

        Args:
            chol_batch (Tensor): The batch of original Cholesky factors $L$.
                Shape: $[B, D, D]$.
            update_vectors (Tensor): The batch of vectors $v$ to update with.
                Shape: $[B, D]$.
            multiplier (Tensor): The batch of scalars $m$ (coefficients).
                Shape: $[B]$.

        Returns:
            Tensor: The updated Cholesky factors $L'$. Shape: $[B, D, D]$.
        """
        pass
