from typing import List, Tuple

import torch
from torch import Tensor

from dpgmm.cholesky_update.cholesky_update_base import CholeskyUpdateBase


class CholeskyUpdateTorch(CholeskyUpdateBase):
    """
    PyTorch implementation of the rank-1 Cholesky update.

    This implementation computes the update column-by-column to preserve the
    triangular structure without recomputing the full Cholesky decomposition.
    """

    def calculate_new_column(
        self,
        curr_i: int,
        columns_ta: List[Tensor],
        omegas: Tensor,
        bs: Tensor,
        chol_diagonals: Tensor,
        batch_size: int,
        chol_batch: Tensor,
        multiplier: Tensor,
        dim: int,
    ) -> Tuple[List[Tensor], Tensor, Tensor]:
        r"""
        Calculates the $i$-th column of the updated Cholesky matrix.

        This method performs the core mathematical logic for the update step at index `curr_i`.
        It computes the new diagonal element, updates the `omegas` (transformed update vectors)
        for subsequent steps, and calculates the new off-diagonal elements for the current column.

        Mathematical stability is enforced using `torch.maximum` on the diagonal squared elements
        to prevent non-positive definite results during downdates.

        Args:
            curr_i (int): The index of the current column being calculated ($0 \le i < D$).
            columns_ta (List[Tensor]): A list accumulating the calculated columns of the new matrix $L'$.
            omegas (Tensor): The transformed update vectors at the current step. Shape: $[B, D]$.
            bs (Tensor): The accumulated scaling factors (denominator) for the update. Shape: $[B]$.
            chol_diagonals (Tensor): The diagonals of the original Cholesky matrices. Shape: $[B, D]$.
            batch_size (int): The size of the batch $B$.
            chol_batch (Tensor): The original Cholesky matrices $L$. Shape: $[B, D, D]$.
            multiplier (Tensor): The update coefficients $m$. Shape: $[B]$.
            dim (int): The dimensionality of the data $D$.

        Returns:
            Tuple[List[Tensor], Tensor, Tensor]:
                1. Updated `columns_ta` with the new column appended.
                2. Updated `omegas` for the next iteration.
                3. Updated `bs` (scaling factors) for the next iteration.
        """
        chol_diagonals_at_i = chol_diagonals[:, curr_i]
        omegas_at_i = omegas[:, curr_i]
        under_sqrt_expr = torch.square(chol_diagonals_at_i) + (
            multiplier / bs
        ) * torch.square(omegas_at_i)
        under_sqrt_expr = torch.maximum(under_sqrt_expr, torch.tensor(1.0e-3))
        new_chol_diagonals_at_i = torch.sqrt(under_sqrt_expr)
        gamma = torch.square(chol_diagonals_at_i) * bs + multiplier * torch.square(
            omegas_at_i
        )
        leading_zeros = torch.zeros(
            (batch_size, curr_i), dtype=torch.float32, device=self.device
        )

        omega_diag_at_i_ratio = torch.unsqueeze(
            omegas_at_i / chol_diagonals_at_i, dim=1
        )
        new_old_diagonals_ratio = torch.unsqueeze(
            new_chol_diagonals_at_i / chol_diagonals_at_i, dim=1
        )
        second_ratio = torch.unsqueeze(
            new_chol_diagonals_at_i * multiplier * omegas_at_i / gamma, dim=1
        )

        if curr_i < dim - 1:
            k = curr_i + 1
            omegas_from_k = omegas[:, k:]
            # new_omegas_from_k.shape -> (batch_size, dim - k)
            new_omegas_from_k = (
                omegas_from_k - omega_diag_at_i_ratio * chol_batch[:, k:, curr_i]
            )
            # new_chol_from_k.shape -> (batch_size, dim - k)
            new_chol_from_k = new_old_diagonals_ratio * chol_batch[:, k:, curr_i]
            new_chol_from_k += second_ratio * new_omegas_from_k
            # new_chol_column.shape -> (batch_size, dim)
            new_chol_column = torch.concat(
                [
                    leading_zeros,
                    torch.unsqueeze(new_chol_diagonals_at_i, dim=1),
                    new_chol_from_k,
                ],
                dim=-1,
            )
        else:
            # curr_i points at last column
            new_chol_column = torch.concat(
                [leading_zeros, torch.unsqueeze(new_chol_diagonals_at_i, dim=1)], dim=-1
            )
            new_omegas_from_k = torch.zeros(
                (batch_size, 0), dtype=torch.float32, device=self.device
            )

        new_omegas = torch.concat([omegas[:, : curr_i + 1], new_omegas_from_k], dim=1)
        new_bs = bs + multiplier * torch.square(
            torch.reshape(omega_diag_at_i_ratio, [-1])
        )
        columns_ta.append(new_chol_column)

        new_omegas.reshape((-1, dim))

        return columns_ta, new_omegas, new_bs

    def cholesky_update(
        self, chol_batch: Tensor, update_vectors: Tensor, multiplier: Tensor
    ) -> Tensor:
        r"""
        Performs the rank-1 Cholesky update on a batch of matrices.

        Computes $L'$ such that:
        $$ L' L'^T = L L^T + m \cdot v v^T $$

        Where $L$ is `chol_batch`, $v$ is `update_vectors`, and $m$ is `multiplier`.

        Args:
            chol_batch (Tensor): The current lower triangular Cholesky factors.
                Shape: $[B, D, D]$.
            update_vectors (Tensor): The vectors to be added (rank-1 update).
                Shape: $[B, D]$.
            multiplier (Tensor): Scalars determining the weight and sign of the update.
                Positive values indicate an update, negative values indicate a downdate.
                Shape: $[B]$.

        Returns:
            Tensor: The new lower triangular Cholesky factors $L'$.
                Shape: $[B, D, D]$.
        """
        if self.device != chol_batch.device:
            raise ValueError(
                f"Device mismatch: Cholesky batch is on {chol_batch.device}, but updater is on {self.device}"
            )

        batch_size = chol_batch.shape[0]
        dim = chol_batch.shape[2]

        chol_diagonals = chol_batch.diagonal(dim1=-2, dim2=-1)
        columns: list[Tensor] = []
        omegas = update_vectors
        bs = torch.ones_like(multiplier, device=self.device)

        for i in range(dim):
            columns, omegas, bs = self.calculate_new_column(
                i,
                columns,
                omegas,
                bs,
                chol_diagonals,
                batch_size,
                chol_batch,
                multiplier,
                dim,
            )

        first_columns_ordered = torch.stack(columns)
        result = first_columns_ordered.permute(1, 2, 0)

        return result
