import torch
import triton
import triton.language as tl

from dpgmm.cholesky_update.cholesky_update_base import CholeskyUpdateBase


class CholeskyUpdateTritonFused(CholeskyUpdateBase):
    r"""
    Triton-based implementation of the Rank-1 Cholesky update using a column-wise kernel strategy.

    Unlike the 'Full' implementation which fuses the entire loop into one kernel,
    this implementation keeps the loop over columns ($0 \dots D$) in Python,
    launching a separate Triton kernel for each column $i$.

    **Algorithm:**
    Given $L L^T + m \cdot v v^T$, this computes the new factor $L'$.
    It iterates $i$ from $0$ to $D-1$. In step $i$, it updates the diagonal $L_{ii}$,
    the column entries $L_{ki}$ (where $k > i$), and the remaining update vector elements.

    **Performance Note:**
    For very high dimensions ($D$), this may incur higher CPU overhead due to
    launching $D$ separate kernels. However, it provides a synchronization point
    between columns if global memory barriers are needed.

    Attributes:
        D (int): The dimensionality of the covariance matrix.
    """

    def __init__(self, device: torch.device):
        super().__init__(device=device)

    def cholesky_update(
        self,
        chol_batch: torch.Tensor,
        update_vectors: torch.Tensor,
        multiplier: torch.Tensor,
    ) -> torch.Tensor:
        """
        Executes the rank-1 Cholesky update.

        Args:
            chol_batch (torch.Tensor): The current lower triangular factors L.
                Shape: [B, D, D]. Must be contiguous.
            update_vectors (torch.Tensor): The update vectors v.
                Shape: [B, D].
            multiplier (torch.Tensor): The scalar coefficients m.
                Shape: [B].

        Returns:
            torch.Tensor: The updated Cholesky factors L'.
                Shape: [B, D, D].
        """
        if self.device != chol_batch.device:
            raise ValueError(
                f"Device mismatch: Cholesky batch is on {chol_batch.device}, but updater is on {self.device}"
            )
        B = chol_batch.shape[0]
        D = chol_batch.shape[1]

        chol_batch_ptr = chol_batch.contiguous()
        update_vectors_ptr = update_vectors.contiguous()
        multiplier_ptr = multiplier.contiguous()

        bs = torch.ones_like(multiplier_ptr, device=self.device)
        omegas_ptr = update_vectors_ptr.clone()
        result = torch.zeros_like(chol_batch_ptr)

        BLOCK = triton.next_power_of_2(D)

        for i in range(D):
            self._cholesky_column_update_kernel[(B,)](
                chol_batch_ptr,
                omegas_ptr,
                multiplier_ptr,
                bs,
                result,
                i,
                D=D,
                BLOCK=BLOCK,
            )

        return result

    @staticmethod
    @triton.jit
    def _cholesky_column_update_kernel(
        chol_ptr,  # float32[B, D, D]
        omega_ptr,  # float32[B, D]
        multiplier_ptr,  # float32[B]
        bs_ptr,  # float32[B]
        result_ptr,  # float32[B, D, D]
        i: tl.constexpr,  # scalar
        D: tl.constexpr,  # data_dim
        BLOCK: tl.constexpr,  # power-of-two tile size >= D
    ):
        r"""
        Triton kernel to update a specific column `i` across the entire batch.

        This kernel performs the following operations for the $i$-th iteration:
        1. Loads the diagonal $L_{ii}$ and update vector component $v_i$.
        2. Computes the new diagonal $L'_{ii}$.
        3. Updates the column elements $L_{ki}$ for $k > i$.
        4. Updates the vector components $v_k$ for $k > i$ (preparing for future iterations).
        5. Updates the scalar tracker `bs`.

        Args:
            chol_ptr (Tensor): Pointer to input Cholesky matrices [B, D, D].
            omega_ptr (Tensor): Pointer to update vectors [B, D]. Modified in-place.
            multiplier_ptr (Tensor): Pointer to scalar multipliers [B].
            bs_ptr (Tensor): Pointer to scaling factors [B]. Modified in-place.
            result_ptr (Tensor): Pointer to output Cholesky matrices [B, D, D].
            i (int): The current column index being processed (0 <= i < D).
            D (int): Matrix dimension.
            BLOCK (int): Power-of-two block size covering D.
        """
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < D

        chol_diag = tl.load(chol_ptr + pid * D * D + i * D + i)
        omega_i = tl.load(omega_ptr + pid * D + i)
        multiplier = tl.load(multiplier_ptr + pid)
        bs = tl.load(bs_ptr + pid)

        chol_diag_sq = chol_diag * chol_diag
        omega_i_sq = omega_i * omega_i

        under_sqrt = chol_diag_sq + (multiplier / bs) * omega_i_sq
        under_sqrt = tl.maximum(under_sqrt, 1e-3)
        new_chol_diag = tl.sqrt(under_sqrt)

        gamma = chol_diag_sq * bs + multiplier * omega_i_sq
        omega_ratio = omega_i / chol_diag
        new_old_ratio = new_chol_diag / chol_diag
        second_ratio = new_chol_diag * multiplier * omega_i / gamma

        res_col = tl.zeros([BLOCK], dtype=tl.float32)
        res_col = tl.where((offs == i) & mask, new_chol_diag, res_col)

        mask_k = (offs > i) & mask
        chol_from_k = tl.load(
            chol_ptr + pid * D * D + offs * D + i, mask=mask_k, other=0.0
        )
        omega_from_k = tl.load(omega_ptr + pid * D + offs, mask=mask, other=0.0)

        new_omega_from_k = omega_from_k - omega_ratio * chol_from_k
        new_chol_from_k = new_old_ratio * chol_from_k + second_ratio * new_omega_from_k
        res_col = tl.where(mask_k, new_chol_from_k, res_col)

        tl.store(
            omega_ptr + pid * D + offs,
            tl.where(mask_k, new_omega_from_k, omega_from_k),
            mask=mask,
        )

        updated_bs = bs + multiplier * omega_ratio * omega_ratio
        tl.store(bs_ptr + pid, updated_bs)
        tl.store(result_ptr + pid * D * D + offs * D + i, res_col, mask=mask)
