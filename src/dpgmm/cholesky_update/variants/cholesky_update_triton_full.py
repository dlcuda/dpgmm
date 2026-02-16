import torch
import triton
import triton.language as tl

from dpgmm.cholesky_update.cholesky_update_base import CholeskyUpdateBase


class CholeskyUpdateTritonFull(CholeskyUpdateBase):
    r"""
    Fully fused Triton implementation of the rank-1 Cholesky update.

    Unlike the standard Triton implementation which launches a kernel for every
    column, this implementation launches a single kernel per batch. The GPU
    thread handles the iteration over columns (0 to D) internally.

    This reduces CPU-side overhead (kernel launch latency) significantly for
    high-dimensional data.

    Equation:
        $$ L' L'^T = L L^T + m \cdot v v^T $$
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
        Executes the fully fused Cholesky update.

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

        self._cholesky_full_update_kernel[(B,)](
            chol_batch_ptr,
            omegas_ptr,
            multiplier_ptr,
            bs,
            result,
            D=D,
            BLOCK=BLOCK,
        )

        return result

    @staticmethod
    @triton.jit
    def _cholesky_full_update_kernel(
        chol_ptr,
        omega_ptr,
        multiplier_ptr,
        bs_ptr,
        result_ptr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """
        Fused Triton Kernel performing the full Cholesky update loop.
        This kernel iterates `i` from 0 to D internally.

        Args:
            chol_ptr (tl.pointer): Input Cholesky batch [B, D, D].
            omega_ptr (tl.pointer): Update vectors [B, D]. (Modified in-place).
            multiplier_ptr (tl.pointer): Multipliers [B].
            bs_ptr (tl.pointer): Scaling factors buffer [B].
            result_ptr (tl.pointer): Output Cholesky batch [B, D, D].
            D (int): Data dimension.
            BLOCK (int): Tile size (power of 2, >= D).
        """
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < D
        multiplier = tl.load(multiplier_ptr + pid)
        bs = tl.load(bs_ptr + pid)

        for i in range(D):
            chol_diag = tl.load(chol_ptr + pid * D * D + i * D + i)
            omega_i = tl.load(omega_ptr + pid * D + i)

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
            new_chol_from_k = (
                new_old_ratio * chol_from_k + second_ratio * new_omega_from_k
            )
            res_col = tl.where(mask_k, new_chol_from_k, res_col)

            tl.store(
                omega_ptr + pid * D + offs,
                tl.where(mask_k, new_omega_from_k, omega_from_k),
                mask=mask,
            )

            bs = bs + multiplier * omega_ratio * omega_ratio
            tl.store(bs_ptr + pid, bs)
            tl.store(result_ptr + pid * D * D + offs * D + i, res_col, mask=mask)
