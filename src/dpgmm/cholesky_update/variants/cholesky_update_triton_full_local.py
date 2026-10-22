import torch
import triton
import triton.language as tl

from dpgmm.cholesky_update.cholesky_update_base import CholeskyUpdateBase


class CholeskyUpdateTritonFullLocal(CholeskyUpdateBase):
    r"""
    Fused rank-1 Cholesky update with program-local recurrence state.

    Unlike the standard Triton implementation which launches a kernel for every
    column, this implementation launches a single kernel per batch. The GPU
    thread handles the iteration over columns (0 to D) internally.

    This reduces CPU-side overhead (kernel launch latency) significantly for
    high-dimensional data.

    The working vector and scale accumulator remain program-local throughout
    the recurrence. No mutable global-memory scratch buffers are needed.

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

        result = torch.zeros_like(chol_batch_ptr)

        BLOCK = triton.next_power_of_2(D)

        self._cholesky_full_update_kernel[(B,)](
            chol_batch_ptr,
            update_vectors_ptr,
            multiplier_ptr,
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
        result_ptr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """
        Fused Triton Kernel performing the full Cholesky update loop.
        This kernel iterates `i` from 0 to D internally.

        Args:
            chol_ptr (tl.pointer): Input Cholesky batch [B, D, D].
            omega_ptr (tl.pointer): Read-only update vectors [B, D].
            multiplier_ptr (tl.pointer): Multipliers [B].
            result_ptr (tl.pointer): Output Cholesky batch [B, D, D].
            D (int): Data dimension.
            BLOCK (int): Tile size (power of 2, >= D).
        """
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < D
        multiplier = tl.load(multiplier_ptr + pid)
        bs = tl.full((), 1.0, tl.float32)
        omegas = tl.load(omega_ptr + pid * D + offs, mask=mask, other=0.0)

        for i in range(D):
            chol_diag = tl.load(chol_ptr + pid * D * D + i * D + i)
            # Keep the recurrence in program-local values. Triton handles the
            # communication needed by this reduction, so the next column never
            # reads another warp's unsynchronized global-memory scratch writes.
            omega_i = tl.sum(tl.where(offs == i, omegas, 0.0), axis=0)

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
            new_omega_from_k = omegas - omega_ratio * chol_from_k
            new_chol_from_k = (
                new_old_ratio * chol_from_k + second_ratio * new_omega_from_k
            )
            res_col = tl.where(mask_k, new_chol_from_k, res_col)

            omegas = tl.where(mask_k, new_omega_from_k, omegas)

            bs = bs + multiplier * omega_ratio * omega_ratio
            tl.store(result_ptr + pid * D * D + offs * D + i, res_col, mask=mask)
