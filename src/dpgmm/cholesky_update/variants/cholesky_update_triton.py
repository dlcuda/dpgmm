import torch
import triton
import triton.language as tl

from dpgmm.cholesky_update.cholesky_update_base import CholeskyUpdateBase


class CholeskyUpdateTriton(CholeskyUpdateBase):
    r"""
    Triton-accelerated implementation of the rank-1 Cholesky update.

    This class uses OpenAI Triton to fuse vector operations on the GPU.
    It computes L' such that:
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
        Orchestrates the Triton kernel execution for the rank-1 update.

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

        bs = torch.ones_like(multiplier, device=self.device)
        omegas = update_vectors_ptr.clone()
        result = torch.zeros_like(chol_batch)

        BLOCK = triton.next_power_of_2(D)

        for i in range(D):
            out_column = torch.zeros((B, D), dtype=torch.float32, device=self.device)
            new_omega = torch.zeros_like(omegas)
            new_bs = torch.zeros_like(bs)

            self._cholesky_column_update_kernel[(B,)](
                chol_batch_ptr,
                omegas,
                multiplier_ptr,
                bs,
                out_column,
                new_omega,
                new_bs,
                i,
                D=D,
                BLOCK=BLOCK,
            )

            result[:, :, i] = out_column
            omegas = new_omega
            bs = new_bs

        return result

    @staticmethod
    @triton.jit
    def _cholesky_column_update_kernel(
        chol_ptr,
        omega_ptr,
        multiplier_ptr,
        bs_ptr,
        out_ptr,
        new_omega_ptr,
        new_bs_ptr,
        i: tl.constexpr,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """
        Triton Kernel for updating a single column 'i' across the entire batch.

        Args:
            chol_ptr (Tensor): Pointer to Cholesky batch. Shape [B, D, D].
            omega_ptr (Tensor): Pointer to current omegas. Shape [B, D].
            multiplier_ptr (Tensor): Pointer to multipliers. Shape [B].
            bs_ptr (Tensor): Pointer to current scaling factors. Shape [B].
            out_ptr (Tensor): Pointer to output column buffer. Shape [B, D].
            new_omega_ptr (Tensor): Pointer to next step omegas. Shape [B, D].
            new_bs_ptr (Tensor): Pointer to next step scaling factors. Shape [B].
            i (int): Scalar index of the current column being updated.
            D (int): Data dimension size (data_dim).
            BLOCK (int): Power-of-two block size for tiling (>= D).
        """
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < D

        chol_diag = tl.load(chol_ptr + pid * D * D + i * D + i)
        omega = tl.load(omega_ptr + pid * D + i)
        multiplier = tl.load(multiplier_ptr + pid)
        bs = tl.load(bs_ptr + pid)

        chol_diag_sq = chol_diag * chol_diag
        omega_sq = omega * omega

        under_sqrt = chol_diag_sq + (multiplier / bs) * omega_sq
        under_sqrt = tl.maximum(under_sqrt, 1e-3)
        new_chol_diag = tl.sqrt(under_sqrt)

        gamma = chol_diag_sq * bs + multiplier * omega_sq
        omega_ratio = omega / chol_diag
        new_old_ratio = new_chol_diag / chol_diag
        second_ratio = new_chol_diag * multiplier * omega / gamma

        result = tl.zeros([BLOCK], dtype=tl.float32)
        result = tl.where((offs == i) & mask, new_chol_diag, result)

        mask_k = (offs > i) & mask
        chol_from_k = tl.load(
            chol_ptr + pid * D * D + offs * D + i, mask=mask_k, other=0.0
        )
        omega_from_k = tl.load(omega_ptr + pid * D + offs, mask=mask_k, other=0.0)

        new_omega_from_k = omega_from_k - omega_ratio * chol_from_k
        new_chol_from_k = new_old_ratio * chol_from_k + second_ratio * new_omega_from_k
        result = tl.where(mask_k, new_chol_from_k, result)

        omega_base = new_omega_ptr + pid * D
        tl.store(
            omega_base + offs,
            tl.where(mask_k, new_omega_from_k, omega),
            mask=mask,
        )

        updated_bs = bs + multiplier * omega_ratio * omega_ratio
        tl.store(new_bs_ptr + pid, updated_bs)
        tl.store(out_ptr + pid * D + offs, result, mask=mask)
