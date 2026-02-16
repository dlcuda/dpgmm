from __future__ import annotations

from typing import List

import numpy as np
import torch
import triton
import triton.language as tl
from torch import Tensor

from dpgmm.samplers.cgs.state import StudentTCalculator
from dpgmm.samplers.cgs.variants.base.utils import init_kappa_0, init_nu_0


class FullCovarianceStudentTCalculatorTriton(StudentTCalculator):
    """
    Calculates the Student-t log probability density for full covariance matrices,
    utilizing a custom Triton kernel specifically for the batched vector norm computation.
    """

    def __init__(self, device: torch.device, data_dim: int):
        """
        Initializes the calculator and compiles the Triton kernel.

        Args:
            device (torch.device, optional): The computing device. Defaults to the
                system's available device.
        """
        self.device = device
        self.nu0 = torch.tensor(
            init_nu_0(data_dim=data_dim), dtype=torch.float32, device=self.device
        )
        self.data_dim_tensor = torch.tensor(data_dim, device=self.device)
        self._define_batched_vector_norm_kernel()

    def log_pdf(
        self, data_batch: Tensor, parameters: List[Tensor], cluster_counts: Tensor
    ) -> Tensor:
        """
        Interface method to trigger the Triton-accelerated log PDF calculation.

        Args:
            data_batch (Tensor): The input data batch of shape (batch_size, data_dim).
            parameters (List[Tensor]): A list containing the mean matrix and Cholesky
                covariance matrices.
            cluster_counts (Tensor): The number of data points currently in each cluster.

        Returns:
            Tensor: Log probabilities of shape (batch_size, clusters_num).
        """
        if data_batch.device != self.device:
            raise ValueError(
                f"Data batch device {data_batch.device} does not match calculator device {self.device}"
            )

        mean_matrix, chol_cov_matrices = parameters
        result = self.t_student_log_pdf_triton(
            mean_matrix, chol_cov_matrices, data_batch, cluster_counts
        )
        return result

    def t_student_log_pdf_triton(
        self,
        mean_matrix: Tensor,
        chol_cov_matrices: Tensor,
        data_batch: Tensor,
        cluster_counts: Tensor,
    ) -> Tensor:
        """
        Executes the mathematical operations for the Student-t log PDF, delegating the
        vector norm squaring step to a Triton kernel.

        Args:
            mean_matrix (Tensor): Cluster means of shape (clusters_num, data_dim).
            chol_cov_matrices (Tensor): Cholesky factors of cluster covariances of
                shape (clusters_num, data_dim, data_dim).
            data_batch (Tensor): The data batch to evaluate.
            cluster_counts (Tensor): Point counts per cluster.

        Returns:
            Tensor: The computed log probabilities.
        """
        data_dim = data_batch.shape[-1]
        clusters_num = mean_matrix.size(0)

        # Degrees of freedom (nu) and kappas
        nus = self.nu0 - (self.data_dim_tensor - 1) + cluster_counts
        kappas = (
            torch.tensor(init_kappa_0(), dtype=torch.float32, device=self.device)
            + cluster_counts
        )

        # Scaled Cholesky covariances
        scale_fact = ((kappas + 1.0) / (kappas * nus)).view(-1, 1, 1).to(self.device)
        chol_cov_scaled = torch.sqrt(scale_fact) * chol_cov_matrices

        # Log determinants
        chol_cov_diagonals = torch.diagonal(chol_cov_scaled, dim1=-2, dim2=-1)
        log_dets_sqrt = torch.sum(torch.log(chol_cov_diagonals), dim=-1)

        # Normalize data
        data_batch_tiled = data_batch.unsqueeze(1).expand(-1, clusters_num, -1)
        data_batch_norm = data_batch_tiled - mean_matrix
        data_batch_norm_transposed = data_batch_norm.permute(1, 2, 0)

        # Triangular solve
        vecs = torch.linalg.solve_triangular(
            chol_cov_scaled, data_batch_norm_transposed, upper=False
        )
        vecs = vecs.permute(2, 0, 1).contiguous()  # [B, C, D]

        # Triton norm computation
        B, C, D = vecs.shape
        vecs_norm = torch.empty((B, C), device=self.device, dtype=torch.float32)
        grid = lambda meta: (B, C)  # noqa: E731

        self._batched_vector_norm_kernel[grid](
            vecs,
            vecs_norm,
            vecs.stride(1),  # stride_bc
            vecs.stride(2),  # stride_cd
            vecs.stride(0),  # stride_bd,
            C,
            D,
            BLOCK_SIZE=128,
        )

        # Student-t log-pdf
        nus = nus.unsqueeze(0)  # shape: [1, C]
        num = torch.lgamma((nus + data_dim) / 2.0).to(self.device)

        denom = (
            torch.lgamma(nus / 2.0)
            + (data_dim / 2.0)
            * (torch.log(nus) + torch.tensor(np.pi).log().to(self.device))
            + log_dets_sqrt
            + ((nus + data_dim) / 2.0) * torch.log1p((vecs_norm / torch.sqrt(nus)) ** 2)
        )

        return num - denom

    def _define_batched_vector_norm_kernel(self):
        """Compiles the Triton JIT kernel for computing batched vector norms."""

        @triton.jit
        def _batched_vector_norm_kernel(
            x_ptr,  # [B, C, D]
            y_ptr,  # [B, C]
            stride_bc,
            stride_cd,
            stride_bd,
            C,
            D: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
        ):
            b = tl.program_id(0)
            c = tl.program_id(1)
            offs_d = tl.arange(0, BLOCK_SIZE)

            x_ptrs = x_ptr + b * stride_bd + c * stride_bc + offs_d * stride_cd
            mask = offs_d < D
            x = tl.load(x_ptrs, mask=mask, other=0.0)
            acc = tl.sum(x * x, axis=0)
            tl.store(y_ptr + b * C + c, tl.sqrt(acc))

        self._batched_vector_norm_kernel = _batched_vector_norm_kernel

    def to(self, device: torch.device) -> FullCovarianceStudentTCalculatorTriton:
        self.device = device
        self.nu0 = torch.as_tensor(self.nu0, device=self.device)
        self.data_dim_tensor = torch.as_tensor(self.data_dim_tensor, device=self.device)
        return self
