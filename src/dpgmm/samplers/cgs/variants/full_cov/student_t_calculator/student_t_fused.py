from __future__ import annotations

from typing import List

import numpy as np
import torch
import triton
import triton.language as tl
from torch import Tensor

from dpgmm.samplers.cgs.state import StudentTCalculator
from dpgmm.samplers.cgs.variants.base.utils import init_kappa_0, init_nu_0


class FullCovarianceStudentTCalculatorTritonFused(StudentTCalculator):
    """
    Highly optimized Student-t log probability calculator that utilizes a fused
    Triton kernel to perform the vector norm and PDF evaluation in a single pass.
    """

    def __init__(self, device: torch.device, data_dim: int):
        """
        Initializes the calculator and compiles the fused Triton kernel.

        Args:
            device (torch.device, optional): The computing device. Defaults to the
                system's available device.
        """
        self.device = device
        self.data_dim = data_dim
        self.data_dim_tensor = torch.tensor(data_dim, device=self.device)
        self.nu0 = torch.tensor(
            init_nu_0(data_dim=data_dim), dtype=torch.float32, device=self.device
        )

        self.kappa_0 = torch.tensor(
            init_kappa_0(), dtype=torch.float32, device=self.device
        )
        self.log_pi = float(np.log(np.pi))
        self.half_data_dim = float(data_dim / 2.0)

        self._define_fused_kernel()

    def log_pdf(
        self, data_batch: Tensor, parameters: List[Tensor], cluster_counts: Tensor
    ) -> Tensor:
        """
        Interface method to trigger the fused Triton-accelerated log PDF calculation.

        Args:
            data_batch (Tensor): The input data batch.
            parameters (List[Tensor]): List containing mean and Cholesky covariance tensors.
            cluster_counts (Tensor): Point counts per cluster.

        Returns:
            Tensor: Log probabilities.
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
        Prepares the parameters and launches the fused Triton kernel to calculate
        the final log probabilities.

        Args:
            mean_matrix (Tensor): Cluster means.
            chol_cov_matrices (Tensor): Cholesky factors of cluster covariances.
            data_batch (Tensor): The data batch to evaluate.
            cluster_counts (Tensor): Point counts per cluster.

        Returns:
            Tensor: The computed log probabilities.
        """
        data_dim = data_batch.shape[-1]
        clusters_num = mean_matrix.size(0)
        batch_size = data_batch.shape[0]

        nus = self.nu0 - (self.data_dim_tensor - 1) + cluster_counts
        kappas = self.kappa_0 + cluster_counts

        scale_fact = torch.sqrt((kappas + 1.0) / (kappas * nus)).view(-1, 1, 1)
        chol_cov_scaled = scale_fact * chol_cov_matrices

        log_dets_sqrt = torch.sum(
            torch.log(torch.diagonal(chol_cov_scaled, dim1=-2, dim2=-1)), dim=-1
        )

        data_batch_norm = data_batch.unsqueeze(1) - mean_matrix  # [B, C, D]

        data_batch_norm_t = data_batch_norm.permute(1, 2, 0).contiguous()  # [C, D, B]
        vecs = torch.linalg.solve_triangular(
            chol_cov_scaled, data_batch_norm_t, upper=False
        )
        vecs = vecs.permute(2, 0, 1).contiguous()  # [B, C, D]

        nu_plus_D = nus + data_dim
        lgamma_num = torch.lgamma(nu_plus_D / 2.0)
        lgamma_denom = torch.lgamma(nus / 2.0)
        log_nu = torch.log(nus)

        result = torch.empty(
            (batch_size, clusters_num), device=self.device, dtype=torch.float32
        )

        BLOCK_SIZE_D = triton.next_power_of_2(data_dim)
        grid = lambda meta: (batch_size, clusters_num)  # noqa: E731

        self._fused_student_t_kernel[grid](
            vecs,
            nus,
            lgamma_num,
            lgamma_denom,
            log_nu,
            log_dets_sqrt,
            result,
            vecs.stride(0),
            vecs.stride(1),
            vecs.stride(2),
            clusters_num,
            data_dim,
            self.half_data_dim,
            self.log_pi,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )

        return result

    def _define_fused_kernel(self):
        """
        Compiles the fused Triton JIT kernel.

        This kernel computes both the squared vector norm and the final Student-t
        log PDF directly into the output tensor to minimize global memory roundtrips.
        """

        @triton.jit
        def _fused_student_t_kernel(
            vecs_ptr,  # [B, C, D]
            nus_ptr,  # [C]
            lgamma_num_ptr,  # [C]
            lgamma_denom_ptr,  # [C]
            log_nu_ptr,  # [C]
            log_dets_ptr,  # [C]
            output_ptr,  # [B, C]
            stride_vb,  # stride for batch dimension
            stride_vc,  # stride for cluster dimension
            stride_vd,  # stride for data dimension
            C,  # number of clusters
            D,  # data dimension
            half_D,  # D/2 as scalar (not constexpr)
            log_pi,  # log(pi) as scalar (not constexpr)
            BLOCK_SIZE_D: tl.constexpr,
        ):
            b = tl.program_id(0)
            c = tl.program_id(1)

            nu = tl.load(nus_ptr + c)
            lgamma_num = tl.load(lgamma_num_ptr + c)
            lgamma_denom = tl.load(lgamma_denom_ptr + c)
            log_nu = tl.load(log_nu_ptr + c)
            log_det = tl.load(log_dets_ptr + c)

            offs_d = tl.arange(0, BLOCK_SIZE_D)
            mask = offs_d < D

            vec_ptr = vecs_ptr + b * stride_vb + c * stride_vc + offs_d * stride_vd
            vec = tl.load(vec_ptr, mask=mask, other=0.0)

            vec_norm_sq = tl.sum(vec * vec, axis=0)

            nu_plus_D = nu + D
            half_nu_plus_D = nu_plus_D * 0.5

            num = lgamma_num

            denom = (
                lgamma_denom
                + half_D * (log_nu + log_pi)
                + log_det
                + half_nu_plus_D * tl.log(1.0 + vec_norm_sq / nu)
            )

            output_idx = b * C + c
            tl.store(output_ptr + output_idx, num - denom)

        self._fused_student_t_kernel = _fused_student_t_kernel

    def to(self, device: torch.device) -> FullCovarianceStudentTCalculatorTritonFused:
        self.device = device
        self.nu0 = torch.as_tensor(self.nu0, device=self.device)
        self.data_dim_tensor = torch.as_tensor(self.data_dim_tensor, device=self.device)
        self.kappa_0 = torch.as_tensor(self.kappa_0, device=self.device)
        return self
