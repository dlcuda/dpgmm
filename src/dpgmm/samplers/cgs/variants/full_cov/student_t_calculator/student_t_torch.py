from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import Tensor

from dpgmm.samplers.cgs.state import StudentTCalculator
from dpgmm.samplers.cgs.variants.base.utils import init_kappa_0, init_nu_0


class FullCovarianceStudentTCalculatorTorch(StudentTCalculator):
    """
    Calculates the Student-t log probability density for full covariance matrices
    using purely vectorized PyTorch operations.
    """

    def __init__(self, device: torch.device, data_dim: int):
        """
        Initializes the PyTorch-based Student-t calculator.

        Args:
            device (torch.device, optional): The computing device. Defaults to the
                system's available device.
        """
        self.device = device
        self.nu0 = torch.tensor(
            init_nu_0(data_dim), dtype=torch.float32, device=self.device
        )
        self.data_dim_tensor = torch.tensor(data_dim, device=self.device)

    def log_pdf(
        self, data_batch: Tensor, parameters: List[Tensor], cluster_counts: Tensor
    ) -> Tensor:
        """
        Interface method to compute the Student-t log PDF.

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
        result = self.t_student_log_pdf_torch(
            mean_matrix, chol_cov_matrices, data_batch, cluster_counts
        )
        return result

    def t_student_log_pdf_torch(
        self,
        mean_matrix: Tensor,
        chol_cov_matrices: Tensor,
        data_batch: Tensor,
        cluster_counts: Tensor,
    ) -> Tensor:
        """
        Executes the batched operations for the Student-t log PDF entirely within PyTorch.

        Args:
            mean_matrix (Tensor): Cluster means.
            chol_cov_matrices (Tensor): Cholesky factors of cluster covariances.
            data_batch (Tensor): The data batch to evaluate.
            cluster_counts (Tensor): Point counts per cluster.

        Returns:
            Tensor: The computed log probabilities.
        """
        # Cast data_dim to float32 for consistency
        data_dim = torch.tensor(
            data_batch.shape[-1], dtype=torch.float32, device=self.device
        )
        clusters_num = mean_matrix.size(0)

        # Calculate degrees of freedom (nus)
        nus = self.nu0 - (self.data_dim_tensor - 1) + cluster_counts

        # Calculate kappas
        kappas = torch.tensor(init_kappa_0(), dtype=torch.float32) + cluster_counts

        # Scale factor calculation with proper broadcasting
        scale_fact = ((kappas + 1.0) / (kappas * nus)).view(-1, 1, 1).to(self.device)
        chol_cov_scaled = torch.sqrt(scale_fact) * chol_cov_matrices

        # Calculate log determinants
        chol_cov_diagonals = torch.diagonal(chol_cov_scaled, dim1=-2, dim2=-1)
        log_dets_sqrt = torch.sum(torch.log(chol_cov_diagonals), dim=-1)

        # Data normalization
        data_batch_tiled = data_batch.unsqueeze(1).expand(-1, clusters_num, -1)
        data_batch_norm = data_batch_tiled - mean_matrix
        data_batch_norm_transposed = data_batch_norm.permute(1, 2, 0)

        # Solve triangular system
        vecs = torch.linalg.solve_triangular(
            chol_cov_scaled, data_batch_norm_transposed, upper=False
        )

        # Calculate vector norms
        vecs_norm = torch.norm(vecs, dim=1)
        vecs_norm = vecs_norm.t()

        # Expand nus for broadcasting
        nus = nus.unsqueeze(0)

        # Calculate numerator and denominator
        num = torch.lgamma((nus + data_dim) / 2.0).to(self.device)
        denom = (
            torch.lgamma(nus / 2.0)
            + (data_dim / 2.0) * (torch.log(nus) + torch.tensor(np.pi).log())
            + log_dets_sqrt
            + ((nus + data_dim) / 2.0) * torch.log1p((vecs_norm / torch.sqrt(nus)) ** 2)
        )

        return num - denom

    def to(self, device: torch.device) -> FullCovarianceStudentTCalculatorTorch:
        self.device = device
        self.nu0 = torch.as_tensor(self.nu0, device=self.device)
        self.data_dim_tensor = torch.as_tensor(self.data_dim_tensor, device=self.device)
        return self
