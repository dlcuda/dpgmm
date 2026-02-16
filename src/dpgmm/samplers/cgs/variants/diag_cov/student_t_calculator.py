from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import Tensor

from dpgmm.samplers.cgs.state import StudentTCalculator
from dpgmm.samplers.cgs.variants.base.utils import init_kappa_0, init_nu_0


class DiagCovarianceStudentTCalculator(StudentTCalculator):
    """
    Vectorized calculator for evaluating Student-t probability density functions
    under an assumption of diagonal covariance (independent dimensions).
    """

    def __init__(self, device: torch.device, data_dim: int):
        """
        Initializes the calculator by loading the prior hyperparameters.

        Args:
            device (torch.device): The device on which tensors should be allocated.
        """
        self.device = device

        self.nu0 = torch.tensor(
            init_nu_0(data_dim), dtype=torch.float32, device=self.device
        )
        self.kappa0 = torch.tensor(
            init_kappa_0(), dtype=torch.float32, device=self.device
        )

    def log_pdf(
        self, data_batch: Tensor, parameters: List[Tensor], cluster_counts: Tensor
    ) -> Tensor:
        """
        Interface method mapping standard parameter lists to the internal tensor logic.

        Args:
            data_batch (Tensor): A batch of data points to evaluate.
            parameters (List[Tensor]): A list containing the mean and variance tensors.
            cluster_counts (Tensor): Tensor of data point counts per cluster.

        Returns:
            Tensor: Log probabilities for the data batch.
        """
        if data_batch.device != self.device:
            raise ValueError(
                f"Data batch device {data_batch.device} does not match calculator device {self.device}"
            )
        mean_matrix, vars_matrix = parameters
        result = self.t_student_log_pdf_torch(
            mean_matrix, vars_matrix, data_batch, cluster_counts
        )
        return result

    def t_student_log_pdf_torch(
        self,
        mean_matrix: Tensor,
        vars_matrix: Tensor,
        data_batch: Tensor,
        cluster_counts: Tensor,
    ) -> Tensor:
        """
        Computes the log predictive probability of data points belonging to clusters
        using highly vectorized operations.

        Since the covariance is diagonal, the multivariate PDF factors into a product
        of 1D PDFs (or a sum in log-space).

        Args:
            mean_matrix (Tensor): Cluster means of shape (clusters_num, data_dim).
            vars_matrix (Tensor): Cluster variances of shape (clusters_num, data_dim).
            data_batch (Tensor): Data points of shape (batch_size, data_dim).
            cluster_counts (Tensor): Number of points in each cluster of shape (clusters_num,).

        Returns:
            Tensor: Log probabilities of shape (batch_size, clusters_num).
        """
        clusters_num = mean_matrix.size(0)

        # Degrees of freedom
        nus = (self.nu0 + cluster_counts).unsqueeze(1)

        # Kappas
        kappas = (self.kappa0 + cluster_counts).unsqueeze(1)

        # Scale factor for diagonal variances
        # scale_fact.shape -> (clusters_num, 1)
        scale_fact = (1.0 + kappas) / kappas
        # stds_matrix.shape -> (clusters_num, data_dim)
        stds_matrix = torch.sqrt(scale_fact) * torch.sqrt(vars_matrix)

        # Normalize data
        # data_batch_norm.shape -> (batch_size, clusters_num, data_dim)
        data_batch_tiled = data_batch.unsqueeze(1).expand(-1, clusters_num, -1)
        data_batch_norm = (data_batch_tiled - mean_matrix) / stds_matrix

        log_data_batch_norm_squared = torch.log1p(
            (data_batch_norm / torch.sqrt(nus)) ** 2
        )

        # Numerator: lgamma((ν+1)/2) - ((ν+1)/2) * log(1 + (x/√ν)^2)
        # num.shape -> (batch_size, clusters_num, data_dim)
        num = (
            torch.lgamma((nus + 1.0) / 2.0)
            - (nus + 1.0) / 2.0 * log_data_batch_norm_squared
        )

        # Denominator: lgamma(ν/2) + 0.5*log(ν) + 0.5*log(pi) + log(stds)
        # denom.shape -> (1, clusters_num, data_dim)
        denom = (
            torch.lgamma(nus / 2.0)
            + 0.5 * torch.log(nus)
            + 0.5
            * torch.log(torch.tensor(np.pi, dtype=torch.float32, device=self.device))
            + torch.log(stds_matrix)
        ).unsqueeze(0)

        # result.shape -> (batch_size, clusters_num)
        data_batch_pds = num - denom
        return torch.sum(data_batch_pds, dim=-1)

    def to(self, device: torch.device) -> DiagCovarianceStudentTCalculator:
        self.device = device
        self.nu0 = torch.as_tensor(self.nu0, device=self.device)
        self.kappa0 = torch.as_tensor(self.kappa0, device=self.device)
        return self
