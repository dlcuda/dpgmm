from __future__ import annotations

from typing import List, Set

import torch

from dpgmm.samplers.cgs.variants.diag_cov.student_t_calculator import (
    DiagCovarianceStudentTCalculator,
)


class DiagonalCovarianceLogLikelihoodCalculator:
    """
    Computes the log-likelihood of a dataset partitioned into a diagonal covariance
    Gaussian Mixture Model.
    """

    def __init__(self, device: torch.device, data_dim: int):
        """
        Initializes the calculator by setting up the underlying Student-t calculator.
        """
        self.device = device
        self.calc = DiagCovarianceStudentTCalculator(device=device, data_dim=data_dim)

    def data_log_likelihood(
        self,
        cluster_assignment: List[Set[int]],
        data: torch.Tensor,
        means: List[torch.Tensor],
        variances: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Evaluates the log-likelihood of each data point belonging to its assigned cluster
        using the diagonal Student-t predictive posterior distribution.

        Args:
            cluster_assignment (List[Set[int]]): A list where each index represents a cluster,
                and the value is a set of data point indices assigned to it.
            data (torch.Tensor): The full input dataset tensor.
            means (List[torch.Tensor]): A list containing the mean vector for each cluster.
            variances (List[torch.Tensor]): A list containing the diagonal variance vector
                for each cluster.

        Returns:
            torch.Tensor: A scalar tensor containing the summed log-likelihood across
                all data points.
        """
        if data.device != self.device:
            raise ValueError(
                f"Data device {data.device} does not match calculator device {self.calc.device}"
            )
        examples_lls = []

        for k, k_examples in enumerate(cluster_assignment):
            if not k_examples:
                continue

            k_cluster_data_points = data[list(k_examples), :].to(self.device)

            k_means = torch.as_tensor(means[k].unsqueeze(0), device=self.device)
            k_vars = torch.as_tensor(variances[k].unsqueeze(0), device=self.device)
            k_counts = torch.tensor(
                [len(k_examples)], dtype=torch.float32, device=self.device
            )

            k_lls = self.calc.t_student_log_pdf_torch(
                k_means, k_vars, k_cluster_data_points, k_counts
            )

            examples_lls.append(k_lls)

        return torch.sum(torch.cat(examples_lls))

    def to(self, device: torch.device) -> DiagonalCovarianceLogLikelihoodCalculator:
        self.calc = self.calc.to(device)
        return self
