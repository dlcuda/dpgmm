from __future__ import annotations

from typing import Dict, List, Set

import torch
from torch import Tensor, tensor

from dpgmm.samplers.cgs.algorithm import CollapsedGibbsSampler
from dpgmm.samplers.cgs.state import PriorPosteriorParametersKeeper, StudentTCalculator
from dpgmm.samplers.cgs.variants.base.utils import init_kappa_0, init_nu_0
from dpgmm.samplers.cgs.variants.full_cov.log_likelihood import (
    FullCovarianceLogLikelihood,
)
from dpgmm.samplers.cgs.variants.full_cov.state import FullCovarianceParametersKeeper
from dpgmm.samplers.cgs.variants.full_cov.student_t_calculator.student_t_calculator_adapter import (
    StudentTCalculatorAdapter,
)


class FullCovarianceCollapsedGibbsSampler(CollapsedGibbsSampler):
    """
    Collapsed Gibbs Sampler variant for a Dirichlet Process Gaussian Mixture Model
    utilizing full covariance matrices.
    """

    def posterior_params_names(self) -> List[str]:
        """Returns the names of the tracked posterior parameters."""
        return ["mean", "cov_chol"]

    def initialize_params_for_cluster(
        self, cluster_data: Tensor, prior_params: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        Initializes the mean and Cholesky-decomposed covariance for a single cluster
        using the data assigned to it and the base prior parameters.

        Args:
            cluster_data (Tensor): The subset of data assigned to this cluster.
            prior_params (Dict[str, Tensor]): Dictionary containing base priors
                ('mean_0' and 'cov_chol_0').

        Returns:
            Dict[str, Tensor]: The initialized 'mean' and 'cov_chol' for the cluster.
        """
        mean_0, cov_chol_0 = (
            prior_params.get("mean_0"),
            prior_params.get("cov_chol_0"),
        )

        if mean_0 is None:
            raise ValueError("mean_0 is not defined in prior_params")
        if cov_chol_0 is None:
            raise ValueError("cov_chol_0 is not defined in prior_params")

        kappa_0 = init_kappa_0()
        n_cluster = cluster_data.shape[0]
        sample_data_mean = torch.mean(cluster_data, dim=0)
        cov_0 = cov_chol_0 @ cov_chol_0.T
        cluster_mean = (kappa_0 * mean_0 + n_cluster * sample_data_mean) / (
            kappa_0 + n_cluster
        )

        cluster_cov = cov_0 + cluster_data.T @ cluster_data
        cluster_cov += kappa_0 * torch.outer(mean_0, mean_0)
        cluster_cov -= (kappa_0 + n_cluster) * torch.outer(cluster_mean, cluster_mean)
        return {"mean": cluster_mean, "cov_chol": torch.cholesky(cluster_cov)}

    def compute_prior_params(
        self, data: Tensor, components_num: int
    ) -> Dict[str, Tensor]:
        """
        Computes the base prior parameters (mean and covariance) from the full dataset.

        Args:
            data (Tensor): The full dataset.
            components_num (int): Initial estimate for the number of components.

        Returns:
            Dict[str, Tensor]: Dictionary containing 'mean_0' and 'cov_chol_0'.
        """
        mean_0 = self.init_mean(data)
        cov_0 = self.init_cov(data, components_num)
        return {"mean_0": mean_0, "cov_chol_0": torch.cholesky(cov_0)}

    def get_t_student_calc(self, data_dim: int) -> StudentTCalculator:
        """Provides the Student-t probability calculator for full covariances."""
        self.t_student_calc = StudentTCalculatorAdapter.get_student_t_calculator(
            device=self.device, data_dim=data_dim
        )
        return self.t_student_calc

    def get_params_keeper(
        self, data_dim: int, init_values: Dict[str, Tensor], components_num: int
    ) -> PriorPosteriorParametersKeeper:
        """Instantiates the state keeper for full covariance parameters."""
        self.params_keeper = FullCovarianceParametersKeeper(
            data_dim, init_values, components_num, device=self.device
        )
        return self.params_keeper

    def data_log_likelihood(
        self,
        cluster_assignment: List[Set[int]],
        data: Tensor,
        cluster_params: Dict[str, List[Tensor]],
    ) -> float:
        """
        Calculates the data log-likelihood given the current cluster assignments.

        Args:
            cluster_assignment (List[Set[int]]): Indices of data points per cluster.
            data (Tensor): The full dataset.
            cluster_params (Dict[str, List[Tensor]]): Dictionary of cluster parameters.

        Returns:
            float: The computed log-likelihood score.
        """
        calc = FullCovarianceLogLikelihood(init_nu_0(data.shape[1]), self.alpha)
        result = float(
            calc.data_log_likelihood(
                cluster_assignment,
                data,
                cluster_params["mean"],
                cluster_params["cov_chol"],
            )
        )
        return result

    @staticmethod
    def init_cov(data: Tensor, components_num: int) -> Tensor:
        """Heuristically initializes a base full covariance matrix."""
        data_mean = torch.mean(data, dim=0)
        data_norm = data - data_mean.unsqueeze(0)

        data_var = (data_norm.T @ data_norm) * (1.0 / data.shape[0])
        div_factor = torch.pow(tensor(components_num), tensor(2.0 / data.shape[1]))

        return torch.diag(torch.diag(data_var / div_factor))

    @staticmethod
    def init_mean(data: Tensor) -> Tensor:
        """Initializes the base mean vector as the empirical mean of the data."""
        return torch.mean(data, dim=0)

    def to(self, device: torch.device) -> FullCovarianceCollapsedGibbsSampler:
        self.device = device
        return self
