from __future__ import annotations

from typing import Dict, List, Set

import torch
from torch import Tensor

from dpgmm.samplers.cgs.algorithm import CollapsedGibbsSampler
from dpgmm.samplers.cgs.state import PriorPosteriorParametersKeeper, StudentTCalculator
from dpgmm.samplers.cgs.variants.diag_cov.log_likelihood import (
    DiagonalCovarianceLogLikelihoodCalculator,
)
from dpgmm.samplers.cgs.variants.diag_cov.state import DiagCovarianceParametersKeeper
from dpgmm.samplers.cgs.variants.diag_cov.student_t_calculator import (
    DiagCovarianceStudentTCalculator,
)


def init_nu_0(data_dim: int) -> float:
    """
    Initializes the prior degrees of freedom based on data dimensionality.

    Args:
        data_dim (int): Dimensionality of the data.

    Returns:
        float: The prior degrees of freedom.
    """
    return 2.0 + data_dim


def init_kappa_0() -> float:
    """
    Initializes the prior mean scale factor.

    Returns:
        float: The prior mean scale.
    """
    return 0.01


class DiagCovarianceCollapsedGibbsSampler(CollapsedGibbsSampler):
    """
    Collapsed Gibbs Sampler variant for a Dirichlet Process Gaussian Mixture Model
    enforcing diagonal covariance matrices.
    """

    def posterior_params_names(self) -> List[str]:
        """
        Retrieves the names of the posterior parameters tracked by this sampler.

        Returns:
            List[str]: A list containing the strings 'mean' and 'var'.
        """
        return ["mean", "var"]

    def initialize_params_for_cluster(
        self, cluster_data: Tensor, prior_params: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        Initializes the mean and diagonal variance for a single cluster using its
        assigned data and the base prior parameters.

        Args:
            cluster_data (Tensor): The subset of data points assigned to the cluster.
            prior_params (Dict[str, Tensor]): Dictionary containing base priors
                ('mean_0' and 'var_0').

        Returns:
            Dict[str, Tensor]: A dictionary with the initialized 'mean' and 'var' tensors.

        Raises:
            ValueError: If 'mean_0' or 'var_0' are missing from the prior_params.
        """
        mean_0, var_0 = prior_params.get("mean_0"), prior_params.get("var_0")

        if mean_0 is None:
            raise ValueError("mean_0 is not defined in prior_params")
        if var_0 is None:
            raise ValueError("var_0 is not defined in prior_params")

        cluster_n = cluster_data.shape[0]
        data_dim = cluster_data.shape[1]

        kappa_0, nu_0 = init_kappa_0(), init_nu_0(data_dim)
        kappa_n, nu_n = kappa_0 + cluster_n, nu_0 + cluster_n

        mean_n = (mean_0 * kappa_0 + torch.sum(cluster_data, dim=0)) / kappa_n

        sum_of_data_squares = torch.sum(cluster_data * cluster_data, dim=0)
        nu_n_times_var_n = (
            nu_0 * var_0
            + kappa_0 * mean_0 * mean_0
            - kappa_n * mean_n * mean_n
            + sum_of_data_squares
        )
        cluster_var = nu_n_times_var_n / nu_n

        return {"mean": mean_n, "var": cluster_var}

    def compute_prior_params(
        self, data: Tensor, components_num: int
    ) -> Dict[str, Tensor]:
        """
        Computes the base prior mean and diagonal variance from the complete dataset.

        Args:
            data (Tensor): The full dataset.
            components_num (int): Expected initial number of clusters.

        Returns:
            Dict[str, Tensor]: A dictionary containing the 'mean_0' and 'var_0' priors.
        """
        mean_0 = self.init_mean(data)
        var_0 = self.init_var(data, components_num)
        return {"mean_0": mean_0, "var_0": var_0}

    def get_t_student_calc(self, data_dim: int) -> StudentTCalculator:
        """
        Provides the calculator instance for evaluating diagonal Student-t probabilities.

        Returns:
            StudentTCalculator: An instance of DiagCovarianceStudentTCalculator.
        """
        self.t_student_calc = DiagCovarianceStudentTCalculator(
            device=self.device, data_dim=data_dim
        )
        return self.t_student_calc

    def get_params_keeper(
        self, data_dim: int, init_values: Dict[str, Tensor], components_num: int
    ) -> PriorPosteriorParametersKeeper:
        """
        Instantiates the state keeper responsible for managing diagonal parameters.

        Args:
            data_dim (int): Dimensionality of the input data.
            init_values (Dict[str, Tensor]): Initial parameter values.
            components_num (int): Initial number of mixture components.

        Returns:
            PriorPosteriorParametersKeeper: An instance of DiagCovarianceParametersKeeper.
        """
        self.params_keeper = DiagCovarianceParametersKeeper(
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
        Calculates the total data log-likelihood given current cluster assignments.

        Args:
            cluster_assignment (List[Set[int]]): Indices of data points per cluster.
            data (Tensor): The full dataset.
            cluster_params (Dict[str, List[Tensor]]): Dictionary of cluster parameters
                containing 'mean' and 'var'.

        Returns:
            float: The computed log-likelihood score.
        """
        calc = DiagonalCovarianceLogLikelihoodCalculator(
            device=self.device, data_dim=data.shape[1]
        )
        result = float(
            calc.data_log_likelihood(
                cluster_assignment,
                data,
                cluster_params["mean"],
                cluster_params["var"],
            )
        )
        return result

    @staticmethod
    def init_var(data: Tensor, components_num: int) -> Tensor:
        """
        Heuristically initializes a base diagonal variance vector from the data.

        Args:
            data (Tensor): The input dataset.
            components_num (int): Expected initial number of mixture components.

        Returns:
            Tensor: A 1D tensor representing the initialized diagonal variance.
        """
        data_mean = torch.mean(data, dim=0)
        data_norm = data - data_mean.unsqueeze(0)

        data_var = (data_norm.T @ data_norm) * (1.0 / data.shape[0])
        data_dim = data.shape[1]

        div_factor_log = (2.0 / data_dim) * torch.log(
            torch.tensor(float(components_num))
        )
        cov_0_log = torch.log(torch.diag(data_var)) - div_factor_log
        cov_0 = torch.exp(cov_0_log)

        return 2.0 * (cov_0 / init_nu_0(data_dim))

    @staticmethod
    def init_mean(data: Tensor) -> Tensor:
        """
        Initializes the base mean vector as the empirical mean of the dataset.

        Args:
            data (Tensor): The input dataset.

        Returns:
            Tensor: A 1D tensor representing the dataset mean.
        """
        return torch.mean(data, dim=0)

    def to(self, device: torch.device) -> DiagCovarianceCollapsedGibbsSampler:
        self.device = device
        return self
