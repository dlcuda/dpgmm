from typing import Union

import numpy as np
import torch
import torch.distributions as dist
from loguru import logger
from torch import Tensor, tensor


def init_cov(data: Tensor, components_num: int) -> Tensor:
    """
    Initializes a heuristic diagonal covariance matrix for a baseline invariant distribution.

    Args:
        data (Tensor): The input data tensor of shape (N, D).
        components_num (int): The number of mixture components.

    Returns:
        Tensor: A diagonal covariance matrix scaled based on the number of components
            and data dimensionality.
    """
    data_mean = torch.mean(data, dim=0)
    data_norm = data - data_mean.unsqueeze(0)

    data_var = (data_norm.T @ data_norm) / data.shape[0]
    div_factor = torch.pow(
        tensor(components_num, dtype=torch.float32),
        tensor(2.0 / data.shape[1], dtype=torch.float32),
    )

    return torch.diag(torch.diag(data_var / div_factor))


class ComplexityEstimator:
    """
    Estimates the complexity of a mixture distribution by comparing it against a
    baseline diagonal Multivariate Normal (MVN) distribution.
    """

    def __init__(
        self,
        mixture_distr: dist.Distribution,
        data: Tensor,
        clusters_num: int,
        samples_num: int,
    ):
        """
        Initializes the ComplexityEstimator.

        Args:
            mixture_distr (dist.Distribution): The prepared mixture distribution
                (e.g., MixtureSameFamily) to evaluate.
            data (Tensor): The actual data tensor of shape (N, D) used to compute
                invariant distribution statistics.
            clusters_num (int): Number of clusters, needed for heuristic covariance
                initialization.
            samples_num (int): Number of samples to generate for entropy estimation.
        """
        self.samples_num = samples_num
        self.device = data.device
        self.data = data
        self.clusters_num = clusters_num
        self.t_student_mixture = mixture_distr
        self.mvn_diag_distr = self.__get_invariant_distr()

    def estimate_entropy_with_sampling(self) -> float:
        """
        Estimates the relative entropy using points sampled directly from the mixture.

        Returns:
            float: The mean log-probability difference between the mixture and the
                baseline invariant distribution.
        """
        sampled_points = self.sample_points_from_mixture()
        log_probs_val = self.estimate_entropy(sampled_points).item()
        logger.info("Computed mean of log probs for samples, val = %.3f", log_probs_val)
        return log_probs_val

    def estimate_entropy_on_data(self, data: Union[np.ndarray, Tensor]) -> float:
        """
        Estimates the relative entropy using a specifically provided dataset.

        Args:
            data (Union[np.ndarray, Tensor]): The data points to evaluate.

        Returns:
            float: The computed mean of log-probability differences on the provided data.
        """
        target_data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        log_probs_val = self.estimate_entropy(target_data).item()
        logger.info(
            "Computed mean of log probs on provided data, val = %.3f", log_probs_val
        )
        return log_probs_val

    def sample_points_from_mixture(self) -> Tensor:
        """
        Generates random samples from the target Student-t mixture distribution.

        Returns:
            Tensor: A tensor of sampled points of shape (samples_num, D).
        """
        samples_generated = self.t_student_mixture.sample((self.samples_num,))
        samples_generated = samples_generated.to(self.device)

        logger.info(
            "Finished sampling points, returning array of shape: %s",
            str(samples_generated.shape),
        )
        return samples_generated

    def estimate_entropy(self, samples_t: Tensor) -> Tensor:
        """
        Computes the expected difference in log probabilities between the mixture
        and the baseline invariant distribution.

        Args:
            samples_t (Tensor): The points at which to evaluate the log probabilities.

        Returns:
            Tensor: A scalar tensor representing the mean log probability difference.
        """
        samples_t = samples_t.to(self.device)
        log_probs_mixture = self.t_student_mixture.log_prob(samples_t).view(-1)
        log_probs_mvn = self.mvn_diag_distr.log_prob(samples_t).view(-1)

        return torch.mean(log_probs_mixture - log_probs_mvn)

    def __get_invariant_distr(self) -> dist.Independent:
        """
        Constructs the baseline diagonal Multivariate Normal distribution.

        Returns:
            dist.Independent: An independent normal distribution matching the data's
                mean and heuristically scaled variance.
        """
        logger.info(
            "Creating diagonal MVN distribution from data shape: %s",
            str(tuple(self.data.shape)),
        )

        data_mean = torch.mean(self.data, dim=0)
        scale_diagonal = torch.sqrt(
            torch.diag(init_cov(self.data, components_num=self.clusters_num))
        )

        mvn_diag_distr = dist.Independent(
            dist.Normal(loc=data_mean, scale=scale_diagonal), 1
        )
        return mvn_diag_distr
