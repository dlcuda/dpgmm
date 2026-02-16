from typing import List, Set

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from dpgmm.samplers.cgs.utils import prob as prob_utils


class FullCovarianceLogLikelihood:
    """
    Computes the log-likelihood of a dataset under a full covariance Gaussian Mixture Model.
    """

    def __init__(self, nu_0: float, alpha_0: float):
        """
        Initializes the log-likelihood calculator with base prior hyperparameters.

        Args:
            nu_0 (float): The prior degrees of freedom.
            alpha_0 (float): The concentration parameter for the Dirichlet Process.
        """
        self.nu_0 = nu_0
        self.alpha_0 = alpha_0

    @staticmethod
    def normal_log_likelihood(
        data: torch.Tensor, mean: torch.Tensor, cov_chol: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluates the log probability density for data points under a Multivariate Normal distribution.

        Args:
            data (torch.Tensor): The input data points of shape (N, D).
            mean (torch.Tensor): The mean vector of the distribution of shape (D,).
            cov_chol (torch.Tensor): The lower Cholesky factor of the covariance matrix of shape (D, D).

        Returns:
            torch.Tensor: A 1D tensor of log probabilities for each data point.
        """
        mvn = MultivariateNormal(loc=mean, scale_tril=cov_chol)
        return mvn.log_prob(data)

    def data_log_likelihood(
        self,
        cluster_assignment: List[Set[int]],
        data: torch.Tensor,
        means: List[torch.Tensor],
        cov_chols: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Calculates the total log-likelihood of the dataset given current cluster assignments.

        Args:
            cluster_assignment (List[Set[int]]): A list where each element is a set of data indices
                belonging to that specific cluster.
            data (torch.Tensor): The full dataset tensor.
            means (List[torch.Tensor]): List of mean vectors for each cluster.
            cov_chols (List[torch.Tensor]): List of Cholesky covariance factors for each cluster.

        Returns:
            torch.Tensor: A scalar tensor representing the total log-likelihood of the assignment.
        """
        examples_assignment = [0 for _ in range(data.shape[0])]
        for cluster, cluster_examples in enumerate(cluster_assignment):
            for ex in cluster_examples:
                examples_assignment[ex] = cluster

        data_dim = data.shape[1]
        sampled_means, sampled_cov_chols = self._sample_marginals_for_mean_and_sigma(
            cluster_assignment, means, cov_chols, data_dim
        )
        data_log_pdfs = []
        for cov_chol, mean in zip(sampled_cov_chols, sampled_means):
            k_log_pdfs = self.normal_log_likelihood(data, mean, cov_chol)
            data_log_pdfs.append(k_log_pdfs)

        data_log_pdfs_ndarray = torch.stack(data_log_pdfs)
        assignment_ll = torch.sum(
            data_log_pdfs_ndarray[examples_assignment, torch.arange(data.shape[0])]
        )
        return assignment_ll

    def _sample_marginals_for_mean_and_sigma(
        self,
        cluster_assignment: List[Set[int]],
        post_means: List[torch.Tensor],
        post_cov_chols: List[torch.Tensor],
        data_dim: int,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Samples mean vectors and covariance Cholesky factors from the posterior Normal-Wishart distribution.

        Args:
            cluster_assignment (List[Set[int]]): Current mapping of data indices to clusters.
            post_means (List[torch.Tensor]): List of posterior mean tensors.
            post_cov_chols (List[torch.Tensor]): List of posterior Cholesky covariance tensors.
            data_dim (int): The dimensionality of the data.

        Returns:
            tuple: A tuple containing:
                - List[torch.Tensor]: Sampled mean vectors.
                - List[torch.Tensor]: Sampled Cholesky covariance matrices.
        """
        from dpgmm.samplers.cgs.variants.full_cov.algorithm import init_kappa_0

        self.device = post_means[0].device

        sigmas_chols, means = [], []
        for k, k_examples in enumerate(cluster_assignment):
            k_mean, k_cov_chol = post_means[k], post_cov_chols[k]
            nu_k = self.nu_0 + len(k_examples)
            kappa_k = init_kappa_0() + len(k_examples)

            s_k = k_cov_chol @ k_cov_chol.T
            s_k_inverse = self._ensure_strict_pd(s_k.inverse(), eps_min=1e-2)

            sigma_k = (
                torch.distributions.Wishart(
                    df=max(nu_k, data_dim + 1), covariance_matrix=s_k_inverse
                )
                .sample()
                .inverse()
            )

            sigma_k = self._ensure_strict_pd(sigma_k, eps_min=1e-3)

            mean_k = prob_utils.multivariate_t_rvs(
                k_mean,
                torch.sqrt(
                    torch.tensor(
                        1.0 / (kappa_k * max(nu_k - data_dim + 1, 1)),
                        device=self.device,
                    )
                )
                * k_cov_chol,
                df=max(nu_k - data_dim + 1, 1),
            )

            sigmas_chols.append(self._robust_cholesky(sigma_k, eps_start=1e-3))
            means.append(mean_k)

        return means, sigmas_chols

    def _ensure_strict_pd(
        self, tensor: torch.Tensor, eps_min: float = 1e-2
    ) -> torch.Tensor:
        """
        Projects a matrix to be strictly positive definite by clipping its eigenvalues.

        Args:
            tensor (torch.Tensor): The input square matrix.
            eps_min (float, optional): The minimum allowed eigenvalue. Defaults to 1e-2.

        Returns:
            torch.Tensor: A strictly positive definite matrix.

        Raises:
            ValueError: If the input tensor is not a square matrix.
        """
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            raise ValueError("Input must be a square matrix")
        tensor_sym = (tensor + tensor.T) / 2
        eigvals, eigvecs = torch.linalg.eigh(tensor_sym)
        eigvals_clipped = torch.clamp(eigvals, min=eps_min)
        tensor_pd = eigvecs @ torch.diag(eigvals_clipped) @ eigvecs.T
        return tensor_pd

    def _robust_cholesky(
        self, matrix: torch.Tensor, eps_start: float = 1e-6, max_tries: int = 5
    ) -> torch.Tensor:
        """
        Attempts a Cholesky decomposition, iteratively adding jitter if numerical instability occurs.

        Args:
            matrix (torch.Tensor): The matrix to decompose.
            eps_start (float, optional): Initial jitter value to add if decomposition fails. Defaults to 1e-6.
            max_tries (int, optional): Maximum number of jitter addition attempts. Defaults to 5.

        Returns:
            torch.Tensor: The lower Cholesky factor.

        Raises:
            RuntimeError: If the decomposition fails after the maximum number of attempts.
        """
        eps = eps_start
        mat = matrix.clone()
        for _ in range(max_tries):
            try:
                return torch.linalg.cholesky(mat)
            except RuntimeError:
                mat = self._ensure_strict_pd(mat, eps_min=eps)
                eps *= 10
        raise RuntimeError("Cholesky decomposition failed even after adding jitter")
