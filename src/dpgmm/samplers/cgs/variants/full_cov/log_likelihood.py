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
            nu_k = max(self.nu_0 + len(k_examples), data_dim + 1)
            kappa_k = init_kappa_0() + len(k_examples)

            sigma_chol_k = self._sample_inverse_wishart_chol(
                df=nu_k, scale_chol=k_cov_chol
            )

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

            sigmas_chols.append(sigma_chol_k)
            means.append(mean_k)

        return means, sigmas_chols

    def _sample_inverse_wishart(self, df: float, scale: torch.Tensor) -> torch.Tensor:
        r"""
        Draws a sample X ~ InverseWishart(df, scale) via the Bartlett decomposition.

        Args:
            df (float): Degrees of freedom. Must satisfy df >= scale.shape[0].
            scale (torch.Tensor): The (D, D) positive definite scale matrix.

        Returns:
            torch.Tensor: A (D, D) positive definite inverse-Wishart sample,
                in the same dtype as the input `scale`.
        """
        chol = self._sample_inverse_wishart_chol(
            df=df, scale_chol=torch.linalg.cholesky(scale.to(torch.float64))
        )
        sigma = chol @ chol.T
        sigma = (sigma + sigma.T) / 2  # symmetrize away float rounding asymmetry
        return sigma.to(scale.dtype)

    def _sample_inverse_wishart_chol(
        self, df: float, scale_chol: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Draws X ~ InverseWishart(df, scale) and returns its lower Cholesky factor,
        given the Cholesky factor of `scale`.

        The sample is built as $X = B B^T$ with $B = L A^{-T}$ (Bartlett), but $X$ is
        never formed: its Cholesky factor is read off the QR of $B^T$ instead. Forming
        $X$ would square the condition number, and a subsequent `cholesky` call on it
        fails outright once a cluster's scale matrix is near-degenerate. Going through
        QR keeps the conditioning linear in $B$ and cannot fail on a nonsingular $B$.

        Args:
            df (float): Degrees of freedom. Must satisfy df >= scale_chol.shape[0].
            scale_chol (torch.Tensor): The lower Cholesky factor (D, D) of the
                positive definite scale matrix.

        Returns:
            torch.Tensor: The (D, D) lower triangular Cholesky factor of an
                inverse-Wishart sample, with positive diagonal, in the same dtype
                as the input `scale_chol`.
        """
        d = scale_chol.shape[0]
        orig_dtype, device = scale_chol.dtype, scale_chol.device
        dtype = torch.float64

        chol_scale = scale_chol.to(dtype)

        # Bartlett factor: chi-distributed diagonal, standard normal strictly below it.
        diag_dfs = torch.tensor([df - i for i in range(d)], dtype=dtype, device=device)
        diag_vals = torch.sqrt(torch.distributions.Chi2(diag_dfs).sample())
        bartlett = torch.randn(d, d, dtype=dtype, device=device).tril(diagonal=-1)
        bartlett = bartlett + torch.diag(diag_vals)

        eye = torch.eye(d, dtype=dtype, device=device)
        bartlett_inv_t = torch.linalg.solve_triangular(bartlett.T, eye, upper=True)

        b = chol_scale @ bartlett_inv_t

        # sigma = b @ b.T. With b.T = Q R, sigma = R.T Q.T Q R = R.T R, so R.T is a
        # lower triangular factor of sigma obtained without ever forming sigma.
        r = torch.linalg.qr(b.T, mode="reduced").R
        # QR leaves the sign of each row of R free; flip so the diagonal is positive,
        # which makes R.T the Cholesky factor proper. R.T R is unchanged by the flip.
        signs = torch.sign(torch.diagonal(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        r = r * signs.unsqueeze(1)

        return r.T.to(orig_dtype)
