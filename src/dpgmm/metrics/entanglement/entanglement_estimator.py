from typing import List

import torch
import torch.distributions as dist
from torch.distributions import Categorical, MixtureSameFamily

from dpgmm.utils.distributions.multivariate_student_t import MultivariateStudentT


class EntanglementEstimator:
    """
    Estimates the degree of feature entanglement within a distribution by comparing
    the joint distribution to the product of its marginal distributions.
    """

    def __init__(
        self,
        joint_distr: dist.Distribution,
        weights: torch.Tensor,
        means: torch.Tensor,
        cov_chols: torch.Tensor,
        dofs: torch.Tensor,
        samples_num: int = 2000,
    ):
        """
        Initializes the EntanglementEstimator.

        Args:
            joint_distr (dist.Distribution): The full joint mixture distribution.
            weights (torch.Tensor): Unnormalized or normalized component weights.
            means (torch.Tensor): Cluster means.
            cov_chols (torch.Tensor): Cholesky factors of the cluster covariances.
            dofs (torch.Tensor): Degrees of freedom for the Student-t components.
            samples_num (int, optional): Number of samples used for Monte Carlo
                KL divergence estimation. Defaults to 2000.
        """
        self.samples_num = samples_num
        self.joint_distr = joint_distr
        self.weights = weights
        self.means = means
        self.cov_chols = cov_chols
        self.dofs = dofs

        if self.means.dim() > 1:
            self.clusters_num, self.data_dim = self.means.shape
        else:
            self.clusters_num = 1
            self.data_dim = self.means.shape[0]

        self.marginals_prod = self._prepare_t_student_marginals_prod()

    def calculate_symmetric_dkl(self) -> float:
        """
        Calculates the symmetric Kullback-Leibler divergence between the joint
        distribution and the product of its marginals.

        Returns:
            float: The symmetric DKL value (DKL(P||Q) + DKL(Q||P)).
        """
        joint_samples = self.joint_distr.sample(sample_shape=(self.samples_num,))
        marginal_prod_samples = [
            m.sample(sample_shape=(self.samples_num,)) for m in self.marginals_prod
        ]
        marginal_prod_samples_tensor = torch.cat(marginal_prod_samples, dim=-1)

        pq_ratio = self.pq_ratio_for_samples(joint_samples)
        qp_ratio = 1.0 / self.pq_ratio_for_samples(marginal_prod_samples_tensor)

        return torch.mean(pq_ratio + qp_ratio).item()

    def calculate_joint_and_prod_dkl(self) -> float:
        """
        Calculates the standard (asymmetric) Kullback-Leibler divergence DKL(P||Q).

        Returns:
            float: The estimated DKL from the joint to the product of marginals.
        """
        joint_samples = self.joint_distr.sample(sample_shape=(self.samples_num,))
        return torch.mean(self.pq_ratio_for_samples(joint_samples)).item()

    def pq_ratio_for_samples(self, samples_tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes the log-probability difference (log P(X) - log Q(X)) for given samples.

        Args:
            samples_tensor (torch.Tensor): A tensor of samples to evaluate.

        Returns:
            torch.Tensor: A 1D tensor of log-probability differences for each sample.
        """
        p_log_probs = self.joint_distr.log_prob(samples_tensor).view(-1)

        prod_log_probs = []
        for i, i_var_distr in enumerate(self.marginals_prod):
            val = i_var_distr.log_prob(samples_tensor[:, i : i + 1]).view(-1)
            prod_log_probs.append(val)

        q_log_probs = torch.stack(prod_log_probs, dim=-1).sum(dim=-1)

        return p_log_probs - q_log_probs

    def _prepare_t_student_marginals_prod(self) -> List[MixtureSameFamily]:
        """
        Constructs the independent 1D marginal distributions from the joint parameters.

        Returns:
            List[MixtureSameFamily]: A list containing the 1D Student-t mixture
                distributions for each dimension.
        """
        cluster_weights_norm = self.weights / torch.sum(self.weights)
        cat_distr = Categorical(probs=cluster_weights_norm)
        dims_mixture_distributions = []

        for dim_index in range(self.data_dim):
            dim_means = self.means[:, dim_index].view(self.clusters_num, 1)
            dim_scales = self.cov_chols[:, dim_index, dim_index].view(
                self.clusters_num, 1, 1
            )

            dim_t_student_distr = MultivariateStudentT(
                df=self.dofs, loc=dim_means, scale_tril=dim_scales
            )
            dim_mixture = MixtureSameFamily(
                mixture_distribution=cat_distr,
                component_distribution=dim_t_student_distr,
            )
            dims_mixture_distributions.append(dim_mixture)

        return dims_mixture_distributions
