import torch
import torch.distributions as dist
from pydantic import field_validator
from torch import Tensor
from torch.distributions import Categorical, MixtureSameFamily

from dpgmm.metrics.cgs_trace.base import TraceReader
from dpgmm.metrics.cgs_trace.cgs_trace_types import BaseModelWithTensorsAllowed
from dpgmm.samplers.cgs.variants.base.utils import init_kappa_0, init_nu_0
from dpgmm.utils.distributions.multivariate_student_t import MultivariateStudentT


class FullCovTStudentParams(BaseModelWithTensorsAllowed):
    """
    Pydantic model representing parameters for a full-covariance Multivariate Student-t distribution.

    Attributes:
        dof (Tensor): Degrees of freedom for the Student-t distribution.
        mean (Tensor): Mean vector of the cluster.
        cov_chol (Tensor): Cholesky decomposition of the scaled covariance matrix.
    """

    dof: Tensor
    mean: Tensor
    cov_chol: Tensor

    @field_validator("dof", "mean", "cov_chol")
    def check_tensor(cls, v):
        """Validates that the provided attributes are PyTorch Tensors."""
        if not isinstance(v, torch.Tensor):
            raise TypeError("Value must be a torch.Tensor")
        return v


class FullCovTraceReader(TraceReader):
    """
    Trace reader for extracting and processing parameters from a full covariance DPGMM trace.
    """

    def __init__(self, trace):
        """
        Initializes the FullCovTraceReader and extracts cluster parameters.

        Args:
            trace (str | dict): Path to the pickled trace file or the loaded trace dictionary.
        """
        super().__init__(trace)

        self.mean = self.cluster_params["mean"]
        self.cov_chol = self.cluster_params["cov_chol"]

    def prepare_full_cov_t_student_params(self, cluster_index) -> FullCovTStudentParams:
        """
        Computes the updated posterior Student-t parameters for a specific cluster.

        Uses the cluster counts and base priors to calculate the updated degrees of freedom
        and scales the Cholesky covariance factor accordingly.

        Args:
            cluster_index (int): Index of the cluster to process.

        Returns:
            FullCovTStudentParams: A validated object containing the computed degrees
                of freedom, mean, and scaled Cholesky covariance matrix.
        """
        mean = torch.as_tensor(self.mean[cluster_index], dtype=torch.float32)
        cov_chol = torch.as_tensor(self.cov_chol[cluster_index], dtype=torch.float32)
        cluster_count = torch.tensor(
            self.cluster_counts[cluster_index], dtype=torch.float32
        )

        kappa_0, nu_0 = self.init_kappa_0(), self.init_nu_0(self.data_dim)
        kappa_n, nu_n = kappa_0 + cluster_count, nu_0 + cluster_count
        dof = nu_n - self.data_dim + 1

        scale_cov_factor = torch.sqrt((kappa_n + 1) / (kappa_n * dof))
        scaled_cov_chol = scale_cov_factor * cov_chol

        return FullCovTStudentParams(dof=dof, mean=mean, cov_chol=scaled_cov_chol)

    def prepare_t_student_mixture(self) -> dist.Distribution:
        """
        Constructs a PyTorch mixture distribution consisting of full-covariance
        Multivariate Student-t components.

        Returns:
            dist.Distribution: A `MixtureSameFamily` distribution representing the
                overall full-covariance mixture.
        """
        cluster_weights_unnorm = torch.as_tensor(
            self.get_cluster_unnormalized_weights(), dtype=torch.float32
        )
        cluster_weights_norm = cluster_weights_unnorm / torch.sum(
            cluster_weights_unnorm
        )

        cat_distr = Categorical(probs=cluster_weights_norm)

        t_student_params = [
            self.prepare_full_cov_t_student_params(ind)
            for ind in range(self.clusters_num)
        ]

        dofs = torch.stack([p.dof for p in t_student_params])
        means = torch.stack([p.mean for p in t_student_params])
        cov_chols = torch.stack([p.cov_chol for p in t_student_params])

        component_dist = MultivariateStudentT(df=dofs, loc=means, scale_tril=cov_chols)

        mixture = MixtureSameFamily(cat_distr, component_dist)
        return mixture

    def init_nu_0(self, data_dim: int) -> float:
        """Retrieves the prior degrees of freedom."""
        return init_nu_0(data_dim)

    def init_kappa_0(self) -> float:
        """Retrieves the prior mean scale."""
        return init_kappa_0()


if __name__ == "__main__":
    trace = "/Users/mateuszlampert/agh/gmm_sampler/out/d_2/n_128/c_8/e_20/results/cgs_19.pkl"
    trace_reader = FullCovTraceReader(trace)
    params = trace_reader.prepare_full_cov_t_student_params(0)
    print(params)

    print(trace_reader.cluster_params.keys())
