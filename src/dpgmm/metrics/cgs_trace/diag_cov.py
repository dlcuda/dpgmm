import torch
from pydantic import field_validator
from torch import Tensor
from torch.distributions import Categorical, Independent, MixtureSameFamily, StudentT

from dpgmm.metrics.cgs_trace.base import TraceReader
from dpgmm.metrics.cgs_trace.cgs_trace_types import BaseModelWithTensorsAllowed
from dpgmm.samplers.cgs.variants.base.utils import init_kappa_0, init_nu_0


class DiagCovTStudentParams(BaseModelWithTensorsAllowed):
    """
    Pydantic model representing parameters for a diagonal-covariance Student-t distribution.

    Attributes:
        dof (Tensor): Degrees of freedom for the Student-t distribution.
        mean (Tensor): Mean vector of the cluster.
        var (Tensor): Scaled diagonal variance vector.
    """

    dof: Tensor
    mean: Tensor
    var: Tensor

    @field_validator("dof", "mean", "var")
    def check_tensor(cls, v):
        """Validates that the provided attributes are PyTorch Tensors."""
        if not isinstance(v, torch.Tensor):
            raise TypeError("Value must be a torch.Tensor")
        return v


class DiagCovTraceReader(TraceReader):
    """
    Trace reader for extracting and processing parameters from a diagonal covariance DPGMM trace.
    """

    def __init__(self, trace):
        """
        Initializes the DiagCovTraceReader and extracts cluster parameters.

        Args:
            trace (str | dict): Path to the pickled trace file or the loaded trace dictionary.
        """
        super().__init__(trace)
        self.mean = self.cluster_params["mean"]
        self.var = self.cluster_params["var"]

    def prepare_diag_cov_t_student_params(self, cluster_index) -> DiagCovTStudentParams:
        """
        Computes the updated posterior Student-t parameters for a specific cluster.

        Uses the cluster counts and base priors to calculate the updated degrees of freedom
        and scales the diagonal variance accordingly.

        Args:
            cluster_index (int): Index of the cluster to process.

        Returns:
            DiagCovTStudentParams: A validated object containing the computed degrees
                of freedom, mean, and scaled variance matrix.
        """
        mean = torch.as_tensor(self.mean[cluster_index], dtype=torch.float32)
        var = torch.as_tensor(self.var[cluster_index], dtype=torch.float32)
        cluster_count = torch.tensor(
            self.cluster_counts[cluster_index], dtype=torch.float32
        )

        kappa_0, nu_0 = self.init_kappa_0(), self.init_nu_0(self.data_dim)
        kappa_n, nu_n = kappa_0 + cluster_count, nu_0 + cluster_count

        dof = nu_n
        scale_var_factor = (kappa_n + 1) / kappa_n
        scaled_var = scale_var_factor * var

        return DiagCovTStudentParams(dof=dof, mean=mean, var=scaled_var)

    def prepare_t_student_mixture(self) -> MixtureSameFamily:
        """
        Constructs a PyTorch mixture distribution consisting of diagonal-covariance
        Student-t components using independent 1D Student-t distributions.

        Returns:
            MixtureSameFamily: The diagonal covariance Student-t mixture distribution.
        """
        cluster_weights_unnorm = torch.as_tensor(
            self.get_cluster_unnormalized_weights(), dtype=torch.float32
        )
        cluster_weights_norm = cluster_weights_unnorm / torch.sum(
            cluster_weights_unnorm
        )

        cat_distr = Categorical(probs=cluster_weights_norm)

        t_student_params = [
            self.prepare_diag_cov_t_student_params(i) for i in range(self.clusters_num)
        ]

        dofs = torch.stack([p.dof for p in t_student_params]).unsqueeze(1)
        means = torch.stack([p.mean for p in t_student_params])
        vars_ = torch.stack([p.var for p in t_student_params])

        component_dist = Independent(StudentT(df=dofs, loc=means, scale=vars_), 1)

        mixture = MixtureSameFamily(cat_distr, component_dist)
        return mixture

    def init_nu_0(self, data_dim: int) -> float:
        """Retrieves the prior degrees of freedom."""
        return init_nu_0(data_dim)

    def init_kappa_0(self) -> float:
        """Retrieves the prior mean scale."""
        return init_kappa_0()


if __name__ == "__main__":
    trace = "/Users/mateuszlampert/agh/gmm_sampler/out/diag_cov/results/cgs_49.pkl"
    trace_reader = DiagCovTraceReader(trace)
    params = trace_reader.prepare_diag_cov_t_student_params(0)
    print(params)

    print(trace_reader.cluster_params.keys())
