import os

import torch

from dpgmm.metrics.cgs_trace import FullCovTraceReader
from dpgmm.metrics.entanglement.entanglement_estimator import EntanglementEstimator


class EntanglementFromTraceEstimator(EntanglementEstimator):
    """
    A factory-like class that initializes an EntanglementEstimator directly from
    a saved DPGMM trace file.
    """

    def __init__(
        self,
        trace_path: str,
        samples_num: int = 2000,
    ):
        """
        Initializes the entanglement estimator by extracting components from a trace.

        Args:
            trace_path (str): File path to the pickled model trace.
            samples_num (int, optional): Number of samples for Monte Carlo DKL
                estimation. Defaults to 2000.

        Raises:
            ValueError: If the trace_path does not exist.
        """
        if not os.path.exists(trace_path):
            raise ValueError(f"Trace file not found: {trace_path}")
        trace_reader = FullCovTraceReader(trace_path)

        joint_distr = trace_reader.prepare_t_student_mixture()
        all_params = [
            trace_reader.prepare_full_cov_t_student_params(k)
            for k in range(trace_reader.clusters_num)
        ]

        means = torch.stack([p.mean for p in all_params])
        cov_chols = torch.stack([p.cov_chol for p in all_params])
        dofs = torch.stack([p.dof for p in all_params])

        weights_np = trace_reader.get_cluster_unnormalized_weights()
        weights = torch.as_tensor(weights_np, dtype=torch.float32)

        super().__init__(
            joint_distr=joint_distr,
            weights=weights,
            means=means,
            cov_chols=cov_chols,
            dofs=dofs,
            samples_num=samples_num,
        )
