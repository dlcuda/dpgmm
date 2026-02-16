import os

import torch

import dpgmm.utils.file_system as fs_utils
from dpgmm.metrics.cgs_trace.full_cov import FullCovTraceReader
from dpgmm.metrics.complexity.complexity_estimator import ComplexityEstimator


class ComplexityFromTraceEstimator(ComplexityEstimator):
    """
    A factory-like class that initializes a ComplexityEstimator directly from
    saved DPGMM trace and data files.
    """

    def __init__(
        self,
        trace_path: str,
        data_trace_path: str,
        samples_num: int = 2000,
    ):
        """
        Initializes the complexity estimator by parsing trace outputs.

        Args:
            trace_path (str): File path to the pickled model trace.
            data_trace_path (str): File path to the pickled data trace.
            samples_num (int, optional): Number of samples for entropy estimation.
                Defaults to 2000.

        Raises:
            ValueError: If either the trace_path or data_trace_path does not exist.
        """
        if not os.path.exists(data_trace_path):
            raise ValueError(f"Data trace file not found: {data_trace_path}")
        trace_reader = FullCovTraceReader(trace_path)

        if not os.path.exists(trace_path):
            raise ValueError(f"Trace file not found: {trace_path}")
        data_trace = fs_utils.read_pickle(data_trace_path)
        data = torch.as_tensor(data_trace["data"], dtype=torch.float32)

        mixture_distr = trace_reader.prepare_t_student_mixture()

        super().__init__(
            mixture_distr=mixture_distr,
            data=data,
            clusters_num=trace_reader.clusters_num,
            samples_num=samples_num,
        )
