import os.path as op
from abc import ABC, abstractmethod

import numpy as np
import torch.distributions as dist
from loguru import logger

from dpgmm.utils import file_system as fs_utils


class TraceReader(ABC):
    """
    Abstract base class for reading and processing sampling traces from DPGMM runs.

    Handles loading the trace data, validating initial dimensionality, and
    extracting cluster counts which are common across various covariance types.
    """

    def __init__(self, trace):
        """
        Initializes the TraceReader, loads trace data, and computes base statistics.

        Args:
            trace (str | dict): Path to the pickled trace file, or the loaded trace dictionary.

        Raises:
            ValueError: If the file is not found, if no clusters are present in the trace,
                or if the data dimensionality is invalid (<=0).
        """
        if isinstance(trace, str):
            if not op.exists(trace):
                raise ValueError(f"Trace file not found: {trace}")
            self.trace_obj = fs_utils.read_pickle(trace)
            logger.info(f"Read trace from path: {trace}")
        else:
            self.trace_obj = trace

        logger.info(f"Keys from trace are: {list(self.trace_obj.keys())}")
        self.cluster_params = self.trace_obj["cluster_params"]

        self.clusters_num = len(self.cluster_params["mean"])
        if self.clusters_num == 0:
            raise ValueError("No clusters found in the trace")
        logger.info(f"Clusters number in the trace: {self.clusters_num}")

        self.data_dim = int(self.cluster_params["mean"][0].shape[0])
        if self.data_dim <= 0:
            raise ValueError("Data dimensionality must be greater than 0")

        self.cluster_counts = {
            i: len(self.trace_obj["cluster_assignment"][i])
            for i in range(self.clusters_num)
        }

        logger.info(f"Data dimensionality: {self.data_dim}")

    def get_cluster_unnormalized_weights(self):
        """
        Calculates the unnormalized weights for the mixture components.

        The weight calculation uses the concentration parameter (alpha) from the
        Dirichlet Process trace and the computed cluster assignment counts.

        Returns:
            np.ndarray: A 1D array of unnormalized weights for each cluster.
        """
        trace_alpha = float(self.trace_obj["alpha"])
        cluster_counts = [
            x[1] for x in sorted(self.cluster_counts.items(), key=lambda x: x[0])
        ]
        weights = trace_alpha + np.array(cluster_counts)
        return weights

    @abstractmethod
    def init_nu_0(self, data_dim: int) -> float:
        """
        Initializes the prior degrees of freedom.

        Args:
            data_dim (int): Dimensionality of the data.

        Returns:
            float: The prior degrees of freedom value.
        """
        pass

    @abstractmethod
    def init_kappa_0(self) -> float:
        """
        Initializes the prior scale factor for the mean.

        Returns:
            float: The prior scale factor value.
        """
        pass

    @abstractmethod
    def prepare_t_student_mixture(self) -> dist.Distribution:
        """
        Constructs a mixture distribution of Student-t components.

        Must be implemented by subclasses to handle specific covariance structures.

        Returns:
            dist.Distribution: A PyTorch mixture distribution.
        """
        pass
