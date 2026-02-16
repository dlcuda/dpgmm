import random
from typing import List, Literal, Set

import numpy as np

from dpgmm.datasets.base_data_generator import BaseDataGenerator
from dpgmm.datasets.data_generator_types import SyntheticDataset

CovarianceType = Literal["full", "diag"]


class GaussianDataGenerator(BaseDataGenerator):
    """
    Generates synthetic datasets based on a Gaussian Mixture Model (GMM).

    This generator creates data distributed across multiple Gaussian clusters.
    Each cluster has a randomly selected mean within the range [-5, 5] and
    a randomly generated covariance matrix (either full or diagonal).

    The resulting data points are shuffled to ensure they are not ordered by cluster.
    """

    def __init__(self, cov_type: CovarianceType = "full"):
        """
        Initializes the Gaussian data generator.

        Args:
            cov_type (CovarianceType, optional): Controls the shape of the clusters.
                - 'full': Generates arbitrary rotated ellipsoids (random positive definite matrices).
                - 'diag': Generates axis-aligned ellipsoids (diagonal matrices).
                Defaults to "full".
        """
        self.cov_type: CovarianceType = cov_type

    def generate(
        self, n_points: int = 500, data_dim: int = 2, num_components: int = 10
    ) -> SyntheticDataset:
        """
        Generates a synthetic Gaussian Mixture Model (GMM) dataset.

        It partitions the total points roughly equally among components, generates
        random parameters for each component based on `self.cov_type`, samples the
        data, and then shuffles the results to randomize the indices.

        Args:
            n_points (int, optional): Total number of data points to generate. Defaults to 500.
            data_dim (int, optional): Dimensionality of the data (number of features). Defaults to 2.
            num_components (int, optional): The number of clusters (mixture components). Defaults to 10.

        Returns:
            SyntheticDataset: A dictionary containing:
                - "data": The shuffled data matrix (N, D).
                - "centers": The true means of the clusters (K, D).
                - "assignment": A list of sets, where the i-th set contains
                  the indices of data points belonging to cluster i.
        """
        examples_per_component = [
            int(n_points / num_components) for _ in range(num_components - 1)
        ]
        examples_per_component.append(n_points - sum(examples_per_component))

        samples: List[np.ndarray] = []
        means: List[np.ndarray] = []
        assignment: List[int] = []

        for i, n_ex in enumerate(examples_per_component):
            mean = np.random.uniform(-5, 5, size=data_dim)

            if self.cov_type == "full":
                A = np.random.randn(data_dim, data_dim)
                cov = np.dot(A, A.T)
            else:
                cov = np.diag(np.random.uniform(0.5, 2.0, size=data_dim))

            comp_samples = np.random.multivariate_normal(mean, cov, size=n_ex)
            samples.extend(comp_samples)
            means.append(mean)
            assignment.extend([i] * n_ex)

        combined = list(zip(samples, assignment))
        random.shuffle(combined)
        samples_shuffled, assignment_shuffled = zip(*combined)

        cluster_assignment: List[Set[int]] = [set() for _ in range(num_components)]
        for idx, cluster_id in enumerate(assignment_shuffled):
            cluster_assignment[cluster_id].add(idx)

        return {
            "data": np.array(samples_shuffled, dtype=np.float32),
            "centers": np.array(means, dtype=np.float32),
            "assignment": cluster_assignment,
        }
