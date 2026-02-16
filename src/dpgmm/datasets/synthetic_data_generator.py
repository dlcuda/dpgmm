from typing import List, Set

import numpy as np

from dpgmm.datasets.base_data_generator import BaseDataGenerator
from dpgmm.datasets.data_generator_types import SyntheticDataset


class SyntheticDataGenerator(BaseDataGenerator):
    """
    A specialized data generator for synthetic clustering datasets.

    This class focuses on creating data following a unique **Imbalanced Isotropic** Gaussian Mixture Model (GMM) strategy. This means the resulting clusters are:
    1.  **Imbalanced:** They have randomized, unequal proportions (weights).
    2.  **Isotropic:** They are spherical (due to identity covariance).
    """

    def generate(
        self, n_points: int = 500, data_dim: int = 2, num_components: int = 10
    ) -> SyntheticDataset:
        """
        Generates synthetic data using the Imbalanced Isotropic cluster strategy.

        Args:
            n_points (int, optional): The total number of data points. Defaults to 500.
            data_dim (int, optional): The dimensionality of the data. Defaults to 2.
            num_components (int, optional): The number of clusters (K) to generate.
                Defaults to 10.

        Returns:
            SyntheticDataset: A dictionary containing the generated 'data',
            'centers', and 'assignment'.
        """
        _num_components = num_components
        return self._generate_imbalanced_isotropic(n_points, data_dim, _num_components)

    @staticmethod
    def _generate_imbalanced_isotropic(
        n_points: int, data_dim: int, num_components: int
    ) -> SyntheticDataset:
        """
        Generates clusters with Imbalanced Weights and Isotropic Covariance.

        Process:
        1. Random, unequal cluster proportions (weights) are determined.
        2. Cluster centers (means) are drawn from a standard normal distribution.
        3. Data points are sampled, assigned to a cluster based on weights, and
           given noise corresponding to identity covariance.

        Args:
            n_points (int): Total points.
            data_dim (int): Data dimensionality.
            num_components (int): The number of clusters (K) to generate.

        Returns:
            SyntheticDataset: The constructed dataset containing shuffled data, true centers,
            and set-based assignments.
        """
        weights = np.square(np.random.uniform(size=num_components)) + 0.2
        weights /= weights.sum()
        means = np.random.randn(num_components, data_dim)
        idx = np.random.choice(np.arange(num_components), size=n_points, p=weights)
        data = np.random.randn(n_points, data_dim) + means[idx, :]
        cluster_assignment: List[Set[int]] = [set() for _ in range(num_components)]
        for i, cluster_id in enumerate(idx):
            cluster_assignment[cluster_id].add(i)

        return {
            "data": data.astype(np.float32),
            "centers": means.astype(np.float32),
            "assignment": cluster_assignment,
        }
