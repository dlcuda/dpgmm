from typing import Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from dpgmm.datasets.data_generator_types import SyntheticDataset


class DataVisualizer:
    """
    A visualization utility for synthetic datasets used in GMM experiments.

    This class provides static methods to inspect generated data by plotting
    2D data points colored by their ground-truth cluster assignments,
    along with the actual cluster centroids.
    """

    @staticmethod
    def plot(
        gen_data: SyntheticDataset,
        title: str = "Generated data",
        out_file: Optional[str] = None,
    ) -> None:
        """
        Generates a 2D scatter plot of the synthetic dataset.

        It extracts data points, cluster centers, and assignments from the
        input dictionary and plots them using the Seaborn 'Set2' palette.

        Args:
            gen_data (SyntheticDataset): A dictionary-like object containing:
                - "data" (np.ndarray): The dataset of shape (N, 2).
                - "centers" (np.ndarray): The cluster centers of shape (K, 2).
                - "assignment" (List[Set[int]]): A list of length K, where each set
                  contains the indices of points belonging to that cluster.
            title (str, optional): The title of the plot. Defaults to "Generated data".
            out_file (Optional[str], optional): The file path to save the figure.
                If None, the plot is displayed interactively using plt.show().

        Raises:
            AssertionError: If the input data is not 2-dimensional (data.shape[1] != 2).
        """
        data = gen_data["data"]
        centers = gen_data["centers"]
        assignment = gen_data["assignment"]

        if data.shape[1] != 2:
            raise ValueError("DataVisualizer only supports 2-dimensional data.")

        _, ax = plt.subplots(figsize=(6, 6))

        colors = sns.color_palette("Set2", n_colors=len(assignment))

        for i, indices in enumerate(assignment):
            pts = data[list(indices), :]
            ax.scatter(
                pts[:, 0], pts[:, 1], color=colors[i], s=10, label=f"Cluster {i}"
            )
            ax.scatter(
                centers[i, 0],
                centers[i, 1],
                color=colors[i],
                marker="x",
                s=60,
                linewidths=2,
            )

        ax.set_title(title)
        ax.legend(loc="best")
        ax.set_xlabel("x_1")
        ax.set_ylabel("x_2")

        if out_file:
            plt.savefig(out_file, bbox_inches="tight")
        else:
            plt.show()


if __name__ == "__main__":
    n_points = 300
    n_clusters = 3

    centers = np.array([[-3, -3], [3, 3], [3, -3]])
    data: list[np.ndarray] = []
    assignment: list[set[int]] = []

    for i in range(n_clusters):
        cluster_pts = np.random.randn(n_points // n_clusters, 2) + centers[i]
        start_idx = len(data)

        data.extend(cluster_pts)
        assignment.append(set(range(start_idx, start_idx + len(cluster_pts))))

    merge_data = np.array(data)

    mock_gen_data = SyntheticDataset(
        data=merge_data, centers=centers, assignment=assignment
    )

    DataVisualizer.plot(mock_gen_data, title="Test Visualization")
