from typing import List, Optional, Set, Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from dpgmm.samplers.cgs.utils.prob import cov_error_ellipse


class ClusterParamsVisualizer:
    """
    A visualization utility for Gaussian Mixture Model (GMM) clustering results.

    This class handles the plotting of data points, cluster centers, and
    confidence ellipses for both individual clusters and the global dataset.
    It supports both full covariance and diagonal covariance assumptions.
    """

    @classmethod
    def plot_params_full_covariance(
        cls,
        data: np.ndarray,
        centers: np.ndarray,
        assignment: List[Set[int]],
        cov_chol: List[np.ndarray],
        trace_alpha: float,
        title: str = "Cluster params (full covariance)",
        out_file: Optional[str] = None,
        ellipse_points: int = 500,
    ) -> None:
        """
        Plots clusters with full covariance matrices.

        This method resolves the scale matrices (Cholesky factors) into covariance
        matrices before plotting.

        Args:
            data (np.ndarray): The dataset of shape (N, D).
            centers (np.ndarray): Cluster centers of shape (K, D).
            assignment (List[Set[int]]): A list where each set contains indices of points
                                          assigned to that cluster.
            cov_chol (List[np.ndarray]): List of K Cholesky decompositions of the
                                         cluster scatter matrices.
            trace_alpha (float): The Dirichlet concentration parameter used for weighting.
            title (str, optional): Title of the plot. Defaults to "Cluster params (full covariance)".
            out_file (Optional[str], optional): Path to save the figure. If None, shows plot.
            ellipse_points (int, optional): Number of points to generate for drawing the ellipse.
        """
        covariances = cls._resolve_full_covariances(cov_chol, assignment)
        cls._plot(
            data,
            centers,
            assignment,
            covariances,
            trace_alpha,
            title,
            out_file,
            ellipse_points,
        )

    @classmethod
    def plot_params_diag_covariance(
        cls,
        data: np.ndarray,
        centers: np.ndarray,
        assignment: List[Set[int]],
        variances: List[np.ndarray],
        trace_alpha: float,
        title: str = "Cluster params (diagonal covariance)",
        out_file: Optional[str] = None,
        ellipse_points: int = 500,
    ) -> None:
        """
        Plots clusters with diagonal covariance matrices.

        This method resolves the variance vectors into diagonal covariance matrices
        before plotting.

        Args:
            data (np.ndarray): The dataset of shape (N, D).
            centers (np.ndarray): Cluster centers of shape (K, D).
            assignment (List[Set[int]]): A list where each set contains indices of points
                                          assigned to that cluster.
            variances (List[np.ndarray]): List of K variance vectors (one per cluster).
            trace_alpha (float): The Dirichlet concentration parameter used for weighting.
            title (str, optional): Title of the plot. Defaults to "Cluster params (diagonal covariance)".
            out_file (Optional[str], optional): Path to save the figure. If None, shows plot.
            ellipse_points (int, optional): Number of points to generate for drawing the ellipse.
        """
        covariances = cls._resolve_diag_covariances(variances, assignment)
        cls._plot(
            data,
            centers,
            assignment,
            covariances,
            trace_alpha,
            title,
            out_file,
            ellipse_points,
        )

    @classmethod
    def _plot(
        cls,
        data: np.ndarray,
        centers: np.ndarray,
        assignment: List[Set[int]],
        covariances: List[np.ndarray],
        trace_alpha: float,
        title: str,
        out_file: Optional[str],
        ellipse_points: int,
    ) -> None:
        """
        Internal shared plotting logic.

        Orchestrates the drawing of scatter points, cluster centroids, cluster-specific
        confidence ellipses, and the global covariance ellipse.

        Args:
            data (np.ndarray): The raw data points.
            centers (np.ndarray): The cluster centroids.
            assignment (List[Set[int]]): Point indices per cluster.
            covariances (List[np.ndarray]): Resolved covariance matrices for each cluster.
            trace_alpha (float): Alpha value for global stats calculation.
            title (str): Plot title.
            out_file (Optional[str]): Save path or None.
            ellipse_points (int): Resolution of the ellipses.
        """
        _, ax = plt.subplots(figsize=(6, 6))
        n_clusters = len(assignment)
        colors = sns.color_palette("Paired", n_colors=n_clusters)

        for i, indices in enumerate(assignment):
            cluster_data = data[list(indices), :]
            cluster_center = centers[i]
            ax.scatter(
                cluster_data[:, 0],
                cluster_data[:, 1],
                color=colors[i],
                edgecolors="none",
                s=10,
                label=f"Cluster {i}",
            )
            ax.scatter(
                cluster_center[0], cluster_center[1], color=colors[i], marker="x", s=30
            )

            if covariances is not None:
                xs, ys = cov_error_ellipse(
                    cluster_center, covariances[i], p=0.95, samples_num=ellipse_points
                )
                ax.scatter(xs, ys, color=colors[i], s=5, edgecolors="none")

        global_mean, global_cov = cls._calculate_global_stats(
            assignment, centers, covariances, trace_alpha
        )
        if global_mean is not None and global_cov is not None:
            ax.scatter(global_mean[0], global_mean[1], color="k", marker="x", s=30)
            xs, ys = cov_error_ellipse(
                global_mean, global_cov, p=0.95, samples_num=ellipse_points * 2
            )
            ax.scatter(
                xs, ys, color="k", s=10, edgecolors="none", label="Global Covariance"
            )

        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend(loc="best")

        if out_file:
            plt.savefig(out_file, bbox_inches="tight")
        else:
            plt.show()

    @classmethod
    def _calculate_global_stats(
        cls,
        assignment: List[Set[int]],
        centers: np.ndarray,
        covariances: List[np.ndarray],
        trace_alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the weighted global mean and global covariance.

        Implements the Law of Total Variance:
        Total Cov = E[Cov(X|Cluster)] + Cov(E[X|Cluster])
                  = Weighted sum of cluster covariances + Weighted covariance of cluster centers.

        Args:
            assignment (List[Set[int]]): Point indices per cluster.
            centers (np.ndarray): Cluster centers (K, D).
            covariances (List[np.ndarray]): Covariance matrices (K, D, D).
            trace_alpha (float): Dirichlet parameter for weight smoothing.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - global_mean (D,)
                - global_cov (D, D)
        """
        n_clusters = len(assignment)
        cluster_weights = cls._get_cluster_unnormalized_weights(
            n_clusters, assignment, trace_alpha
        )
        cluster_weights /= np.sum(cluster_weights)

        means_arr = np.array(centers, dtype=np.float32)
        data_dim = means_arr.shape[1]
        global_mean = np.dot(cluster_weights, means_arr)
        global_cov = np.zeros((data_dim, data_dim), dtype=np.float32)

        for i, w in enumerate(cluster_weights):
            global_cov += w * covariances[i]
            global_cov += w * np.outer(means_arr[i], means_arr[i])
        global_cov -= np.outer(global_mean, global_mean)

        return global_mean, global_cov

    @classmethod
    def _resolve_full_covariances(
        cls, cov_chol: List[np.ndarray], assignment: List[Set[int]]
    ) -> List[np.ndarray]:
        """
        Converts Cholesky factors of scatter matrices into normalized covariance matrices.

        Uses the Inverse-Wishart degrees of freedom (nu) normalization logic.

        Args:
            cov_chol (List[np.ndarray]): Cholesky factors of cluster scatter matrices.
            assignment (List[Set[int]]): Point assignments.

        Returns:
            List[np.ndarray]: The list of full covariance matrices.
        """
        from dpgmm.samplers.cgs.variants.full_cov.algorithm import init_nu_0

        covariances_list = []
        for i, chol in enumerate(cov_chol):
            nu = init_nu_0(chol.shape[0]) + len(assignment[i])
            cov = np.dot(chol, chol.T) / (nu - chol.shape[0] - 1.0)
            covariances_list.append(cov)
        return covariances_list

    @classmethod
    def _resolve_diag_covariances(
        cls, variances: List[np.ndarray], assignment: List[Set[int]]
    ) -> List[np.ndarray]:
        """
        Converts raw variance vectors into diagonal covariance matrices.

        Uses the Inverse-Gamma degrees of freedom (nu) normalization logic.

        Args:
            variances (List[np.ndarray]): Variance vectors for each cluster.
            assignment (List[Set[int]]): Point assignments.

        Returns:
            List[np.ndarray]: The list of diagonal covariance matrices.
        """
        from dpgmm.samplers.cgs.variants.diag_cov.algorithm import init_nu_0

        covariances_list = []
        for i, var in enumerate(variances):
            nu = init_nu_0(var.shape[0]) + len(assignment[i])
            cov = np.diag(var) * (nu / (nu - 2.0))
            covariances_list.append(cov)
        return covariances_list

    @classmethod
    def _get_cluster_unnormalized_weights(
        cls, n_clusters: int, assignment: List[Set[int]], trace_alpha: float
    ) -> np.ndarray:
        """
        Calculates cluster weights based on point counts and a Dirichlet prior (alpha).

        Args:
            n_clusters (int): Number of clusters.
            assignment (List[Set[int]]): Point assignments.
            trace_alpha (float): The concentration parameter.

        Returns:
            np.ndarray: Unnormalized weights (counts + alpha).
        """
        counts = np.array([len(assignment[i]) for i in range(n_clusters)])
        return trace_alpha + counts


if __name__ == "__main__":
    n_points = 500
    data_dim = 2
    n_clusters = 3
    trace_alpha = 1.0

    centers = np.random.uniform(-5, 5, size=(n_clusters, data_dim))

    data_points: list[np.ndarray] = []
    assignment: List[Set[int]] = [set() for _ in range(n_clusters)]
    cov_chol: list[np.ndarray] = []
    variances: list[np.ndarray] = []

    for i in range(n_clusters):
        n_points_cluster = n_points // n_clusters
        A = np.random.randn(data_dim, data_dim)
        cov_i = np.dot(A, A.T)
        scatter_matrix = cov_i * n_points_cluster
        chol_i = np.linalg.cholesky(scatter_matrix)
        cov_chol.append(chol_i)

        var_i = np.random.uniform(0.5, 2.0, size=data_dim)
        variances.append(var_i)

        samples = np.random.multivariate_normal(
            centers[i], cov_i, size=n_points_cluster
        )
        start_idx = len(data_points)
        data_points.extend(samples)
        assignment[i] = set(range(start_idx, start_idx + n_points_cluster))

    data = np.array(data_points, dtype=np.float32)

    ClusterParamsVisualizer.plot_params_full_covariance(
        data=data,
        centers=centers,
        assignment=assignment,
        cov_chol=cov_chol,
        trace_alpha=trace_alpha,
        title="Full Covariance Clusters",
    )

    ClusterParamsVisualizer.plot_params_diag_covariance(
        data=data,
        centers=centers,
        assignment=assignment,
        variances=variances,
        trace_alpha=trace_alpha,
        title="Diagonal Covariance Clusters",
    )
