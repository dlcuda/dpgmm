from __future__ import annotations

import os.path as op
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Set, Tuple

import numpy as np
import torch
from loguru import logger
from scipy import stats
from tqdm import tqdm

from dpgmm.samplers.base import BaseSampler, BaseSamplerFitResult
from dpgmm.samplers.cgs.state import (
    PriorPosteriorParametersKeeper,
    StudentTCalculator,
    TorchCgsSharedComputationsManager,
)
from dpgmm.utils import file_system as fs_utils

InitStrategy = Literal["init_data_stats"]


class CollapsedGibbsSampler(ABC, BaseSampler):
    """
    Abstract base class for implementing Collapsed Gibbs Sampling for
    Dirichlet Process Gaussian Mixture Models (DPGMM).
    """

    def __init__(
        self,
        init_strategy: InitStrategy,
        max_clusters_num: int,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        """
        Initializes the Collapsed Gibbs Sampler.

        Args:
            init_strategy (InitStrategy): Strategy to initialize cluster statistics.
            max_clusters_num (int): The maximum initial number of clusters to consider.
            batch_size (int, optional): Number of data points to process in a batch.
                Defaults to 1.
            **kwargs: Additional hyperparameters, including:
                - a (float): Alpha prior shape parameter. Defaults to 1.0.
                - b (float): Alpha prior rate parameter. Defaults to 1.0.
                - skip_epochs_logging (int): Epoch interval for saving models. Defaults to 1.
                - skip_epochs_ll_calc (int): Epoch interval for calculating log-likelihood.
                    Defaults to 1.
                - restore_snapshot_pkl_path (str): Path to a saved model snapshot to resume
                    training. Defaults to None.
        """
        self.init_strategy = init_strategy
        self.max_clusters_num = max_clusters_num
        self.batch_size = batch_size
        self.device = torch.device("cpu")

        self.a = float(kwargs.get("a", 1.0))
        self.b = float(kwargs.get("b", 1.0))

        self.alpha = self.sample_alpha()

        self.skip_epochs_logging = int(kwargs.get("skip_epochs_logging", 1))
        self.skip_epochs_ll_calc = int(kwargs.get("skip_epochs_ll_calc", 1))
        self.restore_snapshot_pkl_path = (
            kwargs.get("restore_snapshot_pkl_path")
            if "restore_snapshot_pkl_path" in kwargs
            else None
        )

        logger.info(f"Initialized model: {self.print_model()}")

    @abstractmethod
    def posterior_params_names(self) -> list[str]:
        """
        Retrieves the names of the posterior parameters being tracked.

        Returns:
            list[str]: A list of parameter name strings.
        """
        pass

    @abstractmethod
    def compute_prior_params(
        self, data: torch.Tensor, components_num: int
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the prior parameters for the model based on the data.

        Args:
            data (torch.Tensor): The input dataset.
            components_num (int): Number of mixture components.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping parameter names to their prior tensors.
        """
        pass

    @abstractmethod
    def initialize_params_for_cluster(
        self, cluster_data: torch.Tensor, prior_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Initializes parameter tensors for a specific cluster based on assigned data points.

        Args:
            cluster_data (torch.Tensor): The data points currently assigned to the cluster.
            prior_params (Dict[str, torch.Tensor]): The computed prior parameters.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping parameter names to their
                initialized cluster-specific tensors.
        """
        pass

    @abstractmethod
    def get_t_student_calc(self, data_dim: int) -> StudentTCalculator:
        """
        Provides the calculator instance responsible for Student-t probability computations.

        Returns:
            StudentTCalculator: An initialized calculator instance.
        """
        pass

    @abstractmethod
    def get_params_keeper(
        self, data_dim: int, init_values: Dict[str, torch.Tensor], components_num: int
    ) -> PriorPosteriorParametersKeeper:
        """
        Provides the state keeper instance managing prior and posterior parameter tensors.

        Args:
            data_dim (int): Dimensionality of the data.
            init_values (Dict[str, torch.Tensor]): Initial parameter values.
            components_num (int): Number of mixture components.

        Returns:
            PriorPosteriorParametersKeeper: An initialized parameter keeper.
        """
        pass

    @abstractmethod
    def data_log_likelihood(
        self,
        cluster_assignment: List[Set[int]],
        data: torch.Tensor,
        cluster_params: Dict[str, List[torch.Tensor]],
    ) -> float:
        """
        Calculates the log-likelihood of the data given the current cluster assignments
        and parameters.

        Args:
            cluster_assignment (List[Set[int]]): Sets of data indices assigned to each cluster.
            data (torch.Tensor): The input dataset.
            cluster_params (Dict[str, List[torch.Tensor]]): Current parameter values per cluster.

        Returns:
            float: The calculated log-likelihood.
        """
        pass

    @staticmethod
    def get_examples_assignment(cluster_assignment: List[Set[int]]) -> List[int]:
        """
        Converts a list of cluster sets into a flat list of cluster indices for each data point.

        Args:
            cluster_assignment (List[Set[int]]): A list where each element is a set of
                data point indices assigned to that cluster index.

        Returns:
            List[int]: A list where the value at index `i` is the cluster assignment for data point `i`.
        """
        from itertools import chain

        n_points = len(list(chain.from_iterable(cluster_assignment)))
        examples_assignment = [0] * n_points

        for cluster_idx, examples in enumerate(cluster_assignment):
            for ex in examples:
                examples_assignment[ex] = cluster_idx

        return examples_assignment

    def get_initial_assignment(
        self, data: np.ndarray
    ) -> Tuple[List[Set[int]], List[int]]:
        """
        Generates a random initial assignment of data points to clusters.

        Args:
            data (np.ndarray): The input dataset.

        Returns:
            Tuple[List[Set[int]], List[int]]: A tuple containing the clustered sets of indices,
                and the flat mapping of data points to cluster indices.
        """
        clusters_num = min(data.shape[0], self.max_clusters_num)
        init = np.random.randint(0, clusters_num, size=(data.shape[0],))
        clusters_examples: Dict[np.long, Set[int]] = {}
        for example_idx, example_cluster in enumerate(init):
            if example_cluster in clusters_examples:
                clusters_examples[example_cluster].add(example_idx)
            else:
                clusters_examples[example_cluster] = {example_idx}

        cluster_assignment = list(clusters_examples.values())
        examples_assignment = self.get_examples_assignment(cluster_assignment)

        return cluster_assignment, examples_assignment

    def create_computations_manager(
        self,
        data: torch.Tensor,
        clusters_params: dict[str, List[torch.Tensor]],
        batch_size: int,
    ) -> TorchCgsSharedComputationsManager:
        """
        Initializes the manager responsible for batched torch computations during sampling.

        Args:
            data (torch.Tensor): The input dataset.
            clusters_params (dict[str, List[torch.Tensor]]): Current parameter values.
            batch_size (int): The number of data points processed per iteration.

        Returns:
            TorchCgsSharedComputationsManager: An instance managing the vectorized
                sampling logic.
        """
        components_num = len(list(clusters_params.values())[0])
        clusters_params_stacked = {
            k: torch.stack(v).to(dtype=torch.float32) if isinstance(v, list) else v
            for k, v in clusters_params.items()
        }
        init_values = {
            **self.compute_prior_params(data, components_num),
            **clusters_params_stacked,
        }
        params_keeper = self.get_params_keeper(
            data.shape[1], init_values, components_num
        )
        calculator = self.get_t_student_calc(data_dim=data.shape[1])
        return TorchCgsSharedComputationsManager(
            data, components_num, params_keeper, calculator, batch_size
        )

    def fit(
        self,
        iterations_num: int,
        data: torch.Tensor,
        out_dir: str | None = None,
    ) -> BaseSamplerFitResult:
        """
        Executes the main Collapsed Gibbs Sampling loop to fit the model to the data.

        Args:
            iterations_num (int): The total number of sampling iterations to perform.
            data (torch.Tensor): The input dataset as a torch tensor.
            out_dir (str | None, optional): Directory to save model snapshots. Defaults to None.

        Returns:
            BaseSamplerFitResult: A dictionary containing the final `cluster_params`, `cluster_assignment`,
                and `alpha` concentration parameter.
        """
        if data.device != self.device:
            raise ValueError(
                f"Data is on device {data.device}, but sampler is set to {self.device}. Please move the sampler to the correct device using .to(device) before calling fit."
            )

        (
            init_iter,
            ass_ll,
            curr_alpha,
            cluster_assignment,
            examples_assignment,
            cluster_params,
        ) = self._initialize_fit_state(data, data.detach().cpu().numpy())
        n_points = data.shape[0]
        computations_manager = self.create_computations_manager(
            data, cluster_params, self.batch_size
        )

        logger.info("Instantiated computations manager")
        ex_permutation = list(range(data.shape[0]))
        res: dict[str, list[torch.Tensor]] = {}

        for iter_num in range(init_iter, iterations_num):
            if iter_num % self.skip_epochs_ll_calc == 0:
                ass_ll = self.data_log_likelihood(
                    cluster_assignment, data, cluster_params
                )
                logger.info(
                    "Started %d epoch, current clusters number: %d, assignment ll: %.2f, "
                    "curr-alpha: %.2f "
                    % (iter_num, len(cluster_assignment), ass_ll, curr_alpha)
                )

            progress_bar = tqdm(range(0, n_points, self.batch_size))
            for start_batch_index in progress_bar:
                curr_alpha = self.update_alpha(
                    curr_alpha, n_points, len(cluster_assignment)
                )
                data_point_indices = ex_permutation[
                    start_batch_index : start_batch_index + self.batch_size
                ]
                cluster_counts_before_removal = [
                    len(cluster_examples) for cluster_examples in cluster_assignment
                ]
                data_batch_clusters, clusters_batch_removed = (
                    self.remove_assignment_for_batch(
                        data_point_indices, cluster_assignment, examples_assignment
                    )
                )
                # At last data point, fetch current means and covariances for clusters
                return_curr_params = data_point_indices[-1] == ex_permutation[-1]
                curr_cluster_counts = [
                    len(cluster_examples) for cluster_examples in cluster_assignment
                ]

                res = computations_manager.sample_clusters_for_data_batch(
                    data_point_indices,
                    data_batch_clusters,
                    clusters_batch_removed,
                    cluster_counts_before_removal,
                    curr_cluster_counts,
                    curr_alpha,
                    return_curr_params=return_curr_params,
                )

                self._update_assignments_from_batch(
                    res["sampled_clusters"],
                    data_point_indices,
                    cluster_assignment,
                    examples_assignment,
                )

                progress_bar.set_description(f"Epoch: {iter_num}")
                progress_bar.set_postfix(
                    clusters_num=len(cluster_assignment), alpha=curr_alpha
                )

            cluster_params = {
                p_name: res[p_name] for p_name in self.posterior_params_names()
            }
            save_model_cond = (iter_num % self.skip_epochs_logging == 0) or (
                iter_num == iterations_num - 1
            )
            if out_dir is not None and save_model_cond:
                self.save_model(
                    out_dir,
                    cluster_params,
                    cluster_assignment,
                    data,
                    iter_num,
                    curr_alpha,
                    ass_ll,
                )
                logger.info("Saved model from iteration: %d" % iter_num)

            random.shuffle(ex_permutation)

        return {
            "cluster_params": cluster_params,
            "cluster_assignment": cluster_assignment,
            "alpha": self.alpha,
        }

    def _initialize_fit_state(
        self, data: torch.Tensor, data_np: np.ndarray
    ) -> Tuple[
        int, float, float, List[Set[int]], List[int], Dict[str, List[torch.Tensor]]
    ]:
        """
        Extracted method to handle initialization logic for the fit loop.

        Args:
            data (torch.Tensor): Dataset tensor.
            data_np (np.ndarray): Dataset numpy array.

        Returns:
            Tuple: Contains the initial iteration index, log-likelihood, alpha,
                cluster assignment sets, flat example assignments, and initial cluster parameters.
        """
        ass_ll = 0.0
        if self.restore_snapshot_pkl_path is not None:
            snapshot = fs_utils.read_pickle(self.restore_snapshot_pkl_path)
            curr_alpha, ass_ll = snapshot["alpha"], snapshot["ass_ll"]
            cluster_assignment = snapshot["cluster_assignment"]
            components_num = len(cluster_assignment)
            examples_assignment = self.get_examples_assignment(cluster_assignment)
            cluster_params = snapshot["cluster_params"]
            logger.info(
                "Restored params from snapshot path: %s, clusters num: %d"
                % (self.restore_snapshot_pkl_path, components_num)
            )
            init_iter = (
                int(
                    op.splitext(op.basename(self.restore_snapshot_pkl_path))[0].split(
                        "_"
                    )[-1]
                )
                + 1
            )
        else:
            # cluster_assignment = (cluster_id -> example_id[])[]
            # examples_assignment = (example_id -> cluster_id)[]
            cluster_assignment, examples_assignment = self.get_initial_assignment(
                data_np
            )
            components_num = len(cluster_assignment)
            curr_alpha = self.alpha
            prior_params = self.compute_prior_params(data, components_num)
            cluster_params = self.assign_initial_params(
                cluster_assignment, data, prior_params
            )
            init_iter = 0
            logger.info("Initialized params for first assignment")
            logger.info("Chosen first assignment, clusters num: %d" % components_num)

        return (
            init_iter,
            ass_ll,
            curr_alpha,
            cluster_assignment,
            examples_assignment,
            cluster_params,
        )

    def _update_assignments_from_batch(
        self,
        sampled_z_indices: List[torch.Tensor],
        data_point_indices: List[int],
        cluster_assignment: List[Set[int]],
        examples_assignment: List[int],
    ) -> None:
        """
        Extracted method to handle updating data point assignments after a batch is sampled.

        Args:
            sampled_z_indices (List[torch.Tensor]): The newly sampled cluster indices.
            data_point_indices (List[int]): The indices of the processed data points.
            cluster_assignment (List[Set[int]]): The sets mapping clusters to data points.
            examples_assignment (List[int]): The flat mapping of data points to clusters.
        """
        new_cluster_index = len(cluster_assignment)
        z_indices = [
            min(new_cluster_index, int(z_indx.item())) for z_indx in sampled_z_indices
        ]

        for z_indx, data_point_indx in zip(z_indices, data_point_indices):
            if z_indx == new_cluster_index:
                if len(cluster_assignment) == new_cluster_index:
                    cluster_assignment.append({data_point_indx})
                else:
                    cluster_assignment[z_indx].add(data_point_indx)
            else:
                cluster_assignment[z_indx].add(data_point_indx)

            examples_assignment[data_point_indx] = z_indx

    def remove_assignment_for_batch(
        self,
        data_point_indices: List[int],
        cluster_assignment: List[Set[int]],
        examples_assignment: List[int],
    ) -> Tuple[List[int], Dict[int, bool]]:
        """
        Removes a batch of data points from their current clusters to prepare for resampling.

        Args:
            data_point_indices (List[int]): Indices of data points in the current batch.
            cluster_assignment (List[Set[int]]): Current cluster mapping sets.
            examples_assignment (List[int]): Current flat assignment list.

        Returns:
            Tuple[List[int], Dict[int, bool]]: A list of the previous cluster indices for the batch,
                and a dictionary indicating whether a cluster became empty and was removed.
        """
        # In clusters_removed we wanna store info if cluster was removed
        batch_clusters: List[int] = []
        clusters_removed: Dict[int, bool] = {}

        for _, data_point_index in enumerate(data_point_indices):
            data_point_cluster, cluster_removed_after_data_point = (
                self.get_data_point_cluster(
                    data_point_index, cluster_assignment, examples_assignment
                )
            )
            batch_clusters.append(data_point_cluster)
            clusters_removed[data_point_cluster] = cluster_removed_after_data_point

        clusters_to_remove = set(
            [
                cluster
                for cluster, cluster_to_remove in clusters_removed.items()
                if cluster_to_remove
            ]
        )
        self.remove_clusters(
            clusters_to_remove, cluster_assignment, examples_assignment
        )
        return batch_clusters, clusters_removed

    def get_data_point_cluster(
        self,
        data_point_indx: int,
        cluster_assignment: List[Set[int]],
        examples_assignment: List[int],
    ) -> Tuple[int, bool]:
        """
        Retrieves a data point's current cluster and removes it from that cluster.

        Args:
            data_point_indx (int): The index of the data point.
            cluster_assignment (List[Set[int]]): The current cluster assignment sets.
            examples_assignment (List[int]): The current flat assignment list.

        Returns:
            Tuple[int, bool]: The cluster index the data point belonged to, and a boolean
                flag indicating if removing the point left the cluster empty.
        """
        data_point_cluster = examples_assignment[data_point_indx]
        cluster_assignment[data_point_cluster].remove(data_point_indx)

        # not necessary, but for the sake of clarity
        examples_assignment[data_point_indx] = -1

        if len(cluster_assignment[data_point_cluster]) == 0:
            return data_point_cluster, True

        return data_point_cluster, False

    @staticmethod
    def remove_clusters(
        clusters_to_remove: Set[int],
        cluster_assignment: List[Set[int]],
        examples_assignment: List[int],
    ):
        """
        Removes empty clusters from the state and re-indexes the remaining clusters.

        Args:
            clusters_to_remove (Set[int]): A set of cluster indices to delete.
            cluster_assignment (List[Set[int]]): The cluster assignment mapping sets.
            examples_assignment (List[int]): The flat assignment list to be updated.
        """
        new_cluster_assignment = [
            ass
            for index, ass in enumerate(cluster_assignment)
            if index not in clusters_to_remove
        ]
        cluster_assignment.clear()

        for cluster_idx, cluster_examples in enumerate(new_cluster_assignment):
            cluster_assignment.append(cluster_examples)
            for ex in cluster_examples:
                examples_assignment[ex] = cluster_idx

    def assign_initial_params(
        self,
        assignment: List[Set[int]],
        data: torch.Tensor,
        prior_params: Dict[str, torch.Tensor],
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Initializes cluster parameters for all clusters based on an initial assignment.

        Args:
            assignment (List[Set[int]]): The initial cluster assignments.
            data (torch.Tensor): The dataset.
            prior_params (Dict[str, torch.Tensor]): The baseline prior parameters.

        Returns:
            Dict[str, List[torch.Tensor]]: The initialized parameters grouped by parameter name.
        """
        clusters_params: Dict[str, List[torch.Tensor]] = {
            p_name: [] for p_name in self.posterior_params_names()
        }
        for examples in assignment:
            cluster_data = data[list(examples), :]

            cluster_params = self.initialize_params_for_cluster(
                cluster_data, prior_params
            )
            for p_name, p_val in cluster_params.items():
                clusters_params[p_name].append(p_val.to(dtype=data.dtype))

        return clusters_params

    def sample_alpha(self) -> float:
        """
        Samples an initial value for the concentration parameter alpha from a Gamma prior.

        Returns:
            float: A sampled value for alpha.
        """
        return stats.gamma.rvs(self.a, self.b)

    def update_alpha(self, old_alpha: float, n_points: int, k: int) -> float:
        """
        Updates the concentration parameter alpha using auxiliary variable sampling.

        Args:
            old_alpha (float): The current value of alpha.
            n_points (int): The total number of data points.
            k (int): The current number of active clusters.

        Returns:
            float: The newly sampled value for alpha.
        """
        u = stats.bernoulli.rvs(float(n_points) / (n_points + old_alpha))
        v = stats.beta.rvs(old_alpha + 1.0, n_points)

        new_alpha = np.random.gamma(self.a + k - 1 + u, 1.0 / (self.b - np.log(v)))
        return new_alpha

    def save_model(
        self,
        out_dir,
        cluster_params,
        cluster_assignment,
        data,
        it_index,
        curr_alpha,
        ll,
    ):
        """
        Serializes and saves the current model state to disk.

        Args:
            out_dir (str): Directory where the snapshot should be saved.
            cluster_params (dict): The current cluster parameters.
            cluster_assignment (list): The current mapping of points to clusters.
            data (torch.Tensor): The input dataset (saved only on iteration 0).
            it_index (int): The current iteration index.
            curr_alpha (float): The current value of alpha.
            ll (float): The current log-likelihood of the assignment.
        """
        import os.path as op

        obj = {
            "cluster_assignment": cluster_assignment,
            "cluster_params": cluster_params,
            "init_strategy": self.init_strategy,
            "alpha": curr_alpha,
            "ass_ll": ll,
        }
        try:
            if it_index == 0:
                obj.update({"data": data})
            fs_utils.write_pickle(obj, op.join(out_dir, "cgs_%d.pkl" % it_index))
        except Exception as e:
            logger.error(e)

    def print_model(self) -> str:
        """
        Formats the model's initialization parameters as a readable string.

        Returns:
            str: A formatted string representing the model setup.
        """
        return (
            f"CollapsedGibbsSampler(init_strategy={self.init_strategy}, "
            f"max_clusters_num={self.max_clusters_num}, batch_size={self.batch_size}, "
            f"a={self.a}, b={self.b}, "
            f"alpha={self.alpha}, skip_epochs_logging={self.skip_epochs_logging}, "
            f"skip_epochs_ll_calc={self.skip_epochs_ll_calc}, "
            f"restore_snapshot_pkl_path={self.restore_snapshot_pkl_path})"
        )

    def to(self, device: torch.device) -> CollapsedGibbsSampler:
        self.device = device
        return self
