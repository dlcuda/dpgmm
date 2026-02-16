from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Union

import torch
from loguru import logger
from torch import Tensor


class StudentTCalculator(ABC):
    """
    Abstract interface for evaluating multivariate Student-t log probability density
    functions in the context of DP mixture models.
    """

    @abstractmethod
    def log_pdf(
        self, data_batch: Tensor, parameters: List[Tensor], cluster_counts: Tensor
    ) -> Tensor:
        """
        Calculates the log probability density for a batch of data points against
        a set of cluster parameters.

        Args:
            data_batch (Tensor): The input data batch.
            parameters (List[Tensor]): A list of posterior parameters describing the distributions.
            cluster_counts (Tensor): The number of items in each evaluated cluster.

        Returns:
            Tensor: Log probabilities.
        """
        pass

    def to(self, device: torch.device) -> StudentTCalculator:
        self.device = device
        return self


class PriorPosteriorParametersKeeper(ABC):
    """
    Abstract interface for managing and sequentially updating Bayesian posterior
    parameters based on conjugate priors.
    """

    @abstractmethod
    def posterior_parameter_names(self) -> List[str]:
        """Returns the names of tracked posterior parameters."""
        pass

    @abstractmethod
    def assign_posterior_params(self, new_posterior_parameters: List[Tensor]):
        """Overwrites the current posterior parameters with new values."""
        pass

    @abstractmethod
    def downdate(
        self, data_points: Tensor, counts: Tensor, posterior_params: List[Tensor]
    ) -> List[Tensor]:
        """
        Computes the posterior parameters when data points are removed from a cluster.
        """
        pass

    @abstractmethod
    def update(
        self, data_points: Tensor, counts: Tensor, posterior_params: List[Tensor]
    ) -> List[Tensor]:
        """
        Computes the posterior parameters when data points are added to a cluster.
        """
        pass

    @abstractmethod
    def posterior_parameters(self) -> List[Tensor]:
        """Retrieves the current posterior parameters."""
        pass

    @abstractmethod
    def prior_parameters(self) -> List[Tensor]:
        """Retrieves the initial prior parameters."""
        pass

    @abstractmethod
    def posterior_parameters_dims(self) -> List[Union[Tuple[int], Tuple[int, int]]]:
        """Returns the shape dimensions for each tracked posterior parameter."""
        pass

    def to(self, device: torch.device) -> PriorPosteriorParametersKeeper:
        self.device = device
        return self


class TorchCgsSharedComputationsManager:
    """
    Handles batched tensor operations to efficiently sample new cluster assignments
    during Collapsed Gibbs Sampling.
    """

    def __init__(
        self,
        data: Tensor,
        init_clusters_num: int,
        parameters_keeper: PriorPosteriorParametersKeeper,
        student_t_calculator: StudentTCalculator,
        batch_size_max_value: int,
    ):
        """
        Initializes the shared computation manager for batched operations.

        Args:
            data (Tensor): The complete dataset tensor.
            init_clusters_num (int): Initial number of instantiated clusters.
            parameters_keeper (PriorPosteriorParametersKeeper): Object managing the parameters state.
            student_t_calculator (StudentTCalculator): Calculator for log-pdfs.
            batch_size_max_value (int): Maximum batch size for padding operations.
        """
        self.device = data.device
        self.init_clusters_num = init_clusters_num
        self.new_clusters_added = 0
        self.batch_size_max_value = batch_size_max_value

        self.data_sv = data.to(device=self.device, dtype=torch.float32)
        self.parameters_keeper = parameters_keeper
        self.student_t_calculator = student_t_calculator

        self.active_clusters = [True for _ in range(self.init_clusters_num)]
        self.new_old_cluster_mapping = list(range(init_clusters_num))
        self.n_points = data.shape[0]
        self.data_dim = data.shape[1]

        logger.info("Assigned initial values to covariances, means and data points")

    def sample_clusters_for_data_batch(
        self,
        data_points_indices: List[int],
        curr_batch_clusters: List[int],
        clusters_batch_removed: Dict[int, bool],
        cluster_counts_before_removal: List[int],
        cluster_counts_after_removal: List[int],
        alpha: float,
        return_curr_params: bool = False,
    ):
        """
        Coordinates the sampling of new cluster assignments for a batch of data points.

        It maps current cluster indices to the underlying parameter storage, triggers
        the vectorized sampling, and updates internal tracking structures.

        Args:
            data_points_indices (List[int]): Global indices of the data points in the batch.
            curr_batch_clusters (List[int]): The cluster indices these points were just removed from.
            clusters_batch_removed (Dict[int, bool]): Flags if removal emptied any clusters.
            cluster_counts_before_removal (List[int]): Counts prior to removing the batch.
            cluster_counts_after_removal (List[int]): Counts post removal (excluding the current batch).
            alpha (float): Current DP concentration parameter.
            return_curr_params (bool, optional): Whether to yield the updated parameters. Defaults to False.

        Returns:
            dict: Contains the newly sampled clusters, and optionally the updated parameters.
        """
        old_data_batch_clusters = [
            self.new_old_cluster_mapping[point_cluster]
            for point_cluster in curr_batch_clusters
        ]
        # cluster mask needs to be updated after taking the info of clusters removal
        # data_batch_clusters_ns, under index - i, keeps an info about examples count in cluster
        # occupied by example - data_points_indices[i]
        data_batch_clusters_ns: List[int] = []
        curr_clusters_to_remove: Set[int] = set()
        for curr_data_point_cluster, curr_data_point_old_cluster in zip(
            curr_batch_clusters, old_data_batch_clusters
        ):
            cluster_removed = clusters_batch_removed[curr_data_point_cluster]
            if cluster_removed:
                self.active_clusters[curr_data_point_old_cluster] = False
                curr_clusters_to_remove.add(curr_data_point_cluster)
                cluster_n = 0
            else:
                cluster_n = cluster_counts_before_removal[curr_data_point_cluster]

            data_batch_clusters_ns.append(cluster_n)

        self.new_old_cluster_mapping = [
            e
            for i, e in enumerate(self.new_old_cluster_mapping)
            if i not in curr_clusters_to_remove
        ]

        cluster_counts_t = torch.tensor(
            cluster_counts_after_removal, dtype=torch.float32, device=self.device
        )
        data_batch_clusters_ns_t = torch.tensor(
            data_batch_clusters_ns, dtype=torch.float32, device=self.device
        )
        data_points_indices_t = torch.tensor(
            data_points_indices, dtype=torch.int32, device=self.device
        )
        old_data_batch_clusters_t = torch.tensor(
            old_data_batch_clusters, dtype=torch.int32, device=self.device
        )
        active_clusters_t = torch.tensor(
            self.active_clusters, dtype=torch.bool, device=self.device
        )
        alpha_t = torch.tensor(alpha, dtype=torch.float32, device=self.device)
        new_old_cluster_mapping_t = torch.tensor(
            self.new_old_cluster_mapping, dtype=torch.int32, device=self.device
        )
        sampled_clusters, new_posterior_params = (
            self.torch_sample_clusters_for_data_batch(
                data_points_indices_t,
                data_batch_clusters_ns_t,
                old_data_batch_clusters_t,
                cluster_counts_t,
                active_clusters_t,
                alpha_t,
                new_old_cluster_mapping_t,
            )
        )
        sampled_clusters = list(sampled_clusters)
        # Update mapping, if new cluster will be added
        if any(c == len(cluster_counts_after_removal) for c in sampled_clusters):
            self.new_clusters_added += 1
            self.new_old_cluster_mapping.append(
                self.init_clusters_num + self.new_clusters_added - 1
            )
            self.active_clusters.append(True)

        res = {"sampled_clusters": sampled_clusters}
        if return_curr_params:
            res.update(
                {
                    p_name: p_val[self.active_clusters]
                    for p_name, p_val in zip(
                        self.parameters_keeper.posterior_parameter_names(),
                        new_posterior_params,
                    )
                }
            )

        return res

    def torch_sample_clusters_for_data_batch(
        self,
        data_point_indices: Tensor,
        batch_clusters_ns: Tensor,
        old_data_batch_clusters: Tensor,
        cluster_counts: Tensor,
        cluster_mask: Tensor,
        alpha: Tensor,
        new_old_cluster_mapping: Tensor,
    ):
        """
        Executes the tensor-based mathematics to generate new cluster assignments.

        Args:
            data_point_indices (Tensor): Indices of batch items.
            batch_clusters_ns (Tensor): Counts for corresponding clusters.
            old_data_batch_clusters (Tensor): Internal mapping of previous clusters.
            cluster_counts (Tensor): Sizes of currently active clusters.
            cluster_mask (Tensor): Boolean mask of active clusters.
            alpha (Tensor): Concentration parameter tensor.
            new_old_cluster_mapping (Tensor): Mapping table from active indices to stored tensors.

        Returns:
            Tuple[Tensor, List[Tensor]]: 1D tensor of newly sampled clusters, and updated params list.
        """
        data_points = self.data_sv[data_point_indices]
        # downdating old cluster params
        down_params = self.downdate_cluster_params_for_batch_in_aggregate(
            data_points, old_data_batch_clusters, batch_clusters_ns
        )

        self.parameters_keeper.assign_posterior_params(down_params)
        posterior_params = self.parameters_keeper.posterior_parameters()
        active_params = [p[cluster_mask] for p in posterior_params]

        cluster_probs = cluster_counts / (self.n_points + alpha - 1.0)
        # pred_post_log_probs.shape -> (batch_size, clusters_num)
        pred_post_log_probs = self.student_t_calculator.log_pdf(
            data_points, active_params, cluster_counts
        )

        # cluster_log_preds_unnorm.shape -> (batch_size, clusters_num)
        cluster_log_preds_unnorm = (
            torch.log(cluster_probs).unsqueeze(0) + pred_post_log_probs
        )
        new_cluster_prob = alpha / (self.n_points + alpha - 1.0)
        prior_params = [
            p.unsqueeze(0) for p in self.parameters_keeper.prior_parameters()
        ]
        # post_log_pred_new_cluster.shape -> (batch_size, 1)
        post_log_pred_new_cluster = self.student_t_calculator.log_pdf(
            data_points,
            prior_params,
            torch.tensor([0.0], dtype=torch.float32, device=self.device),
        )
        # new_cluster_log_prob_unnorm.shape -> (batch_size, 1)
        new_cluster_log_prob_unnorm = (
            torch.log(new_cluster_prob) + post_log_pred_new_cluster
        )

        # all_logs_probs_unnormalized.shape -> (batch_size, clusters_num + 1)
        all_logs_probs_unnormalized = torch.concat(
            [cluster_log_preds_unnorm, new_cluster_log_prob_unnorm], dim=1
        )

        # norm_const.shape -> (batch_size, 1)
        norm_const = torch.logsumexp(all_logs_probs_unnormalized, dim=-1, keepdim=True)
        all_probs_log_normalized = all_logs_probs_unnormalized - norm_const
        sampled_clusters = torch.reshape(
            torch.multinomial(input=all_probs_log_normalized.exp(), num_samples=1),
            (-1,),
        ).to(torch.int32)

        upd_params = self.update_cluster_params_for_batch_in_aggregate(
            sampled_clusters, data_points, cluster_counts, new_old_cluster_mapping
        )

        self.parameters_keeper.assign_posterior_params(upd_params)

        return sampled_clusters, upd_params

    def downdate_cluster_params_for_batch_in_aggregate(
        self,
        data_batch: Tensor,
        old_data_batch_clusters: Tensor,
        data_batch_clusters_ns: Tensor,
    ):
        """
        Batches the "downdating" of parameters—removing the effects of data points
        from their previous clusters in bulk via padding to avoid Python loop overhead.

        Args:
            data_batch (Tensor): The batch of data points.
            old_data_batch_clusters (Tensor): The mapped indices of their previous clusters.
            data_batch_clusters_ns (Tensor): Prior counts for those clusters.

        Returns:
            List[Tensor]: The updated parameter list post-removal.
        """

        def add_indices(
            curr_clusters_data_points: Tensor,
            i: int,
            curr_clusters_counts: Tensor,
            curr_max: int,
        ):
            cluster = non_empty_clusters_set[i]
            cluster_data_points_indices = torch.reshape(
                torch.nonzero(old_data_batch_clusters == cluster, as_tuple=False),
                (-1,),
            ).to(self.device)
            cluster_size = cluster_data_points_indices.size(0)

            # We are filling -1 to cluster data point indices, to have batch_size at the end
            empty_ = (
                torch.zeros(
                    (self.batch_size_max_value - cluster_size,), dtype=torch.int32
                )
                - 1
            )
            empty_ = empty_.to(self.device)

            curr_clusters_data_points[i] = torch.cat(
                [cluster_data_points_indices, empty_], dim=0
            )
            cluster_count = data_batch_clusters_ns[cluster_data_points_indices[0]].to(
                torch.float32
            )
            curr_clusters_counts[i] = cluster_count
            new_max = max(cluster_size, curr_max)

            return new_max

        def downdate_params_in_aggregate(i, counts_: Tensor, params_: List[Tensor]):
            i_data_points_indices = clusters_data_points_mtx[:, i]
            i_non_empty_indices = (
                torch.nonzero(i_data_points_indices != -1, as_tuple=False)
                .view(-1)
                .to(self.device)
            )
            i_data_points_indices_valid = i_data_points_indices[i_non_empty_indices]
            i_data_points = data_batch[i_data_points_indices_valid]
            i_params = [p[i_non_empty_indices] for p in params_]
            i_counts = counts_[i_non_empty_indices]

            new_i_params = self.parameters_keeper.downdate(
                i_data_points, i_counts, i_params
            )

            for param_, new_i_param in zip(params_, new_i_params):
                param_[i_non_empty_indices] = new_i_param

            return counts_ - 1.0, params_

        points_indices_with_non_empty_clusters = torch.reshape(
            torch.nonzero(data_batch_clusters_ns != 0, as_tuple=False), (-1,)
        ).to(self.device)

        points_non_empty_clusters = old_data_batch_clusters[
            points_indices_with_non_empty_clusters
        ]
        non_empty_clusters_set: Tensor = torch.unique(points_non_empty_clusters).to(
            self.device
        )

        clusters_num = non_empty_clusters_set.size(0)
        clusters_data_points = torch.zeros(
            (clusters_num, self.batch_size_max_value),
            dtype=torch.int32,
            device=self.device,
        )
        clusters_counts = torch.zeros(
            clusters_num, dtype=torch.float32, device=self.device
        )
        max_cluster_size = -1

        for i in range(clusters_num):
            max_cluster_size = add_indices(
                clusters_data_points, i, clusters_counts, max_cluster_size
            )

        clusters_data_points_mtx = clusters_data_points
        clusters_counts_mtx = clusters_counts
        posterior_params = self.parameters_keeper.posterior_parameters()
        init_params = [
            posterior_param[non_empty_clusters_set]
            for posterior_param in posterior_params
        ]

        for i in range(max_cluster_size):
            clusters_counts_mtx, init_params = downdate_params_in_aggregate(
                i, clusters_counts_mtx, init_params
            )

        final_params = init_params

        for posterior_param, final_param in zip(posterior_params, final_params):
            posterior_param[non_empty_clusters_set] = final_param

        return posterior_params

    def update_cluster_params_for_batch_in_aggregate(
        self,
        sampled_clusters: Tensor,
        data_batch: Tensor,
        cluster_counts: Tensor,
        new_old_cluster_mapping: Tensor,
    ):
        """
        Batches the "updating" of parameters—incorporating the effects of data points
        into their newly sampled clusters in bulk via tensor manipulations.

        Args:
            sampled_clusters (Tensor): The cluster indices assigned to the batch.
            data_batch (Tensor): The batch of data points.
            cluster_counts (Tensor): Sizes of the target clusters.
            new_old_cluster_mapping (Tensor): Mapping table to physical parameter tensors.

        Returns:
            List[Tensor]: The updated parameter list post-addition.
        """

        def add_indices(
            curr_clusters_data_points: List[Tensor], curr_counts: List[Tensor]
        ):
            curr_max = -1

            for i in range(clusters_num):
                cluster = unique_clusters_set[i]
                cluster_data_points_indices = torch.reshape(
                    torch.nonzero(sampled_clusters == cluster, as_tuple=False),
                    (-1,),
                ).to(torch.int32)

                cluster_size = cluster_data_points_indices.size(0)
                empty_ = torch.full(
                    (self.batch_size_max_value - cluster_size,),
                    -1,
                    dtype=torch.int32,
                    device=self.device,
                )
                # We are filling -1 to cluster data point indices, to have batch_size at the end
                curr_clusters_data_points.append(
                    torch.concat([cluster_data_points_indices, empty_], dim=0)
                )
                curr_max = max(cluster_size, curr_max)

                if cluster == new_cluster_index:
                    curr_counts.append(
                        torch.tensor(0.0, dtype=torch.float32, device=self.device)
                    )
                else:
                    curr_counts.append(cluster_counts[cluster])

            return curr_max

        def collect_init_params(i, params_ta: List[Tensor]):
            cluster = unique_clusters_set[i]

            if cluster.item() == new_cluster_index:
                for param_ta, prior_param in zip(params_ta, prior_params):
                    param_ta[i] = prior_param

                return param_ta

            for param_ta, posterior_param in zip(params_ta, posterior_params):
                param_ta[i] = posterior_param[new_old_cluster_mapping[cluster]]

            return params_ta

        def update_params_in_aggregate(i, counts_: Tensor, params_: List[Tensor]):
            i_data_points_indices = clusters_data_points_mtx[:, i]
            i_non_empty_indices = (
                torch.nonzero(i_data_points_indices != -1, as_tuple=False)
                .view(-1)
                .to(self.device)
            )
            i_data_points_indices_valid = i_data_points_indices[i_non_empty_indices]
            i_data_points = data_batch[i_data_points_indices_valid]
            i_params = [param_[i_non_empty_indices] for param_ in params_]
            i_counts = counts_[i_non_empty_indices]
            new_i_params = self.parameters_keeper.update(
                i_data_points, i_counts, i_params
            )

            for param_, new_i_param in zip(params_, new_i_params):
                param_[i_non_empty_indices] = new_i_param

            return counts_ + 1.0, params_

        def prepare_indices_for_assignment(
            curr_ta: Tensor, target_params_: List[Tensor]
        ):
            for i in range(clusters_num):
                cluster = unique_clusters_set[i]

                if cluster.item() == new_cluster_index:
                    target_params_ = [
                        torch.cat([posterior_param, final_param[i : i + 1]], dim=0)
                        for posterior_param, final_param in zip(
                            posterior_params, final_params
                        )
                    ]
                    curr_ta[i] = torch.tensor(
                        target_params_[0].shape[0] - 1, device=self.device
                    )
                else:
                    curr_ta[i] = new_old_cluster_mapping[cluster]

            return curr_ta, target_params_

        prior_params = self.parameters_keeper.prior_parameters()
        posterior_params = self.parameters_keeper.posterior_parameters()

        unique_clusters_set: Tensor = torch.unique(sampled_clusters).to(self.device)

        new_cluster_index = cluster_counts.size(0)
        clusters_num = unique_clusters_set.size(0)

        clusters_data_points: List[Tensor] = []
        cluster_counts_ta: List[Tensor] = []

        max_cluster_size = add_indices(clusters_data_points, cluster_counts_ta)
        filled_clusters_data_points = clusters_data_points
        filled_cluster_counts = cluster_counts_ta

        clusters_data_points_mtx = torch.stack(filled_clusters_data_points).to(
            self.device
        )
        cluster_counts_mtx = torch.stack(filled_cluster_counts).to(self.device)

        init_params: List[Tensor] = [
            torch.empty(
                (clusters_num, *param_shape), dtype=torch.float32, device=self.device
            )
            for param_shape in self.parameters_keeper.posterior_parameters_dims()
        ]

        for i in range(clusters_num):
            collect_init_params(i, init_params)

        final_params = init_params

        for i in range(max_cluster_size):
            cluster_counts_mtx, final_params = update_params_in_aggregate(
                i, cluster_counts_mtx, init_params
            )

        indices = torch.zeros(clusters_num, dtype=torch.int32, device=self.device)
        target_params = posterior_params

        filled_indices, target_params = prepare_indices_for_assignment(
            indices, target_params
        )

        result_params = target_params

        for param, final_param in zip(result_params, final_params):
            param[filled_indices] = final_param

        return result_params

    def to(self, device: torch.device) -> TorchCgsSharedComputationsManager:
        self.device = device
        self.data_sv = self.data_sv.to(device=self.device)
        self.parameters_keeper = self.parameters_keeper.to(device=self.device)
        self.student_t_calculator = self.student_t_calculator.to(device=self.device)
        return self
