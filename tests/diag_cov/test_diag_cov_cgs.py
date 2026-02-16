import numpy as np
import pytest
import torch

from dpgmm.samplers.cgs.variants.diag_cov.algorithm import (
    DiagCovarianceCollapsedGibbsSampler,
)

# ----------------------
# Fixtures
# ----------------------


@pytest.fixture(scope="module")
def toy_data():
    """2D Gaussian blobs, 30 points each."""
    np.random.seed(0)
    cluster1 = np.random.normal(loc=[-2, -2], scale=0.5, size=(30, 2))
    cluster2 = np.random.normal(loc=[2, 2], scale=0.5, size=(30, 2))
    return np.vstack([cluster1, cluster2]).astype(np.float32)


@pytest.fixture(scope="module")
def sampler():
    return DiagCovarianceCollapsedGibbsSampler(
        init_strategy="init_data_stats",
        max_clusters_num=5,
        batch_size=5,
    )


# ----------------------
# Unit tests (helpers)
# ----------------------


def test_init_mean_and_var(toy_data, sampler: DiagCovarianceCollapsedGibbsSampler):
    data_t = torch.tensor(toy_data)
    mean = sampler.init_mean(data_t)
    var = sampler.init_var(data_t, components_num=3)
    assert mean.shape == (2,)
    assert var.shape == (2,)
    assert torch.all(var > 0), "Variances must be positive"


def test_compute_prior_params(toy_data, sampler: DiagCovarianceCollapsedGibbsSampler):
    data_t = torch.tensor(toy_data)
    prior = sampler.compute_prior_params(data_t, components_num=3)
    assert "mean_0" in prior and "var_0" in prior
    mean_0, var_0 = prior["mean_0"], prior["var_0"]
    assert mean_0.shape == (2,)
    assert var_0.shape == (2,)
    assert torch.all(var_0 > 0)


def test_initialize_params_for_cluster(
    toy_data, sampler: DiagCovarianceCollapsedGibbsSampler
):
    data_t = torch.tensor(toy_data)
    prior = sampler.compute_prior_params(data_t, components_num=2)
    cluster_data = data_t[:10]
    params = sampler.initialize_params_for_cluster(cluster_data, prior)
    assert set(params.keys()) == {"mean", "var"}
    assert params["mean"].shape == (2,)
    assert params["var"].shape == (2,)
    assert torch.all(params["var"] > 0)


def test_alpha_sampling_and_update(sampler: DiagCovarianceCollapsedGibbsSampler):
    alpha = sampler.sample_alpha()
    assert alpha > 0
    new_alpha = sampler.update_alpha(alpha, n_points=10, k=3)
    assert new_alpha > 0


# ----------------------
# Cluster assignment logic
# ----------------------


def test_get_initial_assignment(toy_data, sampler: DiagCovarianceCollapsedGibbsSampler):
    cluster_assignment, examples_assignment = sampler.get_initial_assignment(toy_data)
    n_points = toy_data.shape[0]
    # Every point assigned
    assert sorted(sum((list(c) for c in cluster_assignment), [])) == list(
        range(n_points)
    )
    # Inverse mapping length check
    assert len(examples_assignment) == n_points


# ----------------------
# Edge cases
# ----------------------


def test_remove_and_reassign(sampler: DiagCovarianceCollapsedGibbsSampler):
    # Simple fake assignment
    cluster_assignment = [{0, 1}, {2}]
    examples_assignment = [0, 0, 1]
    batch_indices = [2]
    batch_clusters, clusters_removed = sampler.remove_assignment_for_batch(
        batch_indices, cluster_assignment, examples_assignment
    )
    # Point 2 removed from its cluster, cluster becomes empty
    assert clusters_removed[1] is True
    assert examples_assignment[2] == -1


# ----------------------
# Log-likelihood consistency
# ----------------------


def test_data_log_likelihood_consistency(
    toy_data, sampler: DiagCovarianceCollapsedGibbsSampler
):
    data_t = torch.tensor(toy_data)
    cluster_assignment, _ = sampler.get_initial_assignment(toy_data)
    prior = sampler.compute_prior_params(data_t, components_num=2)
    cluster_params = {"mean": [], "var": []}

    for cluster in cluster_assignment:
        cluster_data = data_t[list(cluster)]
        params = sampler.initialize_params_for_cluster(cluster_data, prior)
        cluster_params["mean"].append(params["mean"])
        cluster_params["var"].append(params["var"])

    log_likelihood = sampler.data_log_likelihood(
        cluster_assignment, data_t, cluster_params
    )
    assert isinstance(log_likelihood, float)
    assert np.isfinite(log_likelihood)
