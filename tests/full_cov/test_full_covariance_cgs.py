import numpy as np
import pytest
import torch

from dpgmm.cholesky_update import CholeskyUpdateAdapter
from dpgmm.cholesky_update.variants.cholesky_update_torch import CholeskyUpdateTorch
from dpgmm.samplers.cgs.variants.full_cov.algorithm import (
    FullCovarianceCollapsedGibbsSampler,
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
    return FullCovarianceCollapsedGibbsSampler(
        init_strategy="init_data_stats",
        max_clusters_num=5,
        batch_size=5,
    )


# ----------------------
# Unit tests (helpers)
# ----------------------


def test_init_mean_and_cov(toy_data, sampler):
    data_t = torch.tensor(toy_data)
    mean = sampler.init_mean(data_t)
    cov = sampler.init_cov(data_t, components_num=3)
    assert mean.shape == (2,)
    assert cov.shape == (2, 2)
    assert torch.allclose(cov, torch.diag(torch.diag(cov)))  # diagonal


def test_compute_prior_params(toy_data, sampler):
    data_t = torch.tensor(toy_data)
    prior = sampler.compute_prior_params(data_t, components_num=3)
    assert "mean_0" in prior and "cov_chol_0" in prior
    chol = prior["cov_chol_0"]
    assert chol.shape[0] == chol.shape[1]
    # Ensure it's lower triangular
    assert torch.allclose(chol, torch.tril(chol))


def test_initialize_params_for_cluster(toy_data, sampler):
    data_t = torch.tensor(toy_data)
    prior = sampler.compute_prior_params(data_t, components_num=2)
    cluster_data = data_t[:10]
    params = sampler.initialize_params_for_cluster(cluster_data, prior)
    assert set(params.keys()) == {"mean", "cov_chol"}
    assert params["mean"].shape == (2,)
    assert params["cov_chol"].shape == (2, 2)


def test_alpha_sampling_and_update(sampler):
    alpha = sampler.sample_alpha()
    assert alpha > 0
    new_alpha = sampler.update_alpha(alpha, n_points=10, k=3)
    assert new_alpha > 0


# ----------------------
# Cluster assignment logic
# ----------------------


def test_get_initial_assignment(toy_data, sampler):
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


def test_remove_and_reassign(sampler):
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


def test_explicit_backend_survives_state_device_move(monkeypatch, toy_data):
    # Falling back to auto here could run a different kernel in a comparison.
    def require_explicit_torch(device, backend="auto"):
        assert backend == "torch"
        return CholeskyUpdateTorch(device)

    monkeypatch.setattr(
        CholeskyUpdateAdapter, "get_cholesky_updater", require_explicit_torch
    )
    sampler = FullCovarianceCollapsedGibbsSampler(
        "init_data_stats", 2, cholesky_backend="torch", verbose=False
    ).to(torch.device("cpu"))
    data = torch.tensor(toy_data)
    prior = sampler.compute_prior_params(data, 2)
    posterior = sampler.initialize_params_for_cluster(data[:10], prior)
    values = {
        **prior,
        **{name: value.unsqueeze(0) for name, value in posterior.items()},
    }
    keeper = sampler.get_params_keeper(2, values, 1)
    keeper.to(torch.device("cpu"))
    assert isinstance(keeper.cholesky_updater, CholeskyUpdateTorch)
    assert sampler._wandb_config()["cholesky_backend"] == "torch"


@pytest.mark.parametrize(
    "backend", ["triton_full", "triton_full_barrier", "triton_full_local"]
)
def test_explicit_triton_backend_does_not_silently_use_cpu_torch(
    monkeypatch, toy_data, backend
):
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    sampler = FullCovarianceCollapsedGibbsSampler(
        "init_data_stats", 2, cholesky_backend=backend, verbose=False
    )
    data = torch.tensor(toy_data)
    prior = sampler.compute_prior_params(data, 2)
    posterior = sampler.initialize_params_for_cluster(data[:10], prior)
    values = {
        **prior,
        **{name: value.unsqueeze(0) for name, value in posterior.items()},
    }
    with pytest.raises(ValueError, match="Explicit Triton backends require CUDA"):
        sampler.get_params_keeper(2, values, 1)
