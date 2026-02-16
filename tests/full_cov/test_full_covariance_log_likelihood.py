import pytest
import torch

from dpgmm.samplers.cgs.variants.full_cov.log_likelihood import (
    FullCovarianceLogLikelihood,
)
from dpgmm.utils.device import get_device

device = get_device()


@pytest.fixture
def simple_mixture_data():
    # Simple 2D dataset, 3 points, 2 clusters
    data = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], device=device)
    cluster_assignment = [{0, 1}, {2}]
    post_means = [torch.zeros(2, device=device), torch.ones(2, device=device)]
    post_cov_chols = [torch.eye(2, device=device), torch.eye(2, device=device)]
    return data, cluster_assignment, post_means, post_cov_chols


def test_normal_log_likelihood_shape_and_finite(simple_mixture_data):
    data, cluster_assignment, post_means, post_cov_chols = simple_mixture_data
    mean = post_means[0]
    cov_chol = post_cov_chols[0]

    ll = FullCovarianceLogLikelihood.normal_log_likelihood(data, mean, cov_chol)
    assert ll.shape == (data.shape[0],)
    assert torch.isfinite(ll).all()


def test_normal_log_likelihood_peak_at_mean():
    data = torch.tensor([[0.0, 0.0], [0.1, -0.1], [1.0, 1.0]], device=device)
    mean = torch.zeros(2, device=device)
    cov_chol = torch.eye(2, device=device)

    ll = FullCovarianceLogLikelihood.normal_log_likelihood(data, mean, cov_chol)
    # Log-likelihood should be maximal at the mean
    assert ll[0] == ll.max()


def test_sample_marginals_output_shapes_and_pd(simple_mixture_data):
    data, cluster_assignment, post_means, post_cov_chols = simple_mixture_data
    loglike = FullCovarianceLogLikelihood(nu_0=2, alpha_0=1)

    means, cov_chols = loglike._sample_marginals_for_mean_and_sigma(
        cluster_assignment, post_means, post_cov_chols, data_dim=2
    )
    assert len(means) == len(cluster_assignment)
    assert len(cov_chols) == len(cluster_assignment)

    for m, c in zip(means, cov_chols):
        assert m.shape[0] == 2
        assert c.shape == (2, 2)
        # Covariance must be positive definite
        eigvals = torch.linalg.eigvals(c @ c.T).real
        assert (eigvals > 0).all()


def test_data_log_likelihood_scalar_and_finite(simple_mixture_data):
    data, cluster_assignment, post_means, post_cov_chols = simple_mixture_data
    loglike = FullCovarianceLogLikelihood(nu_0=2, alpha_0=1)

    ll = loglike.data_log_likelihood(
        cluster_assignment, data, post_means, post_cov_chols
    )
    assert ll.dim() == 0  # scalar
    assert torch.isfinite(ll)


def test_data_log_likelihood_changes_with_assignment(simple_mixture_data):
    data, cluster_assignment, post_means, post_cov_chols = simple_mixture_data
    loglike = FullCovarianceLogLikelihood(nu_0=2, alpha_0=1)

    ll1 = loglike.data_log_likelihood(
        cluster_assignment, data, post_means, post_cov_chols
    )

    # Change assignment
    cluster_assignment_new = [{0}, {1, 2}]
    ll2 = loglike.data_log_likelihood(
        cluster_assignment_new, data, post_means, post_cov_chols
    )
    # Likelihood should change
    assert ll1 != ll2


@pytest.mark.parametrize("data_dim,n_clusters", [(1, 2), (2, 2), (3, 2), (5, 3)])
def test_high_dimensional_sampling(data_dim, n_clusters):
    cluster_assignment = [set(range(5)), set(range(5, 10))]
    post_means = [torch.zeros(data_dim, device=device) for _ in range(n_clusters)]
    post_cov_chols = [torch.eye(data_dim, device=device) for _ in range(n_clusters)]
    loglike = FullCovarianceLogLikelihood(nu_0=5, alpha_0=1)

    means, cov_chols = loglike._sample_marginals_for_mean_and_sigma(
        cluster_assignment, post_means, post_cov_chols, data_dim
    )
    for m, c in zip(means, cov_chols):
        assert m.shape[0] == data_dim
        assert c.shape == (data_dim, data_dim)
        eigvals = torch.linalg.eigvals(c @ c.T).real
        assert (eigvals > 0).all()


def test_empty_clusters_handling():
    data = torch.randn(3, 2, device=device)
    cluster_assignment = [set(), {0, 1, 2}]
    post_means = [torch.zeros(2, device=device), torch.ones(2, device=device)]
    post_cov_chols = [torch.eye(2, device=device), torch.eye(2, device=device)]
    loglike = FullCovarianceLogLikelihood(nu_0=2, alpha_0=1)

    # Should not crash
    ll = loglike.data_log_likelihood(
        cluster_assignment, data, post_means, post_cov_chols
    )
    assert torch.isfinite(ll)
