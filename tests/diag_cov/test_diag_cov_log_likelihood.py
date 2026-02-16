import pytest
import torch

from dpgmm.samplers.cgs.variants.diag_cov.log_likelihood import (
    DiagonalCovarianceLogLikelihoodCalculator,
)
from dpgmm.utils.device import get_device

device = get_device()


# ----------------------
# Fixtures
# ----------------------


@pytest.fixture
def simple_mixture_data():
    # Simple 2D dataset, 3 points, 2 clusters
    data = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], device=device)
    cluster_assignment = [{0, 1}, {2}]
    post_means = [torch.zeros(2, device=device), torch.ones(2, device=device)]
    post_vars = [torch.ones(2, device=device), torch.ones(2, device=device)]
    return data, cluster_assignment, post_means, post_vars


# ----------------------
# Unit tests
# ----------------------


def test_t_student_log_pdf_shape_and_finite(simple_mixture_data):
    """Ensure log-likelihoods are finite and shaped correctly."""
    from dpgmm.samplers.cgs.variants.diag_cov.student_t_calculator import (
        DiagCovarianceStudentTCalculator,
    )

    data, cluster_assignment, post_means, post_vars = simple_mixture_data
    calc = DiagCovarianceStudentTCalculator(device=device, data_dim=data.shape[1])

    means = post_means[0].unsqueeze(0)
    vars_ = post_vars[0].unsqueeze(0)
    counts = torch.tensor(
        [len(cluster_assignment[0])], dtype=torch.float32, device=device
    )

    lls = calc.t_student_log_pdf_torch(means, vars_, data, counts).squeeze(-1)

    assert lls.shape == (data.shape[0],)
    assert torch.isfinite(lls).all()


def test_log_likelihood_peak_at_mean():
    """Check that likelihood is maximized near the cluster mean."""
    from dpgmm.samplers.cgs.variants.diag_cov.student_t_calculator import (
        DiagCovarianceStudentTCalculator,
    )

    data = torch.tensor([[0.0, 0.0], [0.1, -0.1], [1.0, 1.0]], device=device)
    mean = torch.zeros(2, device=device)
    var = torch.ones(2, device=device)
    calc = DiagCovarianceStudentTCalculator(device=device, data_dim=data.shape[1])

    means = mean.unsqueeze(0)
    vars_ = var.unsqueeze(0)
    counts = torch.tensor([3.0], device=device)

    lls = calc.t_student_log_pdf_torch(means, vars_, data, counts)
    assert lls[0] == lls.max(), "log-likelihood should be highest at mean"


def test_data_log_likelihood_scalar_and_finite(simple_mixture_data):
    """End-to-end test of DiagonalCovarianceLogLikelihoodCalculator."""
    data, cluster_assignment, post_means, post_vars = simple_mixture_data
    loglike = DiagonalCovarianceLogLikelihoodCalculator(
        device=device, data_dim=data.shape[1]
    )

    ll = loglike.data_log_likelihood(cluster_assignment, data, post_means, post_vars)

    assert ll.dim() == 0
    assert torch.isfinite(ll)


def test_data_log_likelihood_changes_with_assignment(simple_mixture_data):
    """Changing assignments should change total likelihood."""
    data, cluster_assignment, post_means, post_vars = simple_mixture_data
    loglike = DiagonalCovarianceLogLikelihoodCalculator(
        device=device, data_dim=data.shape[1]
    )

    ll1 = loglike.data_log_likelihood(cluster_assignment, data, post_means, post_vars)

    # Change assignment
    cluster_assignment_new = [{0}, {1, 2}]
    ll2 = loglike.data_log_likelihood(
        cluster_assignment_new, data, post_means, post_vars
    )
    assert ll1 != ll2, "Changing cluster assignment should change log-likelihood"


@pytest.mark.parametrize("data_dim,n_clusters", [(1, 2), (2, 2), (3, 2), (5, 3)])
def test_high_dimensional_sampling(data_dim, n_clusters):
    """Stress test across multiple dimensions and cluster counts."""
    from dpgmm.samplers.cgs.variants.diag_cov.student_t_calculator import (
        DiagCovarianceStudentTCalculator,
    )

    data = torch.randn(10, data_dim, device=device)
    post_means = [torch.zeros(data_dim, device=device) for _ in range(n_clusters)]
    post_vars = [torch.ones(data_dim, device=device) for _ in range(n_clusters)]

    calc = DiagCovarianceStudentTCalculator(device=device, data_dim=data_dim)

    for mean, var in zip(post_means, post_vars):
        counts = torch.tensor([5.0], device=device)
        lls = calc.t_student_log_pdf_torch(
            mean.unsqueeze(0), var.unsqueeze(0), data, counts
        ).squeeze(-1)
        assert lls.shape == (data.shape[0],)
        assert torch.isfinite(lls).all()


def test_empty_clusters_handling():
    """Ensure empty clusters are handled gracefully."""
    data = torch.randn(3, 2, device=device)
    cluster_assignment = [set(), {0, 1, 2}]
    post_means = [torch.zeros(2, device=device), torch.ones(2, device=device)]
    post_vars = [torch.ones(2, device=device), torch.ones(2, device=device)]
    loglike = DiagonalCovarianceLogLikelihoodCalculator(
        device=device, data_dim=data.shape[1]
    )

    ll = loglike.data_log_likelihood(cluster_assignment, data, post_means, post_vars)
    assert torch.isfinite(ll)
