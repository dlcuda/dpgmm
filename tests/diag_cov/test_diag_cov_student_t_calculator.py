import numpy as np
import torch

from dpgmm.samplers.cgs.variants.diag_cov.student_t_calculator import (
    DiagCovarianceStudentTCalculator,
)
from dpgmm.utils.device import get_device

device = get_device()
DEFAULT_DATA_DIM = 3


def make_diag_variances(clusters: int, dim: int):
    """Create positive diagonal variances for testing."""
    return torch.abs(torch.randn(clusters, dim, device=device)) + 1.0


def test_log_pdf_output_shape_diag():
    calc = DiagCovarianceStudentTCalculator(device=device, data_dim=DEFAULT_DATA_DIM)

    batch_size = 5
    clusters = 4
    data_batch = torch.randn(batch_size, DEFAULT_DATA_DIM, device=device)
    mean_matrix = torch.randn(clusters, DEFAULT_DATA_DIM, device=device)
    vars_matrix = make_diag_variances(clusters, DEFAULT_DATA_DIM)
    cluster_counts = torch.randint(
        1, 10, (clusters,), device=device, dtype=torch.float32
    )

    out = calc.log_pdf(data_batch, [mean_matrix, vars_matrix], cluster_counts)

    assert out.shape == (batch_size, clusters)
    assert torch.isfinite(out).all()


def test_single_cluster_identity_variance_diag():
    calc = DiagCovarianceStudentTCalculator(device=device, data_dim=DEFAULT_DATA_DIM)

    batch_size = 2
    clusters = 1
    data_batch = torch.zeros(batch_size, DEFAULT_DATA_DIM, device=device)
    mean_matrix = torch.zeros(clusters, DEFAULT_DATA_DIM, device=device)
    vars_matrix = torch.ones(clusters, DEFAULT_DATA_DIM, device=device)
    cluster_counts = torch.tensor([5.0], device=device)

    out = calc.log_pdf(data_batch, [mean_matrix, vars_matrix], cluster_counts)

    assert out.shape == (batch_size, clusters)
    assert torch.isfinite(out).all()


def test_increasing_cluster_counts_changes_result_diag():
    calc = DiagCovarianceStudentTCalculator(device=device, data_dim=DEFAULT_DATA_DIM)

    batch_size = 3
    clusters = 2
    data_batch = torch.randn(batch_size, DEFAULT_DATA_DIM, device=device)
    mean_matrix = torch.randn(clusters, DEFAULT_DATA_DIM, device=device)
    vars_matrix = make_diag_variances(clusters, DEFAULT_DATA_DIM)

    cluster_counts_small = torch.tensor([1.0, 1.0], device=device)
    cluster_counts_large = torch.tensor([50.0, 50.0], device=device)

    out_small = calc.log_pdf(
        data_batch, [mean_matrix, vars_matrix], cluster_counts_small
    )
    out_large = calc.log_pdf(
        data_batch, [mean_matrix, vars_matrix], cluster_counts_large
    )

    assert not torch.allclose(out_small, out_large)


def test_no_nan_or_inf_with_large_batch_diag():
    calc = DiagCovarianceStudentTCalculator(device=device, data_dim=DEFAULT_DATA_DIM)

    batch_size = 200
    clusters = 3
    data_batch = torch.randn(batch_size, DEFAULT_DATA_DIM, device=device)
    mean_matrix = torch.randn(clusters, DEFAULT_DATA_DIM, device=device)
    vars_matrix = make_diag_variances(clusters, DEFAULT_DATA_DIM)
    cluster_counts = torch.randint(
        1, 20, (clusters,), device=device, dtype=torch.float32
    )

    out = calc.log_pdf(data_batch, [mean_matrix, vars_matrix], cluster_counts)

    assert torch.isfinite(out).all()


def test_bayesian_student_t_consistency_1d_diag():
    dim = 1
    calc = DiagCovarianceStudentTCalculator(device=device, data_dim=dim)

    batch_size = 21
    clusters = 1
    data = torch.linspace(-5, 5, batch_size, device=device).unsqueeze(1)
    mean_matrix = torch.zeros(clusters, dim, device=device)
    vars_matrix = torch.ones(clusters, dim, device=device)
    cluster_counts = torch.tensor([10.0], device=device)

    out = calc.log_pdf(data, [mean_matrix, vars_matrix], cluster_counts)
    out = out.squeeze().detach().cpu().numpy()

    # Symmetry check
    np.testing.assert_allclose(out, out[::-1], rtol=1e-5, atol=1e-5)

    # Peak at mean
    center_idx = batch_size // 2
    assert out[center_idx] == max(out)

    # Values decay as |x| increases
    assert out[center_idx] > out[0]
    assert out[center_idx] > out[-1]


def test_bayesian_student_t_consistency_2d_diag():
    dim = 2
    calc = DiagCovarianceStudentTCalculator(device=device, data_dim=dim)

    clusters = 1
    mean_matrix = torch.zeros(clusters, dim, device=device)
    vars_matrix = torch.ones(clusters, dim, device=device)
    cluster_counts = torch.tensor([10.0], device=device)

    points = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        device=device,
    )

    out = calc.log_pdf(points, [mean_matrix, vars_matrix], cluster_counts).squeeze()

    # Center should be highest
    assert out[0] == torch.max(out)

    # Equal radii -> approx equal logpdf
    torch.testing.assert_close(out[1], out[2], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[3], out[4], rtol=1e-5, atol=1e-5)

    # Further distance -> smaller logpdf
    assert out[1] > out[3]
