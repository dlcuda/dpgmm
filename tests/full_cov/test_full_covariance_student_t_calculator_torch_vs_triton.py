import pytest
import torch

triton = pytest.importorskip("triton")

from dpgmm.samplers.cgs.variants.full_cov.student_t_calculator.student_t_torch import (
    FullCovarianceStudentTCalculatorTorch,
)
from dpgmm.samplers.cgs.variants.full_cov.student_t_calculator.student_t_triton import (
    FullCovarianceStudentTCalculatorTriton,
)
from dpgmm.utils.device import get_device

device = get_device()
DEFAULT_DATA_DIM = 3


@pytest.fixture(autouse=True)
def skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("Skipping test because no GPU is available")


def make_cholesky_matrices(clusters: int, dim: int):
    covs = []
    for _ in range(clusters):
        A = torch.randn(dim, dim, device=device)
        cov = A @ A.T + dim * torch.eye(dim, device=device)
        chol = torch.linalg.cholesky(cov)
        covs.append(chol)
    return torch.stack(covs)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("clusters", [1, 2, 5])
@pytest.mark.parametrize("batch_size", [1, 5, 10])
def test_triton_vs_torch_general(dim, clusters, batch_size):
    """Compare Triton vs Torch outputs on random data."""
    torch.manual_seed(42)

    data_batch = torch.randn(batch_size, dim, device=device)
    mean_matrix = torch.randn(clusters, dim, device=device)
    chol_cov_matrices = make_cholesky_matrices(clusters, dim)
    cluster_counts = torch.randint(
        1, 20, (clusters,), device=device, dtype=torch.float32
    )

    calc_torch = FullCovarianceStudentTCalculatorTorch(
        device=device, data_dim=DEFAULT_DATA_DIM
    )
    calc_triton = FullCovarianceStudentTCalculatorTriton(
        device=device, data_dim=DEFAULT_DATA_DIM
    )

    out_torch = calc_torch.log_pdf(
        data_batch, [mean_matrix, chol_cov_matrices], cluster_counts
    )
    out_triton = calc_triton.log_pdf(
        data_batch, [mean_matrix, chol_cov_matrices], cluster_counts
    )

    torch.testing.assert_close(out_torch, out_triton, rtol=1e-5, atol=1e-6)


def test_triton_vs_torch_identity_covariance():
    """Single cluster, identity covariance, zero mean."""
    dim = 3
    batch_size = 4
    clusters = 1

    data_batch = torch.randn(batch_size, dim, device=device)
    mean_matrix = torch.zeros(clusters, dim, device=device)
    chol_cov_matrices = torch.eye(dim, device=device).unsqueeze(0)
    cluster_counts = torch.tensor([5.0], device=device)

    calc_torch = FullCovarianceStudentTCalculatorTorch(device=device, data_dim=dim)
    calc_triton = FullCovarianceStudentTCalculatorTriton(device=device, data_dim=dim)

    out_torch = calc_torch.log_pdf(
        data_batch, [mean_matrix, chol_cov_matrices], cluster_counts
    )
    out_triton = calc_triton.log_pdf(
        data_batch, [mean_matrix, chol_cov_matrices], cluster_counts
    )

    torch.testing.assert_close(out_torch, out_triton, rtol=1e-5, atol=1e-6)


def test_triton_vs_torch_large_cluster_counts():
    """Check outputs for high cluster counts."""
    dim = 2
    batch_size = 3
    clusters = 3

    data_batch = torch.randn(batch_size, dim, device=device)
    mean_matrix = torch.randn(clusters, dim, device=device)
    chol_cov_matrices = make_cholesky_matrices(clusters, dim)
    cluster_counts_small = torch.tensor([1.0, 1.0, 1.0], device=device)
    cluster_counts_large = torch.tensor([50.0, 50.0, 50.0], device=device)

    calc_torch = FullCovarianceStudentTCalculatorTorch(device=device, data_dim=dim)
    calc_triton = FullCovarianceStudentTCalculatorTriton(device=device, data_dim=dim)

    out_small_torch = calc_torch.log_pdf(
        data_batch, [mean_matrix, chol_cov_matrices], cluster_counts_small
    )
    out_small_triton = calc_triton.log_pdf(
        data_batch, [mean_matrix, chol_cov_matrices], cluster_counts_small
    )
    torch.testing.assert_close(out_small_torch, out_small_triton, rtol=1e-5, atol=1e-6)

    out_large_torch = calc_torch.log_pdf(
        data_batch, [mean_matrix, chol_cov_matrices], cluster_counts_large
    )
    out_large_triton = calc_triton.log_pdf(
        data_batch, [mean_matrix, chol_cov_matrices], cluster_counts_large
    )
    torch.testing.assert_close(out_large_torch, out_large_triton, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_triton_vs_torch_edge_cases(dim):
    """Edge cases: zero vectors, one-dimensional batch, multiple clusters."""
    batch_size = 1
    clusters = 2

    data_batch = torch.zeros(batch_size, dim, device=device)
    mean_matrix = torch.zeros(clusters, dim, device=device)
    chol_cov_matrices = torch.stack(
        [torch.eye(dim, device=device) for _ in range(clusters)]
    )
    cluster_counts = torch.tensor([3.0, 7.0], device=device)

    calc_torch = FullCovarianceStudentTCalculatorTorch(device=device, data_dim=dim)
    calc_triton = FullCovarianceStudentTCalculatorTriton(device=device, data_dim=dim)

    out_torch = calc_torch.log_pdf(
        data_batch, [mean_matrix, chol_cov_matrices], cluster_counts
    )
    out_triton = calc_triton.log_pdf(
        data_batch, [mean_matrix, chol_cov_matrices], cluster_counts
    )

    torch.testing.assert_close(out_torch, out_triton, rtol=1e-5, atol=1e-6)
