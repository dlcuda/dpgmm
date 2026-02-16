import pytest
import torch

triton = pytest.importorskip("triton")


from dpgmm.cholesky_update.variants.cholesky_update_triton import CholeskyUpdateTriton
from dpgmm.utils.device import get_device

device = get_device()


@pytest.fixture(autouse=True)
def skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("Skipping test because no GPU is available")


def naive_cholesky_update(
    L: torch.Tensor, v: torch.Tensor, alpha: float
) -> torch.Tensor:
    """
    Reference implementation:
    Compute Cholesky of (L L^T + alpha v v^T) directly using torch.linalg.cholesky.
    """
    A = L @ L.transpose(-1, -2) + alpha * torch.ger(v, v)
    return torch.linalg.cholesky(A)


@pytest.mark.parametrize("dim", [1, 2, 5, 6])
@pytest.mark.parametrize("batch_size", [1, 3, 10])
def test_cholesky_update_triton_random_batches(dim, batch_size):
    """Test Triton cholesky_update with random batches and various dimensions"""
    chol_batch = torch.eye(dim, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    update_vectors = torch.randn(batch_size, dim, device=device)
    multiplier = torch.rand(batch_size, device=device) * 2.0
    updater = CholeskyUpdateTriton(device=device)

    updated_chol = updater.cholesky_update(chol_batch, update_vectors, multiplier)

    assert updated_chol.shape == chol_batch.shape
    assert torch.isfinite(updated_chol).all()
    diag = torch.diagonal(updated_chol, dim1=-2, dim2=-1)
    assert (diag > 0).all()


@pytest.mark.parametrize("dim", [1, 2, 5])
def test_cholesky_update_triton_identity_with_zeros(dim):
    """Check update with zero update vectors does not break identity"""
    batch_size = 4
    chol_batch = torch.eye(dim, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    update_vectors = torch.zeros(batch_size, dim, device=device)
    multiplier = torch.ones(batch_size, device=device)
    updater = CholeskyUpdateTriton(device=device)

    updated_chol = updater.cholesky_update(chol_batch, update_vectors, multiplier)

    assert torch.allclose(updated_chol, chol_batch, atol=1e-6)


@pytest.mark.parametrize("dim", [3, 4, 6])
def test_cholesky_update_triton_large_random(dim):
    """Test Triton with large random matrices to catch numerical issues"""
    batch_size = 5
    chol_batch = torch.tril(torch.randn(batch_size, dim, dim, device=device) * 2.0)
    diag = torch.diagonal(chol_batch, dim1=-2, dim2=-1)
    chol_batch = chol_batch + torch.diag_embed(torch.abs(diag) + 1.0)
    update_vectors = torch.randn(batch_size, dim, device=device)
    multiplier = torch.rand(batch_size, device=device) * 5.0
    updater = CholeskyUpdateTriton(device=device)

    updated_chol = updater.cholesky_update(chol_batch, update_vectors, multiplier)

    assert updated_chol.shape == chol_batch.shape
    diag_updated = torch.diagonal(updated_chol, dim1=-2, dim2=-1)
    assert (diag_updated > 0).all()
    assert torch.isfinite(updated_chol).all()


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_cholesky_update_triton_correctness_single(dim, alpha):
    """Compare Triton updater against naive recomputation (single batch)."""
    M = torch.randn(dim, dim, device=device)
    A = M @ M.T + torch.eye(dim, device=device)
    chol = torch.linalg.cholesky(A).unsqueeze(0)

    v = torch.randn(dim, device=device).unsqueeze(0)
    multiplier = torch.tensor([alpha], device=device)

    updater = CholeskyUpdateTriton(device=device)
    updated_chol = updater.cholesky_update(chol, v, multiplier)

    ref_chol = naive_cholesky_update(chol[0], v[0], alpha)

    assert torch.allclose(updated_chol[0], ref_chol, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("batch_size", [2, 4])
def test_cholesky_update_triton_correctness_batched(dim, batch_size):
    """Check Triton batched updates against recomputed Cholesky."""
    M = torch.randn(batch_size, dim, dim, device=device)
    A = M @ M.transpose(-1, -2) + torch.eye(dim, device=device).unsqueeze(0)
    chol_batch = torch.linalg.cholesky(A)

    update_vectors = torch.randn(batch_size, dim, device=device)
    multiplier = torch.rand(batch_size, device=device) * 2.0

    updater = CholeskyUpdateTriton(device=device)
    updated_chol = updater.cholesky_update(chol_batch, update_vectors, multiplier)

    for b in range(batch_size):
        ref_chol = naive_cholesky_update(
            chol_batch[b], update_vectors[b], multiplier[b].item()
        )
        assert torch.allclose(updated_chol[b], ref_chol, atol=1e-5, rtol=1e-5)


def test_cholesky_update_triton_with_fixed_data():
    """Check correctness on small fixed example with exact numbers."""
    A = torch.tensor([[2.0, 1.0], [1.0, 2.0]], device=device)
    L = torch.linalg.cholesky(A).unsqueeze(0)

    v = torch.tensor([[1.0, 2.0]], device=device)
    alpha = torch.tensor([0.5], device=device)

    updater = CholeskyUpdateTriton(device=device)
    updated_L = updater.cholesky_update(L, v, alpha)

    A_new = A + 0.5 * v[0].unsqueeze(1) @ v[0].unsqueeze(0)
    ref_L = torch.linalg.cholesky(A_new)

    assert torch.allclose(updated_L[0], ref_L, atol=1e-6)
