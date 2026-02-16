import pytest
import torch

from dpgmm.cholesky_update.variants.cholesky_update_torch import CholeskyUpdateTorch
from dpgmm.utils.device import get_device

device = get_device()


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
def test_cholesky_update_random_batches(dim, batch_size):
    """Test cholesky_update with random batches and various dimensions"""
    chol_batch = torch.eye(dim, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    update_vectors = torch.randn(batch_size, dim, device=device)
    multiplier = torch.rand(batch_size, device=device) * 2.0  # random multipliers
    updater = CholeskyUpdateTorch(device=device)

    updated_chol = updater.cholesky_update(chol_batch, update_vectors, multiplier)

    # Check shape
    assert updated_chol.shape == chol_batch.shape
    # Check numerical stability
    assert torch.isfinite(updated_chol).all()
    # Diagonal elements should be positive (since Cholesky)
    diag = torch.diagonal(updated_chol, dim1=-2, dim2=-1)
    assert (diag > 0).all()


@pytest.mark.parametrize("dim", [1, 2, 5])
def test_cholesky_update_identity_with_zeros(dim):
    """Check update with zero update vectors does not break identity"""
    batch_size = 4
    chol_batch = torch.eye(dim, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    update_vectors = torch.zeros(batch_size, dim, device=device)
    multiplier = torch.ones(batch_size, device=device)
    updater = CholeskyUpdateTorch(device=device)

    updated_chol = updater.cholesky_update(chol_batch, update_vectors, multiplier)

    # Should remain identity-ish
    assert torch.allclose(updated_chol, chol_batch, atol=1e-6)


@pytest.mark.parametrize("dim", [3, 4, 6])
def test_cholesky_update_large_random(dim):
    """Test with large random matrices to catch numerical issues"""
    batch_size = 5
    chol_batch = torch.tril(torch.randn(batch_size, dim, dim, device=device) * 2.0)
    diag = torch.diagonal(chol_batch, dim1=-2, dim2=-1)
    # Ensure positive diagonals
    chol_batch = chol_batch + torch.diag_embed(torch.abs(diag) + 1.0)
    update_vectors = torch.randn(batch_size, dim, device=device)
    multiplier = torch.rand(batch_size, device=device) * 5.0
    updater = CholeskyUpdateTorch(device=device)

    updated_chol = updater.cholesky_update(chol_batch, update_vectors, multiplier)

    # Check shape
    assert updated_chol.shape == chol_batch.shape
    # Diagonal should remain positive
    diag_updated = torch.diagonal(updated_chol, dim1=-2, dim2=-1)
    assert (diag_updated > 0).all()
    # No NaNs/Infs
    assert torch.isfinite(updated_chol).all()


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_cholesky_update_correctness_single(dim, alpha):
    """Compare updater against naive recomputation of Cholesky factorization (single batch)."""
    # start with a random SPD matrix
    M = torch.randn(dim, dim, device=device)
    A = M @ M.T + torch.eye(dim, device=device) * 1.0
    chol = torch.linalg.cholesky(A).unsqueeze(0)  # shape (1, dim, dim)

    # pick a random update vector
    v = torch.randn(dim, device=device).unsqueeze(0)  # shape (1, dim)
    multiplier = torch.tensor([alpha], device=device)

    updater = CholeskyUpdateTorch(device=device)
    updated_chol = updater.cholesky_update(chol, v, multiplier)

    # Reference result: recompute from scratch
    ref_chol = naive_cholesky_update(chol[0], v[0], alpha)

    assert torch.allclose(updated_chol[0], ref_chol, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("batch_size", [2, 4])
def test_cholesky_update_correctness_batched(dim, batch_size):
    """Check batched updates against recomputed Cholesky."""
    M = torch.randn(batch_size, dim, dim, device=device)
    A = M @ M.transpose(-1, -2) + torch.eye(dim, device=device).unsqueeze(0)
    chol_batch = torch.linalg.cholesky(A)

    update_vectors = torch.randn(batch_size, dim, device=device)
    multiplier = torch.rand(batch_size, device=device) * 2.0

    updater = CholeskyUpdateTorch(device=device)
    updated_chol = updater.cholesky_update(chol_batch, update_vectors, multiplier)

    for b in range(batch_size):
        ref_chol = naive_cholesky_update(
            chol_batch[b], update_vectors[b], multiplier[b].item()
        )
        assert torch.allclose(updated_chol[b], ref_chol, atol=1e-5, rtol=1e-5)


def test_cholesky_update_with_fixed_data():
    """Check correctness on small fixed example with exact numbers."""
    A = torch.tensor([[2.0, 1.0], [1.0, 2.0]], device=device)
    L = torch.linalg.cholesky(A).unsqueeze(0)

    v = torch.tensor([[1.0, 2.0]], device=device)
    alpha = torch.tensor([0.5], device=device)

    updater = CholeskyUpdateTorch(device=device)
    updated_L = updater.cholesky_update(L, v, alpha)

    A_new = A + 0.5 * v[0].unsqueeze(1) @ v[0].unsqueeze(0)
    ref_L = torch.linalg.cholesky(A_new)

    assert torch.allclose(updated_L[0], ref_L, atol=1e-6), (
        f"\nExpected:\n{ref_L}\nGot:\n{updated_L[0]}"
    )
