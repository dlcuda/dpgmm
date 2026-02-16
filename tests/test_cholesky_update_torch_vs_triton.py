import pytest
import torch

triton = pytest.importorskip("triton")


from dpgmm.cholesky_update.variants.cholesky_update_torch import CholeskyUpdateTorch
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
    """Reference computation: L_new = cholesky(L L^T + alpha v v^T)"""
    A = L @ L.transpose(-1, -2) + alpha * torch.ger(v, v)
    return torch.linalg.cholesky(A)


@pytest.mark.parametrize("dim", [2, 3, 5])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_cholesky_update_torch_vs_triton_random(dim, batch_size):
    """Random batches: compare Torch vs Triton implementation."""
    M = torch.randn(batch_size, dim, dim, device=device)
    A = M @ M.transpose(-1, -2) + torch.eye(dim, device=device).unsqueeze(0)
    chol_batch = torch.linalg.cholesky(A)

    update_vectors = torch.randn(batch_size, dim, device=device)
    multiplier = torch.rand(batch_size, device=device) * 2.0

    updater_torch = CholeskyUpdateTorch(device=device)
    updater_triton = CholeskyUpdateTriton(device=device)

    chol_torch = updater_torch.cholesky_update(chol_batch, update_vectors, multiplier)
    chol_triton = updater_triton.cholesky_update(chol_batch, update_vectors, multiplier)

    assert torch.allclose(chol_torch, chol_triton, atol=1e-5, rtol=1e-5), (
        f"Torch vs Triton mismatch:\nTorch:\n{chol_torch}\nTriton:\n{chol_triton}"
    )


def test_cholesky_update_torch_vs_triton_fixed():
    """Fixed 2x2 example: verify Torch vs Triton exactly match."""
    A = torch.tensor([[2.0, 1.0], [1.0, 2.0]], device=device)
    L = torch.linalg.cholesky(A).unsqueeze(0)

    v = torch.tensor([[1.0, 2.0]], device=device)
    alpha = torch.tensor([0.5], device=device)

    updater_torch = CholeskyUpdateTorch(device=device)
    updater_triton = CholeskyUpdateTriton(device=device)

    L_torch = updater_torch.cholesky_update(L, v, alpha)
    L_triton = updater_triton.cholesky_update(L, v, alpha)

    assert torch.allclose(L_torch, L_triton, atol=1e-6), (
        f"Torch vs Triton mismatch:\nTorch:\n{L_torch}\nTriton:\n{L_triton}"
    )


@pytest.mark.parametrize("dim", [3, 5])
@pytest.mark.parametrize("batch_size", [2, 4])
def test_cholesky_update_torch_vs_triton_batched(dim, batch_size):
    """Batched matrices with random updates: compare Torch vs Triton."""
    M = torch.randn(batch_size, dim, dim, device=device)
    A = M @ M.transpose(-1, -2) + torch.eye(dim, device=device).unsqueeze(0)
    chol_batch = torch.linalg.cholesky(A)

    update_vectors = torch.randn(batch_size, dim, device=device)
    multiplier = torch.rand(batch_size, device=device) * 2.0

    updater_torch = CholeskyUpdateTorch(device=device)
    updater_triton = CholeskyUpdateTriton(device=device)

    chol_torch = updater_torch.cholesky_update(chol_batch, update_vectors, multiplier)
    chol_triton = updater_triton.cholesky_update(chol_batch, update_vectors, multiplier)

    for b in range(batch_size):
        assert torch.allclose(chol_torch[b], chol_triton[b], atol=1e-5, rtol=1e-5), (
            f"Batch {b} mismatch\nTorch:\n{chol_torch[b]}\nTriton:\n{chol_triton[b]}"
        )
