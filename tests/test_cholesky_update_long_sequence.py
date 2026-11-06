"""Long downdate/update sequences must not drift.

Single-update tests cannot see errors that only matter when they compound:
a cross-warp race that fires occasionally, or approximate sqrt/div whose
fraction-of-an-ulp bias accumulates over the ~1e5 rank-1 updates a large
cluster receives per epoch. Here every point of a cluster is downdated and
added back, which leaves the exact factor unchanged, so any drift is error.
"""

import numpy as np
import pytest
import torch

from dpgmm.cholesky_update.cholesky_update_adapter import CholeskyUpdateAdapter

KAPPA_0 = 0.01
N_POINTS = 2000
DIM = 64  # > 32 so that rows are split across warps in the fused kernels
EPOCHS = 2
# IEEE float32 (torch, CPU) gives |log-det drift| ~3e-8 and row error ~1.3e-5.
# A mean sqrt bias of 0.05 ulp gives ~-3.4e-3, 0.15 ulp ~-1.0e-2.
MAX_ABS_LOGDET_DRIFT = 1e-3
MAX_REL_ROW_ERROR = 2e-4


def _cases():
    yield pytest.param("torch", "cpu", id="torch-cpu")
    for backend in ("torch", "triton_full_local", "triton_full_barrier"):
        yield pytest.param(
            backend,
            "cuda",
            id=f"{backend}-cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
        )


@pytest.mark.parametrize("backend,device_name", list(_cases()))
def test_down_up_cycles_do_not_drift(backend, device_name):
    device = torch.device(device_name)
    rng = np.random.default_rng(0)
    mixing = rng.normal(size=(DIM, DIM)) / np.sqrt(DIM)
    points = rng.normal(size=(N_POINTS, DIM)) @ mixing + 0.3
    points = points.astype(np.float32).astype(np.float64)

    kappa_n = KAPPA_0 + N_POINTS
    mean = points.sum(0) / kappa_n
    scatter = np.eye(DIM) + points.T @ points - kappa_n * np.outer(mean, mean)
    exact = np.linalg.cholesky(scatter)

    updater = CholeskyUpdateAdapter.get_cholesky_updater(device, backend=backend)
    L = torch.tensor(exact, dtype=torch.float32, device=device)[None].contiguous()
    m = torch.tensor(mean, dtype=torch.float32, device=device)[None]
    X = torch.tensor(points, dtype=torch.float32, device=device)
    c = torch.tensor([[kappa_n]], dtype=torch.float32, device=device)
    down = (-c / (c - 1.0)).reshape(-1)
    up = (c / (c - 1.0)).reshape(-1)

    for _ in range(EPOCHS):
        for j in rng.permutation(N_POINTS):
            x = X[j : j + 1]
            new_m = (m * c - x) / (c - 1.0)
            L = updater.cholesky_update(L, x - m, down)
            m = new_m
            new_m = (m * (c - 1.0) + x) / c
            L = updater.cholesky_update(L, x - new_m, up)
            m = new_m

    result = L[0].double().cpu().numpy()
    logdet_drift = 2 * (np.log(np.diag(result)).sum() - np.log(np.diag(exact)).sum())
    rows = np.linalg.norm(result - exact, axis=1) / np.linalg.norm(exact, axis=1)

    assert abs(logdet_drift) < MAX_ABS_LOGDET_DRIFT, f"log-det drift {logdet_drift:.2e}"
    assert rows.max() < MAX_REL_ROW_ERROR, (
        f"row error max {rows.max():.2e} (rows<32 {rows[:32].mean():.2e}, rows>=32 {rows[32:].mean():.2e})"
    )
