import pytest
import torch

from dpgmm.samplers.cgs.variants.full_cov.state import FullCovarianceParametersKeeper
from dpgmm.utils.device import get_device

device = get_device()


def create_keeper(data_dim, n_components):
    init_values = {
        "mean": torch.zeros(n_components, data_dim, device=device),
        "cov_chol": torch.stack(
            [torch.eye(data_dim, device=device) for _ in range(n_components)]
        ),
        "mean_0": torch.zeros(data_dim, device=device),
        "cov_chol_0": torch.eye(data_dim, device=device),
    }
    return FullCovarianceParametersKeeper(
        data_dim, init_values, n_components, device=device
    )


def test_update_moves_mean_toward_data():
    keeper = create_keeper(2, 2)
    new_point = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    counts = torch.tensor([0.0, 1.0], device=device)

    new_means, new_covs = keeper.update(
        new_point, counts, [keeper.means, keeper.cov_chols]
    )
    # Mean should move closer to the new points
    assert torch.all(new_means > 0)
    assert torch.isfinite(new_covs).all()


def test_downdate_moves_mean_away_from_data():
    keeper = create_keeper(2, 2)
    removed_point = torch.tensor([[1.0, 1.0], [2.0, 2.0]], device=device)
    counts = torch.tensor([1.0, 2.0], device=device)

    new_means, new_covs = keeper.downdate(
        removed_point, counts, [keeper.means, keeper.cov_chols]
    )
    # Mean should move away from removed points
    assert torch.all(new_means <= keeper.means)
    assert torch.isfinite(new_covs).all()


def test_update_downdate_behavior():
    keeper = create_keeper(2, 1)
    point = torch.tensor([[1.0, 1.0]], device=device)
    counts = torch.tensor([1.0], device=device)

    updated_mean, updated_cov = keeper.update(
        point, counts, [keeper.means, keeper.cov_chols]
    )
    restored_mean, restored_cov = keeper.downdate(
        point, counts, [updated_mean, updated_cov]
    )

    # Means should remain finite
    assert torch.isfinite(updated_mean).all()
    assert torch.isfinite(restored_mean).all()

    # Covariances should remain finite
    assert torch.isfinite(updated_cov).all()
    assert torch.isfinite(restored_cov).all()

    # Update moves mean toward the point
    assert torch.all(updated_mean > keeper.means)

    # Downdate moves mean away from the point
    assert torch.all(restored_mean <= updated_mean)


def test_edge_case_zero_counts():
    keeper = create_keeper(2, 1)
    point = torch.tensor([[1.0, 1.0]], device=device)
    counts = torch.tensor([0.0], device=device)

    new_means, new_covs = keeper.update(point, counts, [keeper.means, keeper.cov_chols])
    assert torch.isfinite(new_means).all()
    assert torch.isfinite(new_covs).all()


def test_edge_case_high_counts():
    keeper = create_keeper(2, 1)
    point = torch.tensor([[1.0, 1.0]], device=device)
    counts = torch.tensor([1e6], device=device)

    new_means, new_covs = keeper.update(point, counts, [keeper.means, keeper.cov_chols])
    assert torch.isfinite(new_means).all()
    assert torch.isfinite(new_covs).all()


@pytest.mark.parametrize("data_dim,n_components", [(1, 1), (2, 2), (3, 2), (5, 3)])
def test_multiple_dimensions(data_dim, n_components):
    keeper = create_keeper(data_dim, n_components)

    point = torch.randn(n_components, data_dim, device=device)
    counts = torch.ones(n_components, device=device)

    new_means, new_covs = keeper.update(point, counts, [keeper.means, keeper.cov_chols])
    assert new_means.shape == (n_components, data_dim)
    assert new_covs.shape == (n_components, data_dim, data_dim)
    assert torch.isfinite(new_covs).all()

    restored_means, restored_covs = keeper.downdate(
        point, counts, [new_means, new_covs]
    )
    assert restored_means.shape == (n_components, data_dim)
    assert restored_covs.shape == (n_components, data_dim, data_dim)
    assert torch.isfinite(restored_covs).all()
