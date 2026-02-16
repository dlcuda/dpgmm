import pytest
import torch

from dpgmm.samplers.cgs.variants.diag_cov.state import DiagCovarianceParametersKeeper
from dpgmm.utils.device import get_device

device = get_device()


def create_diag_keeper(data_dim, n_components):
    init_values = {
        "mean": torch.zeros(n_components, data_dim, device=device),
        "var": torch.ones(n_components, data_dim, device=device),
        "mean_0": torch.zeros(data_dim, device=device),
        "var_0": torch.ones(data_dim, device=device),
    }
    return DiagCovarianceParametersKeeper(
        data_dim, init_values, n_components, device=device
    )


def test_update_moves_mean_toward_data_diag():
    keeper = create_diag_keeper(2, 2)
    new_point = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    counts = torch.tensor([0.0, 1.0], device=device)

    new_means, new_vars = keeper.update(new_point, counts, [keeper.means, keeper.vars])
    assert torch.all(new_means >= keeper.means)
    assert torch.isfinite(new_vars).all()


def test_downdate_moves_mean_away_from_data_diag():
    keeper = create_diag_keeper(2, 2)
    removed_point = torch.tensor([[1.0, 1.0], [2.0, 2.0]], device=device)
    counts = torch.tensor([1.0, 2.0], device=device)

    new_means, new_vars = keeper.downdate(
        removed_point, counts, [keeper.means, keeper.vars]
    )
    assert torch.all(new_means <= keeper.means)
    assert torch.isfinite(new_vars).all()


def test_update_downdate_round_trip_diag():
    keeper = create_diag_keeper(2, 1)
    point = torch.tensor([[1.0, 1.0]], device=device)
    counts = torch.tensor([1.0], device=device)

    updated_mean, updated_var = keeper.update(
        point, counts, [keeper.means, keeper.vars]
    )
    restored_mean, restored_var = keeper.downdate(
        point, counts, [updated_mean, updated_var]
    )

    assert torch.isfinite(updated_mean).all()
    assert torch.isfinite(restored_mean).all()
    assert torch.isfinite(updated_var).all()
    assert torch.isfinite(restored_var).all()
    assert torch.all(updated_mean > keeper.means)
    assert torch.all(restored_mean <= updated_mean)


def test_edge_case_zero_counts_diag():
    keeper = create_diag_keeper(2, 1)
    point = torch.tensor([[1.0, 1.0]], device=device)
    counts = torch.tensor([0.0], device=device)

    new_means, new_vars = keeper.update(point, counts, [keeper.means, keeper.vars])
    assert torch.isfinite(new_means).all()
    assert torch.isfinite(new_vars).all()


def test_edge_case_high_counts_diag():
    keeper = create_diag_keeper(2, 1)
    point = torch.tensor([[1.0, 1.0]], device=device)
    counts = torch.tensor([1e6], device=device)

    new_means, new_vars = keeper.update(point, counts, [keeper.means, keeper.vars])
    assert torch.isfinite(new_means).all()
    assert torch.isfinite(new_vars).all()


@pytest.mark.parametrize("data_dim,n_components", [(1, 1), (2, 2), (3, 2), (5, 3)])
def test_multiple_dimensions_diag(data_dim, n_components):
    keeper = create_diag_keeper(data_dim, n_components)

    point = torch.randn(n_components, data_dim, device=device)
    counts = torch.ones(n_components, device=device)

    new_means, new_vars = keeper.update(point, counts, [keeper.means, keeper.vars])
    assert new_means.shape == (n_components, data_dim)
    assert new_vars.shape == (n_components, data_dim)
    assert torch.isfinite(new_vars).all()

    restored_means, restored_vars = keeper.downdate(
        point, counts, [new_means, new_vars]
    )
    assert restored_means.shape == (n_components, data_dim)
    assert restored_vars.shape == (n_components, data_dim)
    assert torch.isfinite(restored_vars).all()


def test_numerical_stability_small_counts_diag():
    keeper = create_diag_keeper(3, 1)
    point = torch.randn(1, 3, device=device)
    counts = torch.tensor([1e-8], device=device)

    new_means, new_vars = keeper.update(point, counts, [keeper.means, keeper.vars])
    assert torch.isfinite(new_means).all()
    assert torch.isfinite(new_vars).all()
