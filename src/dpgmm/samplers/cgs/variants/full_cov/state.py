from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor

from dpgmm.cholesky_update.cholesky_update_adapter import CholeskyUpdateAdapter
from dpgmm.samplers.cgs.state import PriorPosteriorParametersKeeper
from dpgmm.samplers.cgs.variants.base.utils import init_kappa_0


class FullCovarianceParametersKeeper(PriorPosteriorParametersKeeper):
    """
    Manages and sequentially updates Bayesian posterior parameters for Gaussian
    mixture components with full covariance matrices.

    This class tracks the posterior means and the Cholesky factors of the
    covariance matrices. It provides methods to perform rank-1 updates and
    downdates to these parameters as data points are added to or removed
    from components.

    Attributes:
        kappa0 (Tensor): Prior strength (pseudo-observations) for the mean.
        data_dim (int): Dimensionality of the input data.
        means (Tensor): Current posterior means for all components.
        cov_chols (Tensor): Current Cholesky factors of the posterior covariances.
        mean_0 (Tensor): Prior mean parameters.
        cov_chol_0 (Tensor): Prior Cholesky factor parameters.
        n_components (int): Total number of mixture components.
        cholesky_updater: The adapter used for performing rank-1 Cholesky updates.
    """

    def __init__(
        self,
        data_dim: int,
        init_values: Dict[str, torch.Tensor],
        n_components: int,
        device: torch.device,
    ):
        """
        Initializes the state tracker with prior and initial posterior parameters.

        Args:
            data_dim (int): Dimensionality of the data.
            init_values (Dict[str, torch.Tensor]): Initial parameter values.
            n_components (int): Number of mixture components.
        """
        if "cov_chol" not in init_values:
            raise ValueError("cov_chol is not defined in init_values")
        if "mean" not in init_values:
            raise ValueError("mean is not defined in init_values")
        if "mean_0" not in init_values:
            raise ValueError("mean_0 is not defined in init_values")
        if "cov_chol_0" not in init_values:
            raise ValueError("cov_chol_0 is not defined in init_values")
        self.device = device
        self.data_dim = data_dim
        self.cov_chols = torch.as_tensor(
            init_values["cov_chol"], device=self.device, dtype=torch.float32
        )
        self.means = torch.as_tensor(
            init_values["mean"], device=self.device, dtype=torch.float32
        )

        self.kappa0: Tensor = torch.tensor(
            init_kappa_0(), dtype=torch.float32, device=self.device
        )
        self.mean_0 = torch.as_tensor(
            init_values["mean_0"], device=self.device, dtype=torch.float32
        )
        self.cov_chol_0 = torch.as_tensor(
            init_values["cov_chol_0"], device=self.device, dtype=torch.float32
        )
        self.n_components = n_components
        self.cholesky_updater = CholeskyUpdateAdapter.get_cholesky_updater(
            device=self.device
        )

    def posterior_parameter_names(self) -> List[str]:
        """Returns the list of tracked posterior parameter names."""
        return ["mean", "cov_chol"]

    def assign_posterior_params(self, new_posterior_parameters: List[Tensor]):
        """
        Updates the internal state with new posterior parameters.

        Args:
            new_posterior_parameters: A list [new_means, new_cov_chols].
        """
        [new_mean, new_cov_chol] = new_posterior_parameters
        self.means.data = new_mean
        self.cov_chols.data = new_cov_chol

    def posterior_parameters(self) -> List[Tensor]:
        """Returns the current posterior means and Cholesky factors."""
        return [self.means, self.cov_chols]

    def prior_parameters(self) -> List[Tensor]:
        """Returns the prior mean and Cholesky factor parameters."""
        return [self.mean_0, self.cov_chol_0]

    def posterior_parameters_dims(self) -> List[Union[Tuple[int], Tuple[int, int]]]:
        """Returns the shapes of the mean and covariance tensors."""
        return [(self.data_dim,), (self.data_dim, self.data_dim)]

    def downdate(
        self, data_points: Tensor, counts: Tensor, posterior_params: List[Tensor]
    ) -> List[Tensor]:
        """
        Removes the contribution of specific data points from the posterior statistics.

        Args:
            data_points: The observations to remove.
            counts: Current number of points assigned to the components.
            posterior_params: List of [means, cov_chols] to be downdated.

        Returns:
            A list containing the updated [means, cov_chols].
        """
        means, cov_chols = posterior_params
        new_means, new_cov_chols = self.downdate_(data_points, means, cov_chols, counts)
        return [new_means, new_cov_chols]

    def update(
        self, data_points: Tensor, counts: Tensor, posterior_params: List[Tensor]
    ) -> List[Tensor]:
        """
        Adds the contribution of new data points to the posterior statistics.

        Args:
            data_points: The new observations to add.
            counts: Current number of points assigned to the components.
            posterior_params: List of [means, cov_chols] to be updated.

        Returns:
            A list containing the updated [means, cov_chols].
        """
        means, cov_chols = posterior_params
        new_means, new_cov_chols = self.update_(data_points, means, cov_chols, counts)
        return [new_means, new_cov_chols]

    def downdate_(
        self, data_points: Tensor, means_: Tensor, cov_chols_: Tensor, counts_: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Internal logic to downdate means and perform rank-1 Cholesky downdates.

        Uses the formula for recursive Bayesian updates to remove a point's impact.
        """
        c = self.kappa0 + counts_
        c = c.unsqueeze(1)
        new_means = (means_ * c - data_points) / (c - 1.0)
        multipliers = torch.reshape(-c / (c - 1.0), (-1,)).to(self.device)

        new_cov_chols = self.cholesky_updater.cholesky_update(
            cov_chols_, data_points - means_, multiplier=multipliers
        )
        return new_means, new_cov_chols

    def update_(
        self, data_points: Tensor, means_: Tensor, cov_chols_: Tensor, counts_: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Internal logic to update means and perform rank-1 Cholesky updates.

        Adjusts the mean and covariance factor based on the addition of a new observation.
        """
        c = self.kappa0 + counts_
        c = c.unsqueeze(1)
        new_means = (means_ * c + data_points) / (c + 1.0)
        multipliers = torch.reshape((c + 1) / c, (-1,))

        new_cov_chols = self.cholesky_updater.cholesky_update(
            cov_chols_, data_points - new_means, multiplier=multipliers
        )
        return new_means, new_cov_chols

    def to(self, device: torch.device) -> FullCovarianceParametersKeeper:
        super().to(device)
        self.means = torch.as_tensor(self.means, device=self.device)
        self.cov_chols = torch.as_tensor(self.cov_chols, device=self.device)
        self.mean_0 = torch.as_tensor(self.mean_0, device=self.device)
        self.cov_chol_0 = torch.as_tensor(self.cov_chol_0, device=self.device)
        self.kappa0 = torch.as_tensor(self.kappa0, device=self.device)
        self.cholesky_updater = CholeskyUpdateAdapter.get_cholesky_updater(
            device=self.device
        )

        return self
