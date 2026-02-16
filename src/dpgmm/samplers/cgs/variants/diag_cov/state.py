from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor

from dpgmm.samplers.cgs.state import PriorPosteriorParametersKeeper
from dpgmm.samplers.cgs.variants.base.utils import init_kappa_0, init_nu_0


class DiagCovarianceParametersKeeper(PriorPosteriorParametersKeeper):
    """
    Manages and sequentially updates Bayesian posterior parameters (mean and variance vectors)
    for diagonal covariance mixture components.
    """

    def __init__(
        self,
        data_dim: int,
        init_values: Dict[str, Tensor],
        n_components: int,
        device: torch.device,
    ):
        """
        Initializes the state tracker with prior and initial posterior parameters.

        Args:
            data_dim (int): Dimensionality of the data.
            init_values (Dict[str, Tensor]): Initial parameter values containing
                'mean', 'var', 'mean_0', and 'var_0'.
            n_components (int): Initial number of mixture components.

        Raises:
            ValueError: If any required parameter keys are missing from `init_values`.
        """
        if "var" not in init_values:
            raise ValueError("var is not defined in init_values")
        if "mean" not in init_values:
            raise ValueError("mean is not defined in init_values")
        if "mean_0" not in init_values:
            raise ValueError("mean_0 is not defined in init_values")
        if "var_0" not in init_values:
            raise ValueError("var_0 is not defined in init_values")

        self.device = device
        self.data_dim = data_dim
        self.vars = torch.as_tensor(
            init_values["var"], dtype=torch.float32, device=device
        )
        self.means = torch.as_tensor(
            init_values["mean"], dtype=torch.float32, device=device
        )

        self.kappa0 = torch.tensor(init_kappa_0(), dtype=torch.float32, device=device)
        self.mean_0 = torch.as_tensor(
            init_values["mean_0"], dtype=torch.float32, device=device
        )
        self.var_0 = torch.as_tensor(
            init_values["var_0"], dtype=torch.float32, device=device
        )
        self.n_components = n_components
        self.nu0 = torch.tensor(init_nu_0(data_dim), dtype=torch.float32, device=device)

    def posterior_parameter_names(self) -> List[str]:
        """
        Retrieves the names of the tracked posterior parameters.

        Returns:
            List[str]: A list containing the strings 'mean' and 'var'.
        """
        return ["mean", "var"]

    def assign_posterior_params(self, new_posterior_parameters: List[Tensor]):
        """
        Overwrites the current posterior parameters with new values.

        Args:
            new_posterior_parameters (List[Tensor]): A list containing the new mean
                tensor and the new variance tensor.
        """
        [new_mean, new_var] = new_posterior_parameters
        self.means.data = new_mean
        self.vars.data = new_var

    def posterior_parameters(self) -> List[Tensor]:
        """
        Retrieves the current posterior parameters.

        Returns:
            List[Tensor]: A list containing the current mean and variance tensors.
        """
        return [self.means, self.vars]

    def prior_parameters(self) -> List[Tensor]:
        """
        Retrieves the initial base prior parameters.

        Returns:
            List[Tensor]: A list containing the prior mean and prior variance tensors.
        """
        return [self.mean_0, self.var_0]

    def posterior_parameters_dims(self) -> List[Union[Tuple[int], Tuple[int, int]]]:
        """
        Returns the expected dimensions of the posterior parameters.

        Returns:
            List[Union[Tuple[int], Tuple[int, int]]]: A list of shape tuples for the
                mean (D,) and variance (D,).
        """
        return [(self.data_dim,), (self.data_dim,)]

    def downdate(
        self, data_points: Tensor, counts: Tensor, posterior_params: List[Tensor]
    ) -> List[Tensor]:
        """
        Interface method to compute new parameters when data points are removed.

        Args:
            data_points (Tensor): The batch of data points being removed.
            counts (Tensor): The current point counts for the affected clusters.
            posterior_params (List[Tensor]): The current mean and variance tensors.

        Returns:
            List[Tensor]: The newly downdated mean and variance tensors.
        """
        means, vars_ = posterior_params
        new_means, new_vars = self.downdate_(data_points, means, vars_, counts)
        return [new_means, new_vars]

    def update(
        self, data_points: Tensor, counts: Tensor, posterior_params: List[Tensor]
    ) -> List[Tensor]:
        """
        Interface method to compute new parameters when data points are added.

        Args:
            data_points (Tensor): The batch of data points being added.
            counts (Tensor): The current point counts for the affected clusters.
            posterior_params (List[Tensor]): The current mean and variance tensors.

        Returns:
            List[Tensor]: The newly updated mean and variance tensors.
        """
        means, vars_ = posterior_params
        new_means, new_vars = self.update_(data_points, means, vars_, counts)
        return [new_means, new_vars]

    def downdate_(
        self, data_points: Tensor, means_: Tensor, vars_: Tensor, counts_: Tensor
    ):
        """
        Mathematically downdates the mean and variance to reflect removed data points.

        Args:
            data_points (Tensor): Data points being removed.
            means_ (Tensor): Current cluster means.
            vars_ (Tensor): Current cluster variances.
            counts_ (Tensor): Current point counts per cluster.

        Returns:
            Tuple[Tensor, Tensor]: The downdated means and variances.
        """
        kappa_n = (self.kappa0 + counts_).unsqueeze(1)
        nu_n = (self.nu0 + counts_).unsqueeze(1)

        new_means = (means_ * kappa_n - data_points) / (kappa_n - 1.0)
        nu_vars_diff = (
            data_points * data_points
            + (kappa_n - 1.0) * new_means * new_means
            - kappa_n * means_ * means_
        )
        new_nu_times_vars = nu_n * vars_ - nu_vars_diff
        new_vars = new_nu_times_vars / (nu_n - 1.0)

        return new_means, new_vars

    def update_(
        self, data_points: Tensor, means_: Tensor, vars_: Tensor, counts_: Tensor
    ):
        """
        Mathematically updates the mean and variance to reflect added data points.

        Args:
            data_points (Tensor): Data points being added.
            means_ (Tensor): Current cluster means.
            vars_ (Tensor): Current cluster variances.
            counts_ (Tensor): Current point counts per cluster.

        Returns:
            Tuple[Tensor, Tensor]: The updated means and variances.
        """
        kappa_n = (self.kappa0 + counts_).unsqueeze(1)
        nu_n = (self.nu0 + counts_).unsqueeze(1)

        new_means = (means_ * kappa_n + data_points) / (kappa_n + 1.0)
        nu_vars_diff = (
            data_points * data_points
            + kappa_n * means_ * means_
            - (kappa_n + 1.0) * new_means * new_means
        )
        new_nu_times_vars = nu_vars_diff + nu_n * vars_
        new_vars = new_nu_times_vars / (nu_n + 1.0)

        return new_means, new_vars

    def to(self, device: torch.device) -> DiagCovarianceParametersKeeper:
        super().to(device)
        self.means = torch.as_tensor(self.means, device=self.device)
        self.vars = torch.as_tensor(self.vars, device=self.device)
        self.mean_0 = torch.as_tensor(self.mean_0, device=self.device)
        self.var_0 = torch.as_tensor(self.var_0, device=self.device)
        self.kappa0 = torch.as_tensor(self.kappa0, device=self.device)
        self.nu0 = torch.as_tensor(self.nu0, device=self.device)

        return self
