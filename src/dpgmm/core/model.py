from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import torch

from dpgmm.samplers.base import BaseSampler, BaseSamplerFitResult

SamplerType = Literal["cgs"]
CovarianceType = Literal["full", "diag"]


class DPGMM:
    def __init__(
        self,
        sampler: Optional[BaseSampler] = None,
        inference_method: SamplerType = "cgs",
        covariance_type: CovarianceType = "full",
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        self.device = device
        self.covariance_type = covariance_type
        self._fit_result: Optional[BaseSamplerFitResult] = None
        self._data: Optional[torch.Tensor] = None

        if sampler is not None:
            self.sampler = sampler
            self.sampler.to(self.device)
        else:
            self.sampler = self._build_sampler(
                inference_method, covariance_type, device=self.device, **kwargs
            )

    def _build_sampler(
        self,
        method: SamplerType,
        cov_type: CovarianceType,
        device: torch.device,
        **kwargs,
    ) -> BaseSampler:
        if method == "cgs":
            if cov_type == "full":
                from dpgmm.samplers import FullCovarianceCollapsedGibbsSampler

                return FullCovarianceCollapsedGibbsSampler(device=device, **kwargs)
            elif cov_type == "diag":
                from dpgmm.samplers import DiagCovarianceCollapsedGibbsSampler

                return DiagCovarianceCollapsedGibbsSampler(device=device, **kwargs)
        elif method == "vi":
            raise NotImplementedError("VI is not implemented yet.")

        raise ValueError(
            f"Unsupported combination: {method} with {cov_type} covariance."
        )

    def to(self, device: torch.device) -> DPGMM:
        self.device = device
        self.sampler.to(self.device)
        return self

    def fit(
        self,
        data: Union[np.ndarray, torch.Tensor],
        iterations_num: int = 100,
        out_dir: Optional[str] = None,
    ) -> DPGMM:
        self._data = torch.as_tensor(data, device=self.device)
        self._fit_result = self.sampler.fit(
            iterations_num=iterations_num, data=self._data, out_dir=out_dir
        )
        return self

    def _require_fit(self) -> BaseSamplerFitResult:
        if self._fit_result is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self._fit_result

    def cluster(self) -> np.ndarray:
        """
        Returns the MAP cluster assignment for each data point as a flat integer array.

        Returns:
            np.ndarray: Array of shape (N,) with cluster indices.
        """
        result = self._require_fit()
        return np.array(
            self.sampler.get_examples_assignment(result["cluster_assignment"])
        )

    def soft_cluster(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Returns the posterior probability of each point belonging to each cluster.

        Args:
            x: Shape (D,) or (N, D).

        Returns:
            torch.Tensor: Shape (N, K) where entry [i, k] is p(z_i = k | x_i).
        """
        result = self._require_fit()

        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(0)

        cluster_assignment = result["cluster_assignment"]
        params = result["cluster_params"]
        n_clusters = len(cluster_assignment)
        cluster_sizes = torch.tensor(
            [len(c) for c in cluster_assignment],
            dtype=torch.float32,
            device=self.device,
        )
        log_weights = torch.log(cluster_sizes / cluster_sizes.sum())  # (K,)

        if self.covariance_type == "full":
            from torch.distributions import MultivariateNormal

            log_pdfs = torch.stack(
                [
                    MultivariateNormal(
                        loc=params["mean"][k],
                        scale_tril=params["cov_chol"][k],
                    ).log_prob(x_tensor)
                    for k in range(n_clusters)
                ]
            )  # (K, N)

        elif self.covariance_type == "diag":
            from torch.distributions import Normal

            log_pdfs = torch.stack(
                [
                    Normal(loc=params["mean"][k], scale=params["var"][k].sqrt())
                    .log_prob(x_tensor)
                    .sum(dim=-1)
                    for k in range(n_clusters)
                ]
            )  # (K, N)

        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

        log_numerators = log_weights.unsqueeze(1) + log_pdfs  # (K, N)
        log_denominator = torch.logsumexp(log_numerators, dim=0, keepdim=True)  # (1, N)
        posteriors = torch.exp(log_numerators - log_denominator)  # (K, N)
        return posteriors.T  # (N, K)

    def density(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Evaluates the mixture log-density at point(s) x using the fitted cluster parameters.

        Args:
            x: A single point or batch of points, shape (D,) or (N, D).

        Returns:
            torch.Tensor: Log-density value(s) of shape (N,).
        """
        result = self._require_fit()

        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(0)

        cluster_assignment = result["cluster_assignment"]
        params = result["cluster_params"]
        n_clusters = len(cluster_assignment)
        cluster_sizes = torch.tensor(
            [len(c) for c in cluster_assignment],
            dtype=torch.float32,
            device=self.device,
        )
        log_weights = torch.log(cluster_sizes / cluster_sizes.sum())  # (K,)

        if self.covariance_type == "full":
            from torch.distributions import MultivariateNormal

            log_pdfs = torch.stack(
                [
                    MultivariateNormal(
                        loc=params["mean"][k],
                        scale_tril=params["cov_chol"][k],
                    ).log_prob(x_tensor)  # (N,)
                    for k in range(n_clusters)
                ]
            )  # (K, N)

        elif self.covariance_type == "diag":
            from torch.distributions import Normal

            log_pdfs = torch.stack(
                [
                    Normal(loc=params["mean"][k], scale=params["var"][k].sqrt())
                    .log_prob(x_tensor)  # (N, D)
                    .sum(dim=-1)  # (N,)
                    for k in range(n_clusters)
                ]
            )  # (K, N)

        else:
            raise ValueError(f"Unsupported covariance type: {self.covariance_type}")

        return torch.logsumexp(log_weights.unsqueeze(1) + log_pdfs, dim=0)  # (N,)
