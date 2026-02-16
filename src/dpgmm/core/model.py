from typing import Literal, Optional, Union

import numpy as np
import torch

from dpgmm.samplers import BaseSampler

SamplerType = Literal["cgs"]
CovarianceType = Literal["full", "diag"]


class DPGMM:
    def __init__(
        self,
        sampler: Optional[BaseSampler] = None,
        inference_method: SamplerType = "cgs",
        covariance_type: CovarianceType = "full",
        **kwargs,
    ):
        if sampler is not None:
            self.sampler = sampler
        else:
            self.sampler = self._build_sampler(
                inference_method, covariance_type, **kwargs
            )

    def _build_sampler(
        self, method: SamplerType, cov_type: CovarianceType, **kwargs
    ) -> BaseSampler:
        if method == "cgs":
            if cov_type == "full":
                from dpgmm.samplers import FullCovarianceCollapsedGibbsSampler

                return FullCovarianceCollapsedGibbsSampler(**kwargs)
            elif cov_type == "diag":
                from dpgmm.samplers import DiagCovarianceCollapsedGibbsSampler

                return DiagCovarianceCollapsedGibbsSampler(**kwargs)

        elif method == "vi":
            if cov_type == "full":
                raise NotImplementedError(
                    "VI with full covariance is not implemented yet."
                )

        raise ValueError(
            f"Unsupported combination: {method} with {cov_type} covariance."
        )

    def to(self, device: Union[str, torch.device]):
        """Moves the model and its internal state to the specified device."""
        self.device = torch.device(device)
        self.sampler.to(self.device)

        return self

    def fit(
        self,
        data: Union[np.ndarray, torch.Tensor],
        iterations_num: int = 100,
        out_dir: Optional[str] = None,
    ):
        data_tensor = torch.as_tensor(data, device=self.device)

        return self.sampler.fit(
            iterations_num=iterations_num, data=data_tensor, out_dir=out_dir
        )
