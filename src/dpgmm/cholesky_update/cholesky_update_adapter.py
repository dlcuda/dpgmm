import os
from importlib import import_module
from typing import Literal

import torch
from loguru import logger

from dpgmm.cholesky_update.cholesky_update_base import CholeskyUpdateBase
from dpgmm.cholesky_update.variants.cholesky_update_torch import CholeskyUpdateTorch

CholeskyBackend = Literal[
    "auto", "torch", "triton_full", "triton_full_barrier", "triton_full_local"
]

_TRITON_IMPLEMENTATIONS = {
    "triton_full": ("cholesky_update_triton_full", "CholeskyUpdateTritonFull"),
    "triton_full_barrier": (
        "cholesky_update_triton_full_barrier",
        "CholeskyUpdateTritonFullBarrier",
    ),
    "triton_full_local": (
        "cholesky_update_triton_full_local",
        "CholeskyUpdateTritonFullLocal",
    ),
}


class CholeskyUpdateAdapter:
    """
    Select a Cholesky updater, importing optional Triton dependencies lazily.

    Auto uses the fused program-local variant on CUDA and Torch elsewhere.
    Only auto falls back to Torch when a Triton implementation cannot initialize.
    """

    _cholesky_updater: CholeskyUpdateBase

    @classmethod
    def _init_cholesky_updater(
        cls, device: torch.device, backend: CholeskyBackend = "auto"
    ) -> CholeskyUpdateBase:
        """
        Construct the requested implementation without changing its selection
        silently. Auto retains the hardware-based fallback behavior.

        Returns:
            CholeskyUpdateBase: An initialized instance of the selected Cholesky updater.
        """
        if backend not in {"auto", "torch", *_TRITON_IMPLEMENTATIONS}:
            raise ValueError(f"Unknown Cholesky backend: {backend}")
        interpreting = (
            device.type == "cpu" and os.environ.get("TRITON_INTERPRET") == "1"
        )
        if (
            backend in _TRITON_IMPLEMENTATIONS
            and device.type != "cuda"
            and not interpreting
        ):
            raise ValueError(
                "Explicit Triton backends require CUDA or CPU with TRITON_INTERPRET=1"
            )
        if backend in _TRITON_IMPLEMENTATIONS or (
            backend == "auto" and device.type == "cuda"
        ):
            try:
                module_name, class_name = _TRITON_IMPLEMENTATIONS[
                    "triton_full_local" if backend == "auto" else backend
                ]
                module = import_module(f"dpgmm.cholesky_update.variants.{module_name}")
                cls._cholesky_updater = getattr(module, class_name)(device=device)
                return cls._cholesky_updater
            except Exception as e:
                if backend != "auto":
                    raise
                logger.warning(
                    f"Failed to initialize CholeskyUpdateTritonFullLocal, falling back to CholeskyUpdateTorch: {e}"
                )

        cls._cholesky_updater = CholeskyUpdateTorch(device=device)
        return cls._cholesky_updater

    @classmethod
    def get_cholesky_updater(
        cls, device: torch.device, backend: CholeskyBackend = "auto"
    ) -> CholeskyUpdateBase:
        """
        Selects an updater. Explicit backends raise rather than silently falling
        back. CPU Triton execution requires an installed Triton interpreter and
        TRITON_INTERPRET=1 set before importing the kernel modules.

        Returns:
            CholeskyUpdateBase: The instantiated Cholesky update object
                (either Triton-based or PyTorch-based depending on device compatibility).
        """
        return cls._init_cholesky_updater(device=device, backend=backend)
