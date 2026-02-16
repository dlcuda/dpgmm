import torch
from loguru import logger

from dpgmm.cholesky_update.cholesky_update_base import CholeskyUpdateBase
from dpgmm.cholesky_update.variants.cholesky_update_torch import CholeskyUpdateTorch


class CholeskyUpdateAdapter:
    """
    An adapter class to manage the selection and initialization of a Cholesky updater.

    This class provides a factory interface to retrieve the optimal Cholesky update
    implementation for the current hardware. It prefers a highly optimized Triton
    implementation if CUDA is available, gracefully falling back to a standard PyTorch
    implementation if Triton fails to initialize or no GPU is present.
    """

    _cholesky_updater: CholeskyUpdateBase

    @classmethod
    def _init_cholesky_updater(cls, device: torch.device) -> CholeskyUpdateBase:
        """
        Initializes the appropriate Cholesky update implementation based on the device.

        If the current device is CUDA-enabled, it attempts to dynamically import and
        instantiate `CholeskyUpdateTritonFull`. If this fails (e.g., due to missing
        dependencies or compilation errors), it logs a warning and falls back to
        `CholeskyUpdateTorch`. For non-CUDA devices, it strictly defaults to the
        PyTorch implementation.

        Returns:
            CholeskyUpdateBase: An initialized instance of the selected Cholesky updater.
        """
        if device.type == "cuda":
            try:
                from dpgmm.cholesky_update.variants.cholesky_update_triton_full import (
                    CholeskyUpdateTritonFull,
                )

                cls._cholesky_updater = CholeskyUpdateTritonFull(device=device)
                return cls._cholesky_updater
            except Exception as e:
                logger.warning(
                    f"Failed to initialize CholeskyUpdateTritonFull, falling back to CholeskyUpdateTorch: {e}"
                )

        cls._cholesky_updater = CholeskyUpdateTorch(device=device)
        return cls._cholesky_updater

    @classmethod
    def get_cholesky_updater(cls, device: torch.device) -> CholeskyUpdateBase:
        """
        Retrieves an instance of the optimal Cholesky updater for the current hardware.

        Returns:
            CholeskyUpdateBase: The instantiated Cholesky update object
                (either Triton-based or PyTorch-based depending on device compatibility).
        """
        return cls._init_cholesky_updater(device=device)
