import torch
from loguru import logger

from dpgmm.samplers.cgs.state import StudentTCalculator
from dpgmm.samplers.cgs.variants.full_cov.student_t_calculator.student_t_torch import (
    FullCovarianceStudentTCalculatorTorch,
)


class StudentTCalculatorAdapter:
    """
    An adapter class to automatically select and initialize the optimal Student-t
    calculator implementation based on the current hardware environment.
    """

    _student_t_calculator: StudentTCalculator

    @classmethod
    def _init_student_t_calculator(
        cls, device: torch.device, data_dim: int
    ) -> StudentTCalculator:
        """
        Initializes the calculator implementation.

        Prefers the fused Triton implementation if running on a CUDA-enabled device.
        Falls back to the pure PyTorch implementation if Triton fails to compile/import
        or if the device is CPU/MPS.

        Returns:
            StudentTCalculator: An initialized instance of the selected calculator.
        """
        if device.type == "cuda":
            try:
                from dpgmm.samplers.cgs.variants.full_cov.student_t_calculator.student_t_fused import (
                    FullCovarianceStudentTCalculatorTritonFused,
                )

                cls._student_t_calculator = FullCovarianceStudentTCalculatorTritonFused(
                    device=device, data_dim=data_dim
                )
                return cls._student_t_calculator
            except Exception as e:
                logger.warning(
                    f"Failed to initialize FullCovarianceStudentTCalculatorTritonFused, falling back to FullCovarianceStudentTCalculatorTorch: {e}"
                )

        cls._student_t_calculator = FullCovarianceStudentTCalculatorTorch(
            device=device, data_dim=data_dim
        )
        return cls._student_t_calculator

    @classmethod
    def get_student_t_calculator(
        cls, device: torch.device, data_dim: int
    ) -> StudentTCalculator:
        """
        Retrieves the appropriate Student-t calculator instance.

        Returns:
            StudentTCalculator: The configured calculator object.
        """
        return cls._init_student_t_calculator(device=device, data_dim=data_dim)
