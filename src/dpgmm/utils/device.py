import torch
from loguru import logger


def get_device(verbose: bool = False) -> torch.device:
    """
    Detects and returns the available PyTorch hardware device.

    Args:
        verbose (bool, optional): If True, logs the selected device type. Defaults to False.

    Returns:
        torch.device: A device object representing 'cuda' if available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if verbose:
        logger.info(f"Using device: {device.type}")
    return device
