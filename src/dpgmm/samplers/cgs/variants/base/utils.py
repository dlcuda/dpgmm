def init_kappa_0() -> float:
    """
    Initializes the prior scalar precision (kappa_0).

    Returns:
        float: The initial value for kappa_0.
    """
    return 0.01


def init_nu_0(data_dim: int) -> float:
    """
    Initializes the prior degrees of freedom (nu_0) based on data dimensionality.

    Args:
        data_dim (int): The dimensionality of the input data.

    Returns:
        float: The initial value for nu_0.
    """
    return float(data_dim + 2)
