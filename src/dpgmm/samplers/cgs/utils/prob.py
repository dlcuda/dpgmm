from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor


def multivariate_t_rvs(m: Tensor, s_chol: Tensor, df: float) -> Tensor:
    """
    Generates a single random sample from a multivariate Student-t distribution.

    Args:
        m (Tensor): Mean vector of the distribution.
        s_chol (Tensor): Lower Cholesky factor of the scale matrix.
        df (float): Degrees of freedom.

    Returns:
        Tensor: A single sampled vector from the specified Student-t distribution.
    """
    device = m.device
    d = m.size(0)
    if df == torch.inf:
        x = torch.tensor(1.0, device=device)
    else:
        x = torch.distributions.Chi2(df).sample().to(device) / df

    std_normal = torch.randn(d).to(device)
    z = s_chol @ std_normal

    return m + z / torch.sqrt(x)


def multivariate_t_rvs_full_and_cov(
    m: np.ndarray, s_chol: np.ndarray, df: float, samples_num: int
) -> Dict[str, np.ndarray]:
    """
    Generates multiple random samples from both a full-covariance and a
    diagonal-covariance multivariate Student-t distribution simultaneously.

    This function is useful for generating comparative samples using the same
    underlying standard normal noise and chi-square scaling factors.

    Args:
        m (np.ndarray): Mean vector of the distribution.
        s_chol (np.ndarray): Lower Cholesky factor of the full scale matrix.
        df (float): Degrees of freedom.
        samples_num (int): The number of samples to generate.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing two keys:
            - "sample_full": Array of samples using the full Cholesky factor.
            - "sample_diag": Array of samples using only the diagonal of the Cholesky factor.
    """
    m = np.asarray(m)
    x = np.random.chisquare(df, size=samples_num) / df

    std_normal = np.random.randn(m.shape[0], samples_num)
    s_chol_diag = np.diag(np.diag(s_chol))

    z = np.dot(s_chol, std_normal)
    z_diag = np.dot(s_chol_diag, std_normal)
    # z.shape = (d, samples_num)
    x = np.expand_dims(x, axis=0)
    m = np.expand_dims(m, axis=-1)
    result = {
        "sample_full": (m + z / np.sqrt(x)).T,
        "sample_diag": (m + z_diag / np.sqrt(x)).T,
    }

    return result


def cov_error_ellipse(
    mean: np.ndarray, cov: np.ndarray, p: float, samples_num: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the contour coordinates of a 2D covariance error ellipse.

    The ellipse represents a specific confidence region (probability mass) for
    a 2D normal distribution characterized by the given mean and covariance.

    Args:
        mean (np.ndarray): A 2-dimensional mean vector [x, y].
        cov (np.ndarray): A 2x2 covariance matrix.
        p (float): The probability mass (confidence level) the ellipse should enclose
            (e.g., 0.95 for a 95% confidence region).
        samples_num (int, optional): The number of points to generate along the
            perimeter of the ellipse. Defaults to 100.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two 1D arrays:
            - The x-coordinates of the ellipse boundary.
            - The y-coordinates of the ellipse boundary.

    Raises:
        ValueError: If the provided mean vector is not strictly 2-dimensional.
    """
    if mean.shape[0] != 2:
        raise ValueError(
            "Mean vector must be 2-dimensional for error ellipse calculation"
        )

    s = -2.0 * np.log(1.0 - p)
    vals, vecs = np.linalg.eig(cov * s)

    t = np.linspace(0, 2 * np.pi, num=samples_num)
    rs = vecs * np.expand_dims(np.sqrt(vals), axis=0)

    t_vals = np.dot(rs, [np.cos(t), np.sin(t)])

    xs = mean[0] + t_vals[0, :]
    ys = mean[1] + t_vals[1, :]

    return xs, ys
