import numpy as np
import torch

from dpgmm.samplers.cgs.utils import prob as prob_utils
from dpgmm.utils.device import get_device

device = get_device()


def test_multivariate_t_rvs_shape_and_finite():
    m = torch.zeros(3, device=device)
    s_chol = torch.eye(3, device=device)
    df = 5.0

    sample = prob_utils.multivariate_t_rvs(m, s_chol, df)
    assert sample.shape == m.shape
    assert torch.isfinite(sample).all()


def test_multivariate_t_rvs_df_inf():
    m = torch.tensor([1.0, 2.0], device=device)
    s_chol = torch.eye(2, device=device)
    df = float("inf")

    sample = prob_utils.multivariate_t_rvs(m, s_chol, df)
    # With df=inf, behaves like normal
    assert sample.shape == m.shape
    assert torch.isfinite(sample).all()


def test_multivariate_t_rvs_matches_theoretical_values():
    """Check sampled mean and covariance approximate theoretical values"""
    np.random.seed(123)
    torch.manual_seed(123)

    df = 5
    mean = torch.tensor([1.0, -1.0], device=device)
    cov = torch.tensor([[2.0, 0.5], [0.5, 1.0]], device=device)
    chol = torch.linalg.cholesky(cov)

    samples = torch.stack(
        [prob_utils.multivariate_t_rvs(mean, chol, df) for _ in range(100000)]
    )
    samples_np = samples.cpu().numpy()

    # Theoretical mean and covariance
    # For df > 1, mean exists; for df > 2, covariance exists
    theoretical_mean = np.array([1.0, -1.0])
    theoretical_cov = (
        cov.cpu().numpy() * df / (df - 2)
    )  # scale factor for multivariate t

    sample_mean = samples_np.mean(axis=0)
    sample_cov = np.cov(samples_np, rowvar=False)

    # Allow some tolerance due to sampling noise
    np.testing.assert_allclose(sample_mean, theoretical_mean, rtol=0.05, atol=1e-2)
    np.testing.assert_allclose(sample_cov, theoretical_cov, rtol=0.05, atol=1e-2)


def test_multivariate_t_rvs_full_and_cov_shapes():
    m = np.zeros(3)
    s_chol = np.eye(3)
    df = 5.0
    samples_num = 10

    result = prob_utils.multivariate_t_rvs_full_and_cov(m, s_chol, df, samples_num)
    sample_full = result["sample_full"]
    sample_diag = result["sample_diag"]

    assert sample_full.shape == (samples_num, 3)
    assert sample_diag.shape == (samples_num, 3)
    assert np.all(np.isfinite(sample_full))
    assert np.all(np.isfinite(sample_diag))


def test_cov_error_ellipse_shape_and_finite():
    mean = np.zeros(2)
    cov = np.eye(2)
    p = 0.9
    samples_num = 50

    xs, ys = prob_utils.cov_error_ellipse(mean, cov, p, samples_num)
    assert xs.shape[0] == samples_num
    assert ys.shape[0] == samples_num
    assert np.all(np.isfinite(xs))
    assert np.all(np.isfinite(ys))


def test_cov_error_ellipse_scaling():
    mean = np.array([1.0, 2.0])
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    p = 0.95
    samples_num = 100

    xs, ys = prob_utils.cov_error_ellipse(mean, cov, p, samples_num)
    # Sanity check bounds: ellipse roughly surrounds mean
    assert np.all(xs >= mean[0] - 5)
    assert np.all(xs <= mean[0] + 5)
    assert np.all(ys >= mean[1] - 5)
    assert np.all(ys <= mean[1] + 5)


def test_multivariate_t_rvs_full_and_cov_randomness():
    m = np.zeros(2)
    s_chol = np.eye(2)
    df = 3
    samples_num = 5

    np.random.seed(123)
    result1 = prob_utils.multivariate_t_rvs_full_and_cov(m, s_chol, df, samples_num)
    np.random.seed(123)
    result2 = prob_utils.multivariate_t_rvs_full_and_cov(m, s_chol, df, samples_num)

    assert np.allclose(result1["sample_full"], result2["sample_full"])
    assert np.allclose(result1["sample_diag"], result2["sample_diag"])
