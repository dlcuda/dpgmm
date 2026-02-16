import numpy as np
import pytest
import torch
from scipy.stats import multivariate_t

from dpgmm.utils.distributions.multivariate_student_t import MultivariateStudentT

torch.manual_seed(42)


def test_sample_shape():
    df = torch.tensor([5.0, 10.0])
    loc = torch.zeros(2, 3)
    scale_tril = torch.eye(3).repeat(2, 1, 1)

    dist = MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril)
    samples = dist.sample((100,))
    assert samples.shape == (100, 2, 3), (
        f"Expected shape (100,2,3), got {samples.shape}"
    )


def test_rsample_shape():
    df = torch.tensor([5.0, 10.0])
    loc = torch.zeros(2, 3)
    scale_tril = torch.eye(3).repeat(2, 1, 1)

    dist = MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril)
    samples = dist.rsample((100,))
    assert samples.shape == (100, 2, 3), (
        f"Expected shape (100,2,3), got {samples.shape}"
    )


def test_log_prob_finite():
    df = torch.tensor([5.0])
    loc = torch.zeros(1, 2)
    scale_tril = torch.eye(2).repeat(1, 1, 1)

    dist = MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril)
    samples = dist.sample((10,))
    log_probs = dist.log_prob(samples)
    assert torch.isfinite(log_probs).all(), "log_prob returned non-finite values"


def test_mean_approximation():
    df = torch.tensor([10.0])
    loc = torch.zeros(1, 2)
    scale_tril = torch.eye(2).repeat(1, 1, 1)

    dist = MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril)
    samples = dist.sample((10000,))
    mean_estimate = samples.mean(dim=0)
    assert torch.allclose(mean_estimate, loc, atol=0.1), (
        f"Estimated mean {mean_estimate} differs from loc {loc}"
    )


def test_invalid_loc_dimension():
    df = torch.tensor([5.0])
    loc = torch.tensor(0.0)
    scale_tril = torch.eye(2)

    with pytest.raises(ValueError):
        MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril)


def test_invalid_scale_tril_dimension():
    df = torch.tensor([5.0])
    loc = torch.zeros(2)
    scale_tril = torch.tensor([1.0, 0.0])

    with pytest.raises(ValueError):
        MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril)


def test_log_prob_known_values():
    df = torch.tensor([3.0])
    loc = torch.tensor([[0.0, 0.0]])
    scale_tril = torch.eye(2).unsqueeze(0)
    dist = MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril)

    x_close = torch.tensor([[0.1, -0.1]])
    x_far = torch.tensor([[2.0, 2.0]])

    logp_close = dist.log_prob(x_close)
    logp_far = dist.log_prob(x_far)

    assert logp_close > logp_far, "Closer point should have higher log_prob"


def test_rsample_scaling():
    df = torch.tensor([5.0])
    loc = torch.tensor([[0.0, 0.0]])
    scale_tril = torch.eye(2).unsqueeze(0)

    dist = MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril)
    samples = dist.rsample((10000,))

    samples_reshaped = samples.reshape(-1, 2)
    cov_est = torch.cov(samples_reshaped.T)

    expected_cov = 5.0 / 3.0 * torch.eye(2)
    assert torch.allclose(cov_est, expected_cov, atol=0.1), (
        f"Covariance mismatch: {cov_est} vs {expected_cov}"
    )


def test_mahalanobis_distance():
    df = torch.tensor([4.0])
    loc = torch.tensor([[1.0, 2.0]])
    scale_tril = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]])

    dist = MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril)
    x = torch.tensor([[3.0, 3.0]])

    inv_scale = torch.inverse(scale_tril[0])
    diff = x - loc
    mah_sq_manual = (diff @ inv_scale.T) ** 2
    mah_sq_manual = mah_sq_manual.sum(dim=-1)

    logp = dist.log_student_t_pdf(x)
    logp_far = dist.log_student_t_pdf(torch.tensor([[10.0, 10.0]]))
    assert logp_far < logp, "Farther point should have smaller log_prob"
    assert mah_sq_manual.item() > 0.0


def test_log_prob_against_scipy():
    df = torch.tensor([5.0])
    loc = torch.tensor([[1.0, 2.0]])
    scale_tril = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]])

    dist = MultivariateStudentT(df=df, loc=loc, scale_tril=scale_tril)

    xs = torch.tensor([[1.0, 2.0], [2.0, 3.0], [0.0, 0.0]])

    logp_torch = dist.log_prob(xs).detach().numpy().flatten()

    cov = (scale_tril[0] @ scale_tril[0].T).numpy()
    loc_np = loc[0].numpy()
    df_np = df.item()

    logp_scipy = np.array(
        [multivariate_t.logpdf(x, loc=loc_np, shape=cov, df=df_np) for x in xs]
    )

    np.testing.assert_allclose(logp_torch, logp_scipy, rtol=1e-5, atol=1e-5)
