import numpy as np
import pytest
import torch
from scipy.stats import invwishart


def sample_inverse_wishart(df: float, scale: torch.Tensor) -> torch.Tensor:
    from dpgmm.samplers.cgs.variants.full_cov.log_likelihood import (
        FullCovarianceLogLikelihood,
    )

    sampler = FullCovarianceLogLikelihood(nu_0=0.0, alpha_0=0.0)
    return sampler._sample_inverse_wishart(df, scale)


def _random_pd_matrix(d: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    a = rng.randn(d, d)
    return a @ a.T + d * np.eye(d)


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.mark.parametrize("d,df", [(3, 10.0), (5, 6.0), (5, 6.5), (8, 12.0), (2, 3.0)])
def test_output_is_positive_definite(d, df):
    scale = torch.tensor(_random_pd_matrix(d, seed=1), dtype=torch.float64)
    n_samples = 2000
    n_failures = 0
    for _ in range(n_samples):
        x = sample_inverse_wishart(df, scale)
        assert torch.allclose(x, x.T, atol=1e-8), "sample is not symmetric"
        eigvals = torch.linalg.eigvalsh(x)
        assert torch.isfinite(eigvals).all(), "sample produced nan/inf eigenvalues"
        if not torch.all(eigvals > 0):
            n_failures += 1
    assert n_failures == 0, (
        f"{n_failures}/{n_samples} samples were not positive definite"
    )


@pytest.mark.parametrize("d,df", [(3, 10.0), (4, 8.0), (8, 15.0)])
def test_matches_closed_form_mean(d, df):
    """E[InverseWishart(df, scale)] = scale / (df - d - 1), for df > d + 1."""
    scale_np = _random_pd_matrix(d, seed=2)
    scale = torch.tensor(scale_np, dtype=torch.float64)

    n_samples = 20000
    samples = torch.stack([sample_inverse_wishart(df, scale) for _ in range(n_samples)])
    empirical_mean = samples.mean(dim=0).numpy()

    theoretical_mean = scale_np / (df - d - 1)
    rel_err = np.linalg.norm(empirical_mean - theoretical_mean) / np.linalg.norm(
        theoretical_mean
    )
    assert rel_err < 0.1, f"empirical mean deviates {rel_err:.3f} from closed-form mean"


@pytest.mark.parametrize("d,df", [(3, 10.0), (5, 6.0), (8, 12.0)])
def test_matches_scipy_median(d, df):
    """
    Cross-check against scipy's independently-implemented inverse-Wishart.
    Uses median trace (robust to heavy tails near the df ~ d+1 boundary)
    rather than mean, which can be a noisy statistic in that regime for
    *either* sampler.
    """
    scale_np = _random_pd_matrix(d, seed=3)
    scale = torch.tensor(scale_np, dtype=torch.float64)

    n_samples = 20000
    ours = torch.stack(
        [sample_inverse_wishart(df, scale) for _ in range(n_samples)]
    ).numpy()
    scipy_samples = invwishart.rvs(
        df=df, scale=scale_np, size=n_samples, random_state=123
    )
    if scipy_samples.ndim == 2:  # d == 1 edge case
        scipy_samples = scipy_samples.reshape(n_samples, 1, 1)

    ours_median_trace = np.median(np.trace(ours, axis1=1, axis2=2))
    scipy_median_trace = np.median(np.trace(scipy_samples, axis1=1, axis2=2))
    rel_diff = abs(ours_median_trace - scipy_median_trace) / abs(scipy_median_trace)
    assert rel_diff < 0.1, (
        f"median trace differs from scipy by {rel_diff:.3f} "
        f"(ours={ours_median_trace:.4f}, scipy={scipy_median_trace:.4f})"
    )

    # Per-dimension eigenvalue medians should also line up.
    ours_eigs = np.median(np.sort(np.linalg.eigvalsh(ours), axis=1), axis=0)
    scipy_eigs = np.median(np.sort(np.linalg.eigvalsh(scipy_samples), axis=1), axis=0)
    eig_rel_err = np.linalg.norm(ours_eigs - scipy_eigs) / np.linalg.norm(scipy_eigs)
    assert eig_rel_err < 0.1, (
        f"eigenvalue medians differ from scipy by {eig_rel_err:.3f}"
    )


def test_low_df_boundary_is_stable():
    """
    df == d + 1 is the riskiest regime for naive (double-inversion)
    Wishart-based implementations: this is where s_k_inverse and the
    downstream sample are most likely to become ill-conditioned.
    The Bartlett approach should still produce clean, finite, PD samples.
    """
    d = 6
    df = d + 1  # exact boundary
    scale = torch.tensor(_random_pd_matrix(d, seed=4), dtype=torch.float64)

    for _ in range(500):
        x = sample_inverse_wishart(df, scale)
        assert torch.isfinite(x).all()
        eigvals = torch.linalg.eigvalsh(x)
        assert torch.all(eigvals > 0)


def test_raises_or_handles_subboundary_df_gracefully():
    """
    df < d is mathematically invalid (Chi2(df - i) needs df - i > 0 for
    all i in [0, d-1], i.e. df >= d). Calling code is expected to clamp
    nu_k to at least data_dim + 1 before reaching this function -- this
    test documents that the raw sampler itself will fail loudly (not
    silently produce garbage) if that invariant is violated upstream.
    """
    d = 5
    df_too_low = d - 1
    scale = torch.tensor(_random_pd_matrix(d, seed=5), dtype=torch.float64)

    with pytest.raises(Exception):
        # Chi2 with non-positive df is invalid; this should raise rather
        # than silently returning a nan/broken sample.
        x = sample_inverse_wishart(df_too_low, scale)
        assert torch.isfinite(x).all()  # secondary guard if no raise occurs
