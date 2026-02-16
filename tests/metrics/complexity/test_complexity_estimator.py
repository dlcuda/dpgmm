from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.distributions as dist

from dpgmm.metrics.complexity import ComplexityEstimator


@pytest.fixture
def mock_data():
    torch.manual_seed(42)
    return torch.randn(100, 2)


@pytest.fixture
def mock_mixture_distr():
    mean = torch.zeros(2)
    std = torch.ones(2)
    return dist.Independent(dist.Normal(mean, std), 1)


@pytest.fixture
def estimator(mock_mixture_distr, mock_data):
    return ComplexityEstimator(
        mixture_distr=mock_mixture_distr, data=mock_data, clusters_num=3, samples_num=50
    )


def test_estimate_entropy_with_sampling(estimator):
    with pytest.MonkeyPatch().context() as m:
        mock_sample = MagicMock(return_value=torch.randn(50, 2))
        m.setattr(estimator.t_student_mixture, "sample", mock_sample)

        entropy = estimator.estimate_entropy_with_sampling()

        assert isinstance(entropy, float)
        mock_sample.assert_called_once()
        assert mock_sample.call_args[0][0] == (50,)


def test_estimate_entropy_on_data_with_tensor(estimator, mock_data):
    entropy = estimator.estimate_entropy_on_data(mock_data)
    assert isinstance(entropy, float)
    assert np.isfinite(entropy)


def test_estimate_entropy_on_data_with_numpy(estimator):
    data_np = np.random.randn(20, 2).astype(np.float32)
    entropy = estimator.estimate_entropy_on_data(data_np)

    assert isinstance(entropy, float)
    assert np.isfinite(entropy)


def test_internal_estimate_entropy_logic(estimator):
    test_sample = torch.zeros(1, 2)

    log_prob_model = estimator.t_student_mixture.log_prob(test_sample)
    log_prob_invariant = estimator.mvn_diag_distr.log_prob(test_sample)
    expected_val = torch.mean(log_prob_model - log_prob_invariant).item()

    calculated_tensor = estimator.estimate_entropy(test_sample)

    assert calculated_tensor.item() == pytest.approx(expected_val, rel=1e-5)
