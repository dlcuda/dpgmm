import pytest
import torch
from torch import Tensor

from dpgmm.metrics.cgs_trace.full_cov import (
    FullCovTraceReader,
    FullCovTStudentParams,
)

# -------------------------
# Fixtures
# -------------------------


@pytest.fixture
def dummy_trace():
    clusters_num = 3
    data_dim = 2

    means = [torch.randn(data_dim) for _ in range(clusters_num)]
    cov_chols = [torch.eye(data_dim) for _ in range(clusters_num)]

    cluster_assignment = [{0, 3, 7}, {1, 4, 5, 9}, {2, 6, 8}]

    trace = {
        "cluster_params": {"mean": means, "cov_chol": cov_chols},
        "cluster_counts": [len(c) for c in cluster_assignment],
        "clusters_num": clusters_num,
        "data_dim": data_dim,
        "cluster_assignment": cluster_assignment,
        "alpha": 1.0,
    }

    return trace


@pytest.fixture
def trace_reader(dummy_trace):
    return FullCovTraceReader(dummy_trace)


# -------------------------
# Tests
# -------------------------


def test_full_cov_t_student_params_tensor_validation():
    mean = torch.zeros(2)
    cov_chol = torch.eye(2)
    dof = torch.tensor(5.0)

    # valid creation
    params = FullCovTStudentParams(dof=dof, mean=mean, cov_chol=cov_chol)
    assert isinstance(params.dof, Tensor)
    assert isinstance(params.mean, Tensor)
    assert isinstance(params.cov_chol, Tensor)

    # invalid type should raise TypeError
    import pytest

    with pytest.raises(Exception):
        FullCovTStudentParams(dof=5.0, mean=mean, cov_chol=cov_chol)


def test_prepare_full_cov_t_student_params_shapes(trace_reader: FullCovTraceReader):
    params = trace_reader.prepare_full_cov_t_student_params(0)
    assert isinstance(params, FullCovTStudentParams)
    assert params.mean.shape[0] == trace_reader.data_dim
    assert params.cov_chol.shape[0] == trace_reader.data_dim
    assert params.dof.numel() == 1


def test_prepare_full_cov_t_student_params_scaled_cov(trace_reader: FullCovTraceReader):
    params = trace_reader.prepare_full_cov_t_student_params(0)
    # Scale factor should be positive
    assert torch.all(torch.diag(params.cov_chol) > 0)


def test_prepare_t_student_mixture_type(trace_reader: FullCovTraceReader):
    mixture = trace_reader.prepare_t_student_mixture()
    from torch.distributions import MixtureSameFamily

    assert isinstance(mixture, MixtureSameFamily)


def test_prepare_t_student_mixture_shapes(trace_reader: FullCovTraceReader):
    mixture = trace_reader.prepare_t_student_mixture()
    # Check mixture component dimensions
    assert mixture.component_distribution.loc.shape[1] == trace_reader.data_dim


def test_prepare_full_cov_t_student_params_consistency(
    trace_reader: FullCovTraceReader,
):
    for i in range(trace_reader.clusters_num):
        params = trace_reader.prepare_full_cov_t_student_params(i)
        # dof, mean, cov_chol are all tensors
        assert isinstance(params.dof, Tensor)
        assert isinstance(params.mean, Tensor)
        assert isinstance(params.cov_chol, Tensor)


def test_all_clusters_params(trace_reader: FullCovTraceReader):
    for i in range(trace_reader.clusters_num):
        params = trace_reader.prepare_full_cov_t_student_params(i)
        assert params.mean.shape[0] == trace_reader.data_dim
        assert params.cov_chol.shape[0] == trace_reader.data_dim
