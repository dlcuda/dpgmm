import pytest
import torch
from torch import Tensor
from torch.distributions import Categorical, Independent, MixtureSameFamily, StudentT

from dpgmm.metrics.cgs_trace.diag_cov import (
    DiagCovTraceReader,
    DiagCovTStudentParams,
)

# -------------------------
# Fixtures
# -------------------------


@pytest.fixture
def dummy_diag_trace():
    clusters_num = 3
    data_dim = 2

    means = [torch.randn(data_dim) for _ in range(clusters_num)]
    vars_ = [torch.ones(data_dim) for _ in range(clusters_num)]

    cluster_assignment = [{0, 3, 7}, {1, 4, 5, 9}, {2, 6, 8}]

    trace = {
        "cluster_params": {"mean": means, "var": vars_},
        "cluster_counts": [len(c) for c in cluster_assignment],
        "clusters_num": clusters_num,
        "data_dim": data_dim,
        "cluster_assignment": cluster_assignment,
        "alpha": 1.0,
    }

    return trace


@pytest.fixture
def diag_trace_reader(dummy_diag_trace):
    return DiagCovTraceReader(dummy_diag_trace)


# -------------------------
# Tests
# -------------------------


def test_diag_cov_t_student_params_tensor_validation():
    mean = torch.zeros(2)
    var = torch.ones(2)
    dof = torch.tensor(5.0)

    # valid creation
    params = DiagCovTStudentParams(dof=dof, mean=mean, var=var)
    assert isinstance(params.dof, Tensor)
    assert isinstance(params.mean, Tensor)
    assert isinstance(params.var, Tensor)

    # invalid type should raise TypeError
    import pytest

    with pytest.raises(Exception):
        DiagCovTStudentParams(dof=5.0, mean=mean, var=var)


def test_prepare_diag_cov_t_student_params_shapes(
    diag_trace_reader: DiagCovTraceReader,
):
    params = diag_trace_reader.prepare_diag_cov_t_student_params(0)
    assert isinstance(params, DiagCovTStudentParams)
    assert params.mean.shape[0] == diag_trace_reader.data_dim
    assert params.var.shape[0] == diag_trace_reader.data_dim
    assert params.dof.numel() == 1


def test_prepare_diag_cov_t_student_params_scaled_var(
    diag_trace_reader: DiagCovTraceReader,
):
    params = diag_trace_reader.prepare_diag_cov_t_student_params(0)
    # Scale factor should be positive
    assert torch.all(params.var > 0)


def test_prepare_t_student_mixture_type(diag_trace_reader: DiagCovTraceReader):
    mixture = diag_trace_reader.prepare_t_student_mixture()
    assert isinstance(mixture, MixtureSameFamily)
    assert isinstance(mixture.mixture_distribution, Categorical)
    # Component distribution should be Independent of StudentT
    comp = mixture.component_distribution
    assert isinstance(comp, Independent)
    assert isinstance(comp.base_dist, StudentT)


def test_prepare_t_student_mixture_shapes(diag_trace_reader: DiagCovTraceReader):
    mixture = diag_trace_reader.prepare_t_student_mixture()
    # Check mixture component shapes
    base = mixture.component_distribution.base_dist
    assert base.loc.shape[1] == diag_trace_reader.data_dim


def test_prepare_diag_cov_t_student_params_consistency(
    diag_trace_reader: DiagCovTraceReader,
):
    for i in range(diag_trace_reader.clusters_num):
        params = diag_trace_reader.prepare_diag_cov_t_student_params(i)
        # dof, mean, var are all tensors
        assert isinstance(params.dof, Tensor)
        assert isinstance(params.mean, Tensor)
        assert isinstance(params.var, Tensor)


def test_all_clusters_params(diag_trace_reader: DiagCovTraceReader):
    for i in range(diag_trace_reader.clusters_num):
        params = diag_trace_reader.prepare_diag_cov_t_student_params(i)
        assert params.mean.shape[0] == diag_trace_reader.data_dim
        assert params.var.shape[0] == diag_trace_reader.data_dim
