from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributions as dist

from dpgmm.metrics.cgs_trace.full_cov import FullCovTraceReader
from dpgmm.metrics.entanglement.entanglement_estimator import EntanglementEstimator
from dpgmm.utils.distributions.multivariate_student_t import MultivariateStudentT

SAMPLES_NUM = 100_000
SEED = 42


@pytest.fixture(autouse=True)
def set_random_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)


@pytest.fixture
def real_trace_path():
    current_dir = Path(__file__).parent
    trace_file = current_dir / "trace.pickle"
    if not trace_file.exists():
        pkls = list(current_dir.glob("*.pkl"))
        if pkls:
            trace_file = pkls[0]
        else:
            pytest.skip(f"No trace file found in {current_dir}")
    return str(trace_file)


@pytest.fixture
def dkl_calc(real_trace_path):
    trace_reader = FullCovTraceReader(real_trace_path)

    joint_distr = trace_reader.prepare_t_student_mixture()
    sorted_indices = sorted(trace_reader.cluster_counts.keys())

    all_params = [
        trace_reader.prepare_full_cov_t_student_params(k) for k in sorted_indices
    ]
    means = torch.stack([p.mean for p in all_params])
    cov_chols = torch.stack([p.cov_chol for p in all_params])
    dofs = torch.stack([p.dof for p in all_params])
    weights = torch.as_tensor(
        trace_reader.get_cluster_unnormalized_weights(), dtype=torch.float32
    )

    return EntanglementEstimator(
        joint_distr=joint_distr,
        weights=weights,
        means=means,
        cov_chols=cov_chols,
        dofs=dofs,
        samples_num=SAMPLES_NUM,
    )


def test_pq_ratio_for_samples(dkl_calc):
    samples = dkl_calc.joint_distr.sample((SAMPLES_NUM,))
    ratios = dkl_calc.pq_ratio_for_samples(samples)
    ratios_np = ratios.detach().cpu().numpy()

    assert ratios_np.shape == (SAMPLES_NUM,)
    assert np.mean(ratios_np) > 0


def test_calculate_joint_and_prod_dkl(dkl_calc):
    dkl = dkl_calc.calculate_joint_and_prod_dkl()

    assert dkl > 0
    assert not np.isnan(dkl)


def test_calculate_symmetric_dkl(dkl_calc):
    dkl = dkl_calc.calculate_symmetric_dkl()

    assert dkl > 0
    assert not np.isnan(dkl)


def test_dkl_correctness(real_trace_path):
    trace_reader = FullCovTraceReader(real_trace_path)

    params = trace_reader.prepare_full_cov_t_student_params(0)

    L = params.cov_chol
    L_diag = torch.diag(torch.diagonal(L))

    means = params.mean.unsqueeze(0)
    cov_chols = L_diag.unsqueeze(0)
    dofs = params.dof.unsqueeze(0)
    weights = torch.tensor([1.0])

    mix = dist.Categorical(weights)
    comp = MultivariateStudentT(df=dofs, loc=means, scale_tril=cov_chols)
    joint_density_diag = dist.MixtureSameFamily(mix, comp)

    temp_estimator = EntanglementEstimator(
        joint_distr=joint_density_diag,
        weights=weights,
        means=means,
        cov_chols=cov_chols,
        dofs=dofs,
        samples_num=SAMPLES_NUM,
    )

    dkl_mean = temp_estimator.calculate_joint_and_prod_dkl()
    assert dkl_mean < 0.1
