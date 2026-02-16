import torch

import dpgmm.utils.file_system as fs_utils
from dpgmm.metrics import ComplexityFromTraceEstimator

TRACE_PATH = "out/results/cgs_19.pkl"
DATA_TRACE_PATH = "out/results/cgs_0.pkl"
SAMPLES_NUM = 100_000

data_trace = fs_utils.read_pickle(DATA_TRACE_PATH)
data = torch.as_tensor(data_trace["data"], dtype=torch.float32)

estimator = ComplexityFromTraceEstimator(
    trace_path=TRACE_PATH,
    data_trace_path=DATA_TRACE_PATH,
    samples_num=SAMPLES_NUM,
)

entropy_sampled = estimator.estimate_entropy_with_sampling()
entropy_data = estimator.estimate_entropy_on_data(data)
relative_diff = abs(entropy_sampled - entropy_data) / abs(entropy_data) * 100

print(f"Entropy from sampling: {entropy_sampled}")
print(f"Entropy on data: {entropy_data}")
print(f"Relative difference: {relative_diff:.2f}%")
