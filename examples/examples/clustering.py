import torch

from dpgmm.core.model import DPGMM
from dpgmm.datasets import GaussianDataGenerator
from dpgmm.visualisation import ClusterParamsVisualizer

data_generator_full = GaussianDataGenerator(cov_type="full")
data_generator_diag = GaussianDataGenerator(cov_type="diag")
data_full = data_generator_full.generate(n_points=256, data_dim=2, num_components=4)
data_diag = data_generator_diag.generate(n_points=256, data_dim=2, num_components=4)

data_visualizer = ClusterParamsVisualizer()

# --- Full Covariance ---
mm_full = DPGMM(
    inference_method="cgs",
    covariance_type="full",
    init_strategy="init_data_stats",
    max_clusters_num=10,
    batch_size=1,
    verbose=False,
)
mm_full.fit(data_full["data"], iterations_num=100)

labels_full = mm_full.cluster()
print(f"[full] unique clusters: {len(set(labels_full))}")

x_test = torch.as_tensor(data_full["data"][:5], dtype=torch.float32)

log_p_full = mm_full.density(x_test)
print(f"[full] log-densities:  {log_p_full}")
print(f"[full] densities:      {torch.exp(log_p_full)}")

probs_full = mm_full.soft_cluster(x_test)
print(f"[full] soft assignments (N=5, K={probs_full.shape[1]}):")
for i, row in enumerate(probs_full):
    hard = row.argmax().item()
    print(f"  point {i}: {row.numpy().round(3)}  → hard cluster: {hard}")
# sanity check: soft assignments should sum to 1
assert torch.allclose(probs_full.sum(dim=1), torch.ones(5), atol=1e-5), (
    "Soft assignments don't sum to 1!"
)
print("[full] soft assignment rows sum to 1 ✓")

# sanity check: hard cluster from soft_cluster should match cluster()
soft_hard = probs_full.argmax(dim=1).numpy()
assert (soft_hard == labels_full[:5]).all(), (
    f"Hard labels mismatch: {soft_hard} vs {labels_full[:5]}"
)
print("[full] soft_cluster argmax matches cluster() ✓")

result_full = mm_full._fit_result
data_visualizer.plot_params_full_covariance(
    data_full["data"],
    centers=result_full["cluster_params"]["mean"],
    cov_chol=result_full["cluster_params"]["cov_chol"],
    assignment=result_full["cluster_assignment"],
    trace_alpha=result_full["alpha"],
)

# --- Diagonal Covariance ---
mm_diag = DPGMM(
    inference_method="cgs",
    covariance_type="diag",
    init_strategy="init_data_stats",
    max_clusters_num=10,
    batch_size=1,
    verbose=False,
)
mm_diag.fit(data_diag["data"], iterations_num=100)

labels_diag = mm_diag.cluster()
print(f"\n[diag] unique clusters: {len(set(labels_diag))}")

x_test_diag = torch.as_tensor(data_diag["data"][:5], dtype=torch.float32)

log_p_diag = mm_diag.density(x_test_diag)
print(f"[diag] log-densities:  {log_p_diag}")
print(f"[diag] densities:      {torch.exp(log_p_diag)}")

probs_diag = mm_diag.soft_cluster(x_test_diag)
print(f"[diag] soft assignments (N=5, K={probs_diag.shape[1]}):")
for i, row in enumerate(probs_diag):
    hard = row.argmax().item()
    print(f"  point {i}: {row.numpy().round(3)}  → hard cluster: {hard}")

assert torch.allclose(probs_diag.sum(dim=1), torch.ones(5), atol=1e-5), (
    "Soft assignments don't sum to 1!"
)
print("[diag] soft assignment rows sum to 1 ✓")

soft_hard_diag = probs_diag.argmax(dim=1).numpy()
assert (soft_hard_diag == labels_diag[:5]).all(), (
    f"Hard labels mismatch: {soft_hard_diag} vs {labels_diag[:5]}"
)
print("[diag] soft_cluster argmax matches cluster() ✓")

result_diag = mm_diag._fit_result
data_visualizer.plot_params_diag_covariance(
    data_diag["data"],
    centers=result_diag["cluster_params"]["mean"],
    variances=result_diag["cluster_params"]["var"],
    assignment=result_diag["cluster_assignment"],
    trace_alpha=result_diag["alpha"],
)
