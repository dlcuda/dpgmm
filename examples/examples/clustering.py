import torch

from dpgmm.datasets import GaussianDataGenerator
from dpgmm.samplers import (
    DiagCovarianceCollapsedGibbsSampler,
    FullCovarianceCollapsedGibbsSampler,
)
from dpgmm.visualisation import ClusterParamsVisualizer

data_generator_full = GaussianDataGenerator(cov_type="full")
data_generator_diag = GaussianDataGenerator(cov_type="diag")
data_full = data_generator_full.generate(n_points=256, data_dim=2, num_components=4)
data_diag = data_generator_diag.generate(n_points=256, data_dim=2, num_components=4)

data_visualizer = ClusterParamsVisualizer()

# Full Covariance CGS on Gaussian data with full covariance matrix
sampler_full = FullCovarianceCollapsedGibbsSampler(
    init_strategy="init_data_stats", max_clusters_num=10, batch_size=1
)

data_full_tensor = torch.as_tensor(data_full["data"])
result = sampler_full.fit(iterations_num=100, data=data_full_tensor)

cluster_params = result["cluster_params"]
cluster_assignment = result["cluster_assignment"]
alpha = result["alpha"]

data_visualizer.plot_params_full_covariance(
    data_full["data"],
    centers=cluster_params["mean"],
    cov_chol=cluster_params["cov_chol"],
    assignment=cluster_assignment,
    trace_alpha=alpha,
)

# Diagonal Covariance CGS on Gaussian data with diagonal covariance matrix
sampler_diag = DiagCovarianceCollapsedGibbsSampler(
    init_strategy="init_data_stats", max_clusters_num=10, batch_size=1
)

data_diag_tensor = torch.as_tensor(data_diag["data"])
result = sampler_diag.fit(iterations_num=100, data=data_diag_tensor)

cluster_params = result["cluster_params"]
cluster_assignment = result["cluster_assignment"]
alpha = result["alpha"]

data_visualizer.plot_params_diag_covariance(
    data_diag["data"],
    centers=cluster_params["mean"],
    variances=cluster_params["var"],
    assignment=cluster_assignment,
    trace_alpha=alpha,
)
