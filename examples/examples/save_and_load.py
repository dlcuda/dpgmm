import torch

from dpgmm.datasets import GaussianDataGenerator
from dpgmm.samplers import FullCovarianceCollapsedGibbsSampler

data_generator_full = GaussianDataGenerator(cov_type="full")
data = data_generator_full.generate(n_points=256, data_dim=2, num_components=4)

additional_args = {
    "skip_epochs_logging": 5,
}

sampler = FullCovarianceCollapsedGibbsSampler(
    init_strategy="init_data_stats",
    max_clusters_num=10,
    batch_size=1,
    **additional_args,
)

data_tensor = torch.as_tensor(data["data"])
_ = sampler.fit(iterations_num=25, data=data_tensor, out_dir="out/save_and_load")

additional_kwargs = {"restore_snapshot_pkl_path": "out/save_and_load/"}

sampler_restored = FullCovarianceCollapsedGibbsSampler(
    init_strategy="init_data_stats",
    max_clusters_num=10,
    batch_size=1,
    **additional_kwargs,
)
