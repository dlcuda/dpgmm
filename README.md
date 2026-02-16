# dpgmm

**dpgmm** is a library implementing high-performance `MCMC` sampler for Dirichlet Process Gaussian Mixture Models (`DPGMM`). Built on `PyTorch` and accelerated with `Triton` kernels, it is designed to handle high-dimensional data efficiently.

## Key Features

**High Performance**: Optimized Gibbs sampling leveraging GPU acceleration via `PyTorch` and `Triton` kernels.

**Data Generation**: Built-in utilities to generate high-dimensional synthetic datasets for validation.

**Observability**: Native integration with `Weights & Biases` for experiment tracking.

**Metrics**: Comprehensive tools for calculating assignment log likelihood, data complexity, and data dimensions entanglement.

**Modern Stack**: Developed using modern Python tooling (`uv`, `ruff`, `pytest`).

## Installation

The package will be available on PyPI soon.

```bash
# Coming soon
pip install gmm_sampler
```

For now, you can clone the repository and set up the environment using `uv` or `pip`.

## Usage

### 1. Generating data and running the sampler

Initialize the generator and the Gibbs sampler.

```py
import torch
from dpgmm.datasets import GaussianDataGenerator
from dpgmm.samplers import FullCovarianceCollapsedGibbsSampler, DiagCovarianceCollapsedGibbsSampler

# 1. Generate synthetic data
data_generator = GaussianDataGenerator(cov_type="full")
data_payload = data_generator.generate(n_points=256, data_dim=2, num_components=4)
data_tensor = torch.as_tensor(data_payload["data"])

# 2. Initialize the Sampler
sampler = FullCovarianceCollapsedGibbsSampler(
    init_strategy="init_data_stats",
    max_clusters_num=10,
    batch_size=1
)

# 3. Fit the model
result = sampler.fit(iterations_num=100, data=data_tensor)

# Access results
cluster_params = result["cluster_params"]
cluster_assignment = result["cluster_assignment"]
alpha = result["alpha"]
```

### 2. Visualizing results

Visualize the clusters, covariance matrices, and assignments.

```py
from dpgmm.visualisation import ClusterParamsVisualizer

data_visualizer = ClusterParamsVisualizer()

data_visualizer.plot_params_full_covariance(
    data_payload["data"],
    centers=cluster_params["mean"],
    cov_chol=cluster_params["cov_chol"],
    assignment=cluster_assignment,
    trace_alpha=alpha,
)
```

### 3. Checkpointing

You can save checkpoints during training and resume from them later.

```py
# To save during training, specify an out_dir
sampler.fit(iterations_num=25, data=data_tensor, out_dir="out/save_and_load")

# To resume, pass the path to the snapshot directory in kwargs
additional_kwargs = {"restore_snapshot_pkl_path": "out/save_and_load/"}

sampler_restored = FullCovarianceCollapsedGibbsSampler(
    init_strategy="init_data_stats",
    max_clusters_num=10,
    batch_size=1,
    **additional_kwargs,
)
```

### 4. Calculating metrics

#### Data complexity

Estimate entropy from sampling versus data to gauge model fit.

```py
from dpgmm.metrics import ComplexityFromTraceEstimator

estimator = ComplexityFromTraceEstimator(
    trace_path="/path/to/results/cgs_19.pkl",
    data_trace_path="/path/to/results/cgs_0.pkl",
    samples_num=100_000,
)

entropy_sampled = estimator.estimate_entropy_with_sampling()
entropy_data = estimator.estimate_entropy_on_data(data_tensor)

print(f"Entropy from sampling: {entropy_sampled}")
print(f"Entropy on data: {entropy_data}")
```

#### Data dimensions entanglement

Calculate the `KL divergence` between joint and product marginals to measure feature entanglement.

```py
from dpgmm.metrics import EntanglementFromTraceEstimator

estimator = EntanglementFromTraceEstimator(
    trace_path="/path/to/results/cgs_99.pkl",
    samples_num=100_000
)

dkl_joint_prod = estimator.calculate_joint_and_prod_dkl()
dkl_symmetric = estimator.calculate_symmetric_dkl()

print(f"KL(Joint || Marginals Prod):  {dkl_joint_prod:.4f}")
```

## Integrations & observability

The sampler supports `W&B` out of the box for tracking loss curves, cluster evolution, and system metrics. To enable experiment tracking, just make sure to export `WAND_API_KEY` environment variable.

```bash
export WANDB_API_KEY=your_key_here
```

## Benchmarks

Thanks to `Triton` kernels, `dpgmm` achieves significant speedups compared to standard implementations, especially in high-dimensional experiments. The following table showcases the average iteration time (in seconds) for $N=1000$ points using the full covariance model.

| Data dim | PyTorch CPU [s]       | Optimized GPU [s]  | Speedup        |
| -------- | --------------------- | ------------------ | -------------- |
| 128      | $2.893 \pm 0.702$     | $0.656 \pm 0.004$  | $\times 4.41$  |
| 256      | $6.300 \pm 1.237$     | $0.520 \pm 0.027$  | $\times 12.11$ |
| 512      | $23.857 \pm 1.843$    | $0.456 \pm 0.018$  | $\times 52.36$ |
| 1024     | $53.204 \pm 2.394$    | $1.196 \pm 0.024$  | $\times 44.47$ |
| 2048     | $140.795 \pm 4.535$   | $3.141 \pm 0.138$  | $\times 44.83$ |
| 4096     | $494.269 \pm 6.651$   | $13.414 \pm 0.611$ | $\times 36.85$ |
| 8192     | $3479.447 \pm 72.901$ | $37.880 \pm 1.458$ | $\times 91.85$ |

## Development

This project uses `uv` for dependency management and `Task` (`go-task`) for orchestrating development workflows.

```bash
# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install
```

### Task Automation

A `Taskfile.yml` is provided to simplify common development command - use the following commands:

```bash
# Run linter and formatter (Ruff)
uv run task lint

# Run security audits (Bandit & Safety)
uv run task audit

# Check code complexity (Xenon)
uv run task complexity

# Run all quality and security checks
uv run task check-all

# Build documentation
uv run task build-docs

# Run all tests
uv run task run-tests
```
