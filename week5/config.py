"""Define immutable default settings for all experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "digits"
    input_dim: int = 64
    pixel_max: float = 16.0
    test_size: float = 0.2
    split_seed: int = 47
    seeds: tuple[int, ...] = (47, 48, 49)

    # AE / VAE
    hidden_dim: int = 128
    latent_dim: int = 16
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    beta: float = 0.01

    # PCA / K-means
    pca_dim: int = 16
    num_clusters: int = 10
    kmeans_n_init: int = 100

    # Client partition
    num_clients: int = 5
    partition_seed: int = 47

    # Federated training
    communication_rounds: int = 200
    local_epochs: int = 1

    # Stage 6: noise std = multiplier × clipping norm
    clip_norm: float = 0.5
    noise_multipliers: tuple[float, ...] = (0.00, 0.01, 0.05, 0.10)

    # Stage 7–9: fixed noise std, independent of clipping
    ablation_clip_norm: float = 0.3
    ablation_noise_std: float = 0.005


DEFAULT_CONFIG = ExperimentConfig()
