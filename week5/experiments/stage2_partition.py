"""Stage 2: visualize IID and non-IID client class distributions."""

import matplotlib.pyplot as plt

from ..data import load_data
from ..partition import partition_data
from ..visualization import plot_client_distributions


def run_stage2(config, output_dir):
    X_train, _, y_train, _ = load_data(
        split_seed=config.split_seed,
        config=config,
    )

    # 1. Create the IID partition.
    iid_indices = partition_data(
        y_train,
        num_clients=config.num_clients,
        mode="iid",
        seed=config.partition_seed,
    )

    # 2. Create the non-IID partition.
    noniid_indices = partition_data(
        y_train,
        num_clients=config.num_clients,
        mode="noniid",
        seed=config.partition_seed,
    )

    # 3. Plot and save the client class distributions.
    save_path = output_dir / "stage2_distribution.png"

    fig = plot_client_distributions(
        y_train,
        iid_indices,
        noniid_indices,
        save_path=save_path,
    )

    plt.close(fig)
    print(f"Stage 2's result: {output_dir}/stage2_distribution.png")

    return iid_indices, noniid_indices
