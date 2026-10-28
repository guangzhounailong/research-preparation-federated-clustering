"""Stage 2: partition."""

import matplotlib.pyplot as plt

from ..data import load_data
from ..partition import partition_data
from ..visualization import plot_client_distributions


def run_stage2(config, output_dir):
    X_train, _, y_train, _ = load_data(
        split_seed=config.split_seed,
        config=config,
    )

    # 1. 生成 IID 划分
    iid_indices = partition_data(
        y_train,
        num_clients=config.num_clients,
        mode="iid",
        seed=config.partition_seed,
    )

    # 2. 生成 non-IID 划分
    noniid_indices = partition_data(
        y_train,
        num_clients=config.num_clients,
        mode="noniid",
        seed=config.partition_seed,
    )

    # 3. 绘制、保存分布图
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
