from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
from sklearn.decomposition import PCA


def plot_representations(representations, y_test, seed, save_path=None):
    """分别拟合二维 PCA；测试标签仅用于着色，不参与降维。"""
    classes, color_ids = np.unique(y_test, return_inverse=True)
    cmap = plt.get_cmap("tab10", len(classes))
    norm = BoundaryNorm(np.arange(len(classes) + 1) - 0.5, cmap.N)
    fig, axes = plt.subplots(
        2, 3, figsize=(16, 9), constrained_layout=True,
    )

    for ax, (title, (Z_train, Z_test)) in zip(axes.flat, representations.items()):
        # 每组使用相同的 PCA 设置，仅在训练特征上拟合。
        pca = PCA(n_components=2, svd_solver="full")
        pca.fit(Z_train)
        coordinates = pca.transform(Z_test)
        points = ax.scatter(
            coordinates[:, 0], coordinates[:, 1],
            c=color_ids, cmap=cmap, norm=norm,
            s=16, alpha=0.7, linewidths=0,
        )
        variance = pca.explained_variance_ratio_ * 100
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"PC1 ({variance[0]:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({variance[1]:.1f}% variance)")
        ax.grid(alpha=0.2)

    for ax in axes.flat[len(representations):]:
        ax.set_visible(False)

    fig.colorbar(
        points, ax=list(axes.flat[:len(representations)]),
        ticks=np.arange(len(classes)), label="Reference digit label",
        shrink=0.8,
    ).set_ticklabels(classes)
    fig.suptitle(
        f"2D PCA of representations | seed={seed} | same {len(y_test)} test samples\n"
        "PCA fitted on training features separately; axes are not aligned.\n"
        "Clipping + noise is a privacy-mechanism simulation (no formal DP guarantee).",
        fontsize=13,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def count_client_classes(y, client_indices, classes):
    y = np.asarray(y)

    counts = np.zeros(
        (len(client_indices), len(classes)),
        dtype=int,
    )

    for client_id, indices in enumerate(client_indices):
        client_labels = y[indices]

        for class_id, label in enumerate(classes):
            counts[client_id, class_id] = np.count_nonzero(
                client_labels == label
            )

    return counts


def plot_client_distributions(
    y,
    iid_indices,
    noniid_indices,
    save_path=None,
):
    y = np.asarray(y)
    classes = np.unique(y)

    iid_counts = count_client_classes(
        y, iid_indices, classes
    )
    noniid_counts = count_client_classes(
        y, noniid_indices, classes
    )

    # 两张图共用相同的颜色尺度
    vmax = max(iid_counts.max(), noniid_counts.max())

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 5),
        constrained_layout=True,
    )

    for ax, counts, title in zip(
        axes,
        [iid_counts, noniid_counts],
        ["IID partition", "Non-IID partition"],
    ):
        image = ax.imshow(
            counts,
            cmap="Blues",
            vmin=0,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_title(title)
        ax.set_xlabel("Digit class")
        ax.set_ylabel("Client")

        ax.set_xticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)

        # 客户端名称后面附上该客户端的总样本数
        client_totals = counts.sum(axis=1)

        ax.set_yticks(np.arange(len(counts)))
        ax.set_yticklabels([
            f"Client {i} (n={total})"
            for i, total in enumerate(client_totals)
        ])

        # 在每个格子里标注样本数
        for row in range(counts.shape[0]):
            for col in range(counts.shape[1]):
                value = counts[row, col]

                ax.text(
                    col,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > vmax / 2 else "black",
                    fontsize=9,
                )

    fig.colorbar(
        image,
        ax=axes,
        label="Number of samples",
        shrink=0.85,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

    return fig


def plot_generated_samples(samples_by_seed, seeds, title, save_path):
    fig, axes = plt.subplots(
        len(samples_by_seed),
        len(samples_by_seed[0]),
        figsize=(10, 1.3 * len(samples_by_seed)),
        squeeze=False,
        constrained_layout=True,
    )

    for row, (samples, seed) in enumerate(zip(samples_by_seed, seeds)):
        for col, sample in enumerate(samples):
            ax = axes[row, col]
            ax.imshow(sample, cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(f"Seed {seed}")

    fig.suptitle(title)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig

def plot_federated_curves(history_df, save_path=None):
    metrics = [
        ("Reconstruction_loss", "Reconstruction loss"),
        ("ARI", "ARI"),
        ("NMI", "NMI"),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 4.5),
        constrained_layout=True,
    )

    for ax, (column, title) in zip(axes, metrics):
        for mode, label in [
            ("iid", "IID"),
            ("noniid", "Non-IID"),
        ]:
            # 取出当前数据划分下的所有实验记录
            mode_df = history_df[
                history_df["Partition"] == mode
            ]

            # 同一轮的多个种子：计算均值和标准差
            stats = (
                mode_df.groupby("Round")[column]
                .agg(["mean", "std"])
                .sort_index()
            )

            rounds = stats.index.to_numpy()
            mean = stats["mean"].to_numpy()
            std = stats["std"].to_numpy()

            line, = ax.plot(rounds, mean, label=label)

            ax.fill_between(
                rounds,
                mean - std,
                mean + std,
                color=line.get_color(),
                alpha=0.15,
            )

        ax.set_title(title)
        ax.set_xlabel("Communication round")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

    return fig
