from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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