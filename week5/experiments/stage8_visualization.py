"""Stage 8: 使用同次运行保存的表示作图，不重新训练模型。"""

import matplotlib.pyplot as plt

from ..data import load_data
from ..visualization import plot_representations
from .common import load_features


def run_stage8(config, output_dir):
    X_train, X_test, _, y_test = load_data(config=config)
    seed = config.seeds[0]
    representations = {
        "Raw features": (X_train, X_test),
        "Centralized AE": load_features(output_dir.parent / "stage1", f"stage1_ae_seed{seed}"),
    }
    for mode, label in [("iid", "IID"), ("noniid", "non-IID")]:
        title = f"Federated AE ({label}, {config.communication_rounds} rounds)"
        representations[title] = load_features(
            output_dir.parent / "stage4", f"stage4_ae_{mode}_seed{seed}",
        )

    title = (
        f"Federated AE (non-IID, {config.communication_rounds} rounds)\n"
        f"Clipping C={config.ablation_clip_norm}, noise std={config.ablation_noise_std}"
    )
    representations[title] = load_features(output_dir.parent / "stage7", f"stage7_full_seed{seed}")
    save_path = output_dir / "stage8_representations.png"
    fig = plot_representations(representations, y_test, seed, save_path=save_path)
    plt.close(fig)
    print(f"Stage 8's result: {save_path}")
    return fig
