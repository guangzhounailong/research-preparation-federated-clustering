"""Stage 5: federated vae."""

import matplotlib.pyplot as plt
import pandas as pd
import torch

from ..data import load_data
from ..federated_training import run_federated_training
from ..models import Autoencoder, VAE
from ..partition import partition_data
from ..visualization import plot_generated_samples


def run_stage5(config, output_dir):
    X_train, X_test, y_train, y_test = load_data(
        split_seed=config.split_seed,
        config=config,
    )

    final_results = []
    previous = pd.read_csv(output_dir.parent / "stage4" / "stage4_final.csv")

    for mode in ["iid", "noniid"]:
        generated_samples = []
        # 同一种划分下，AE 和 VAE 使用相同的客户端数据
        train_indices = partition_data(
            y_train,
            num_clients=config.num_clients,
            mode=mode,
            seed=config.partition_seed,
        )

        for model_name, model_class in [
            ("AE", Autoencoder),
            ("VAE", VAE),
        ]:
            for seed in config.seeds:
                if model_name == "AE":
                    # 与 Stage 4 完全相同的训练，直接复用逐种子最终结果。
                    row = previous[
                        (previous["Partition"] == mode) & (previous["Seed"] == seed)
                    ]
                    history = [row.iloc[0].drop(
                        labels=["Partition", "Weighted_ARI", "Weighted_NMI"],
                    ).to_dict()]
                else:
                    print(f"Stage 5: {mode}, VAE, seed={seed}", flush=True)
                    global_model, history = run_federated_training(
                        X_train=X_train,
                        X_test=X_test,
                        y_test=y_test,
                        train_indices=train_indices,
                        seed=seed,
                        model_class=model_class,
                        config=config,
                    )

                # 保存当前实验最后一轮的结果
                final_results.append({
                    "Partition": mode,
                    "Model": model_name,
                    **history[-1],
                })

                if model_name == "VAE":
                    # 固定同一组先验样本，且不改变训练使用的随机状态
                    generator = torch.Generator().manual_seed(config.seeds[0])
                    global_model.eval()
                    with torch.no_grad():
                        z = torch.randn(
                            10, config.latent_dim, generator=generator
                        )
                        samples = global_model.decoder(z)
                    generated_samples.append(samples.numpy().reshape(-1, 8, 8))

        save_path = output_dir / f"stage5_vae_generated_{mode}.png"
        fig = plot_generated_samples(
            generated_samples,
            config.seeds,
            title=f"Federated VAE prior samples - {mode}",
            save_path=save_path,
        )
        plt.close(fig)
        print(f"Stage 5's generated samples: {output_dir}/{save_path.name}")

    final_df = pd.DataFrame(final_results)
    final_df.to_csv(output_dir / "stage5_final.csv", index=False, float_format="%.4f")

    metric_columns = [
        "Reconstruction_loss",
        "KL_loss",
        "ARI",
        "NMI",
        "Silhouette",
        "Training_time_seconds",
    ]

    # 对同一种划分、同一种模型的不同种子结果做统计
    comparison_df = (
        final_df
        .groupby(["Partition", "Model"], sort=False)[metric_columns]
        .agg(["mean"])
    )

    # 将两层列名展开，例如 ARI + mean → ARI_mean
    comparison_df.columns = [
        f"{metric}_{stat}"
        for metric, stat in comparison_df.columns
    ]
    comparison_df = comparison_df.reset_index()

    comparison_df.to_csv(
        output_dir / "stage5_comparison.csv",
        index=False,
        float_format="%.4f",
    )

    print(f"Stage 5's result: {output_dir}/stage5_comparison.csv")

    return comparison_df
