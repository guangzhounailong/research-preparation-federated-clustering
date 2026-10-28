"""Stage 7: ablation."""

import pandas as pd

from ..data import load_data
from ..federated_training import run_federated_training
from ..models import Autoencoder
from ..partition import partition_data
from ..training import extract_features
from .common import save_features


def run_stage7(config, output_dir):
    X_train, X_test, y_train, y_test = load_data(
        split_seed=config.split_seed,
        config=config,
    )

    train_indices = partition_data(
        y_train,
        num_clients=config.num_clients,
        mode="noniid",
        seed=config.partition_seed,
    )

    ablation_clip_norm = config.ablation_clip_norm
    # 固定实际噪声标准差，单独观察裁剪阈值变化的影响
    noise_std = config.ablation_noise_std

    settings = [
        ("Full method", ablation_clip_norm, noise_std),
        ("w/o clipping", None, noise_std),
        ("w/o noise", ablation_clip_norm, 0.0),
        ("w/o both", None, 0.0),
    ]

    history_results = []
    final_results = []

    for setting_name, clip_norm, setting_noise_std in settings:
        for seed in config.seeds:
            print(f"Stage 7: {setting_name}, seed={seed}", flush=True)
            global_model, history = run_federated_training(
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                train_indices=train_indices,
                seed=seed,
                model_class=Autoencoder,
                clip_norm=clip_norm,
                noise_std=setting_noise_std,
                config=config,
            )
            if setting_name == "Full method" and seed == config.seeds[0]:
                save_features(
                    output_dir, f"stage7_full_seed{seed}",
                    extract_features(global_model, X_train),
                    extract_features(global_model, X_test),
                )
            setting = {
                "Setting": setting_name,
                "Partition": "noniid",
                "Model": "AE",
                "Clip_norm": clip_norm,
                "Noise_sigma": setting_noise_std,
                "Local_epochs": config.local_epochs,
            }
            for row in history:
                history_results.append({
                    **setting,
                    **row,
                })
            final_results.append({
                **setting,
                **history[-1],
            })

    history_df = pd.DataFrame(history_results)
    final_df = pd.DataFrame(final_results)
    final_df.to_csv(output_dir / "stage7_final.csv", index=False, float_format="%.4f")

    metric_columns = [
        "Reconstruction_loss",
        "ARI",
        "NMI",
        "Silhouette",
        "Norm_before_mean",
        "Norm_after_mean",
        "Training_time_seconds",
    ]

    # 每组实验：汇总不同种子的最后一轮结果
    comparison_df = (
        final_df
        .groupby(
            [
                "Setting",
                "Partition",
                "Model",
                "Clip_norm",
                "Noise_sigma",
                "Local_epochs",
                "Round",
            ],
            sort=False,
            dropna=False,
        )[metric_columns]
        .agg(["mean"])
    )

    comparison_df.columns = [
        f"{metric}_{stat}"
        for metric, stat in comparison_df.columns
    ]
    comparison_df = comparison_df.reset_index()

    history_df.to_csv(
        output_dir / "stage7_history.csv",
        index=False,
        float_format="%.4f",
    )

    comparison_df.to_csv(
        output_dir / "stage7_comparison.csv",
        index=False,
        float_format="%.4f",
    )

    print(
        f"Stage 7's result: {output_dir}/stage7_history.csv, "
        f"{output_dir}/stage7_comparison.csv"
    )

    return history_df, comparison_df
