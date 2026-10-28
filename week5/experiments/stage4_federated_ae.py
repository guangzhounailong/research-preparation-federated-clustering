"""Stage 4: federated ae."""

import matplotlib.pyplot as plt
import pandas as pd

from ..clustering import run_kmeans
from ..data import load_data
from ..evaluation import evaluate_client_weighted
from ..federated_training import run_federated_training
from ..partition import partition_data
from ..training import extract_features
from ..visualization import plot_federated_curves
from .common import save_features


def run_stage4(config, output_dir):
    X_train, X_test, y_train, y_test = load_data(
        split_seed=config.split_seed,
        config=config,
    )

    history_results = []
    final_results = []

    for mode in ["iid", "noniid"]:
        train_indices = partition_data(
            y_train,
            num_clients=config.num_clients,
            mode=mode,
            seed=config.partition_seed,
        )

        test_indices = partition_data(
            y_test,
            num_clients=config.num_clients,
            mode=mode,
            seed=config.partition_seed,
        )

        for seed in config.seeds:
            print(f"Stage 4: {mode}, seed={seed}", flush=True)
            global_model, history = run_federated_training(
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                train_indices=train_indices,
                seed=seed,
                config=config,
            )

            for row in history:
                history_results.append({
                    "Partition": mode,
                    **row,
                })

            Z_train = extract_features(global_model, X_train)
            Z_test = extract_features(global_model, X_test)
            if seed == config.seeds[0]:
                save_features(output_dir, f"stage4_ae_{mode}_seed{seed}", Z_train, Z_test)
            y_pred, _ = run_kmeans(Z_train, Z_test, seed, config=config)

            weighted_metrics = evaluate_client_weighted(
                y_true=y_test,
                y_pred=y_pred,
                test_indices=test_indices,
            )

            final_results.append({
                "Partition": mode,
                **history[-1],
                **weighted_metrics,
            })

    history_df = pd.DataFrame(history_results)
    final_df = pd.DataFrame(final_results)
    final_df.to_csv(output_dir / "stage4_final.csv", index=False, float_format="%.4f")

    metric_columns = [
        "Reconstruction_loss",
        "ARI",
        "NMI",
        "Silhouette",
        "Training_time_seconds",
        "Weighted_ARI",
        "Weighted_NMI",
    ]

    summary_df = (
        final_df
        .groupby("Partition", sort=False)[metric_columns]
        .agg(["mean"])
    )

    summary_df.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary_df.columns
    ]
    summary_df = summary_df.reset_index()

    history_df.to_csv(
        output_dir / "stage4_history.csv",
        index=False,
        float_format="%.4f",
    )

    summary_df.to_csv(
        output_dir / "stage4_summary.csv",
        index=False,
        float_format="%.4f",
    )

    print(
        f"Stage 4's result: {output_dir}/stage4_history.csv, "
        f"{output_dir}/stage4_summary.csv"
    )

    fig = plot_federated_curves(
        history_df,
        save_path=output_dir / "stage4_curves.png",
    )

    plt.close(fig)
    print(f"Stage 4's curves: {output_dir}/stage4_curves.png")

    # 读取相同配置下的 Stage 3 结果
    local_df = pd.read_csv(
        output_dir.parent / "stage3" / "stage3_weighted.csv"
    )

    # 提取 Stage 4 的客户端加权均值
    federated_df = summary_df[[
        "Partition",
        "Weighted_ARI_mean",
        "Weighted_NMI_mean",
    ]].rename(columns={
        "Weighted_ARI_mean": "Weighted_ARI",
        "Weighted_NMI_mean": "Weighted_NMI",
    })

    # 按数据划分对应合并
    comparison_df = local_df.merge(
        federated_df,
        on="Partition",
        suffixes=("_local", "_federated"),
        validate="one_to_one",
    )

    # 每种数据划分分别列出 ARI、NMI，差值保留正负号
    comparison_rows = []
    for _, row in comparison_df.iterrows():
        for metric in ["ARI", "NMI"]:
            local_value = row[f"Weighted_{metric}_local"]
            federated_value = row[f"Weighted_{metric}_federated"]
            comparison_rows.append({
                "Partition": {"iid": "IID", "noniid": "non-IID"}[row["Partition"]],
                "metric": metric,
                "Local-only": local_value,
                "Federated": federated_value,
                "Fed - Local": f"{federated_value - local_value:+.4f}",
            })

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_df.to_csv(
        output_dir / "stage4_local_comparison.csv",
        index=False,
        float_format="%.4f",
    )

    print(
        "Stage 4's comparison: "
        f"{output_dir}/stage4_local_comparison.csv"
    )

    return history_df, summary_df
