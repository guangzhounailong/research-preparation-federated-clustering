from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

import config
from data import load_data
from evaluation import evaluate_clustering, run_kmeans, evaluate_client_weighted
from models import Autoencoder, VAE, extract_features, train_model

import matplotlib.pyplot as plt

from partition import partition_data
from visualization import (
    plot_client_distributions,
    plot_federated_curves,
)

from client import run_local_only

from federated_training import run_federated_training

def run_stage1():
    X_train, X_test, _, y_test = load_data(config.SPLIT_SEED)
    results = []

    for seed in config.SEEDS:
        for method in ["Raw", "PCA", "AE", "VAE"]:
            np.random.seed(seed)
            torch.manual_seed(seed)

            if method == "Raw":
                Z_train, Z_test = X_train, X_test
                representation_training_time = 0.0

            elif method == "PCA":
                pca = PCA(n_components=config.PCA_DIM, svd_solver="full")
                start = perf_counter()
                pca.fit(X_train)
                representation_training_time = perf_counter() - start

                Z_train = pca.transform(X_train)
                Z_test = pca.transform(X_test)

            else:
                model_class = Autoencoder if method == "AE" else VAE
                model = model_class(latent_dim=config.LATENT_DIM)
                start = perf_counter()
                train_model(model, X_train, seed)
                representation_training_time = perf_counter() - start

                Z_train = extract_features(model, X_train)
                Z_test = extract_features(model, X_test)

            y_pred, kmeans_training_time = run_kmeans(Z_train, Z_test, seed)
            metrics = evaluate_clustering(
                Z_test, y_test, y_pred,
                training_time=representation_training_time + kmeans_training_time,
            )
            results.append({"Model": method, **metrics})

    results = pd.DataFrame(results)
    results = results.rename(columns={
        "Silhouette": "Silhouette score",
        "Training_time_seconds": "Training time (s)",
    })


    means = results.groupby("Model", sort=False).mean()
    output_dir = Path(__file__).resolve().parent / "result"
    output_dir.mkdir(parents=True, exist_ok=True)
    means.to_csv(output_dir / "stage1.csv", float_format="%.4f")
    print("Stage 1's result: week5/result/stage1.csv")
    return results

def run_stage2():
    X_train, _, y_train, _ = load_data(
        split_seed=config.SPLIT_SEED
    )

    # 1. 生成 IID 划分
    iid_indices = partition_data(
        y_train,
        num_clients=config.NUM_CLIENTS,
        mode="iid",
        seed=config.PARTITION_SEED,
    )

    # 2. 生成 non-IID 划分
    noniid_indices = partition_data(
        y_train,
        num_clients=config.NUM_CLIENTS,
        mode="noniid",
        seed=config.PARTITION_SEED,
    )

    # 3. 绘制、保存分布图
    save_path = (
        Path(__file__).resolve().parent
        / "result"
        / "stage2_distribution.png"
    )

    fig = plot_client_distributions(
        y_train,
        iid_indices,
        noniid_indices,
        save_path=save_path,
    )

    print("Stage 2's result: week5/result/stage2_distribution.png")

    return iid_indices, noniid_indices

def run_stage3():
    X_train, X_test, y_train, y_test = load_data(
        split_seed=config.SPLIT_SEED
    )

    client_results = []
    weighted_results = []

    for mode in ["iid", "noniid"]:
        train_indices = partition_data(
            y_train,
            num_clients=config.NUM_CLIENTS,
            mode=mode,
            seed=config.PARTITION_SEED,
        )

        test_indices = partition_data(
            y_test,
            num_clients=config.NUM_CLIENTS,
            mode=mode,
            seed=config.PARTITION_SEED,
        )

        for seed in config.SEEDS:
            rows = run_local_only(
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                train_indices=train_indices,
                test_indices=test_indices,
                seed=seed,
            )

            seed_df = pd.DataFrame(rows)

            # 当前种子：按各客户端测试样本数计算加权平均
            weighted_results.append({
                "Partition": mode,
                "Weighted_ARI": np.average(
                    seed_df["ARI"],
                    weights=seed_df["N_test"],
                ),
                "Weighted_NMI": np.average(
                    seed_df["NMI"],
                    weights=seed_df["N_test"],
                ),
            })

            for row in rows:
                client_results.append({
                    "Partition": mode,
                    "Client": row["Client"],
                    "ARI": row["ARI"],
                    "NMI": row["NMI"],
                })

    # 每个客户端的 ARI、NMI：对随机种子取平均
    client_df = (
        pd.DataFrame(client_results)
        .groupby(
            ["Partition", "Client"],
            as_index=False,
            sort=False,
        )[["ARI", "NMI"]]
        .mean()
    )

    # 每个种子先计算客户端加权平均，再对随机种子取平均
    weighted_df = (
        pd.DataFrame(weighted_results)
        .groupby(
            "Partition",
            as_index=False,
            sort=False,
        )[["Weighted_ARI", "Weighted_NMI"]]
        .mean()
    )

    output_dir = Path(__file__).resolve().parent / "result"
    output_dir.mkdir(parents=True, exist_ok=True)

    client_df.to_csv(
        output_dir / "stage3_clients.csv",
        index=False,
        float_format="%.4f",
    )

    weighted_df.to_csv(
        output_dir / "stage3_weighted.csv",
        index=False,
        float_format="%.4f",
    )

    print(
        "Stage 3's result: week5/result/stage3_clients.csv, "
        "week5/result/stage3_weighted.csv"
    )

    return client_df, weighted_df

def run_stage4():
    X_train, X_test, y_train, y_test = load_data(
        split_seed=config.SPLIT_SEED
    )

    history_results = []
    final_results = []

    for mode in ["iid", "noniid"]:
        train_indices = partition_data(
            y_train,
            num_clients=config.NUM_CLIENTS,
            mode=mode,
            seed=config.PARTITION_SEED,
        )

        test_indices = partition_data(
            y_test,
            num_clients=config.NUM_CLIENTS,
            mode=mode,
            seed=config.PARTITION_SEED,
        )

        for seed in config.SEEDS:
            global_model, history = run_federated_training(
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                train_indices=train_indices,
                seed=seed,
            )

            for row in history:
                    history_results.append({
                        "Partition": mode,
                        **row,
                    })

            Z_train = extract_features(global_model, X_train)
            Z_test = extract_features(global_model, X_test)
            y_pred, _ = run_kmeans(Z_train, Z_test, seed)

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
        .agg(["mean", "std"])
    )

    summary_df.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary_df.columns
    ]
    summary_df = summary_df.reset_index()

    output_dir = Path(__file__).resolve().parent / "result"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "Stage 4's result: week5/result/stage4_history.csv, "
        "week5/result/stage4_summary.csv"
    )

    fig = plot_federated_curves(
        history_df,
        save_path=output_dir / "stage4_curves.png",
    )

    print("Stage 4's curves: week5/result/stage4_curves.png")

    # 读取相同配置下的 Stage 3 结果
    local_df = pd.read_csv(
        output_dir / "stage3_weighted.csv"
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
        "week5/result/stage4_local_comparison.csv"
    )

    return history_df, summary_df

def main():
    torch.set_num_threads(1)
    run_stage4()


if __name__ == "__main__":
    main()
