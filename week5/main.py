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
    plot_generated_samples,
    plot_representations,
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
        .agg(["mean"])
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

def run_stage5():
    X_train, X_test, y_train, y_test = load_data(
        split_seed=config.SPLIT_SEED
    )

    final_results = []
    output_dir = Path(__file__).resolve().parent / "result"
    output_dir.mkdir(parents=True, exist_ok=True)

    for mode in ["iid", "noniid"]:
        generated_samples = []
        # 同一种划分下，AE 和 VAE 使用相同的客户端数据
        train_indices = partition_data(
            y_train,
            num_clients=config.NUM_CLIENTS,
            mode=mode,
            seed=config.PARTITION_SEED,
        )

        for model_name, model_class in [
            ("AE", Autoencoder),
            ("VAE", VAE),
        ]:
            for seed in config.SEEDS:
                global_model, history = run_federated_training(
                    X_train=X_train,
                    X_test=X_test,
                    y_test=y_test,
                    train_indices=train_indices,
                    seed=seed,
                    model_class=model_class,
                )

                # 保存当前实验最后一轮的结果
                final_results.append({
                    "Partition": mode,
                    "Model": model_name,
                    **history[-1],
                })

                if model_name == "VAE":
                    # 固定同一组先验样本，且不改变训练使用的随机状态
                    generator = torch.Generator().manual_seed(config.SEEDS[0])
                    global_model.eval()
                    with torch.no_grad():
                        z = torch.randn(
                            10, config.LATENT_DIM, generator=generator
                        )
                        samples = global_model.decoder(z)
                    generated_samples.append(samples.numpy().reshape(-1, 8, 8))

        save_path = output_dir / f"stage5_vae_generated_{mode}.png"
        fig = plot_generated_samples(
            generated_samples,
            config.SEEDS,
            title=f"Federated VAE prior samples - {mode}",
            save_path=save_path,
        )
        plt.close(fig)
        print(f"Stage 5's generated samples: week5/result/{save_path.name}")

    final_df = pd.DataFrame(final_results)

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

    print("Stage 5's result: week5/result/stage5_comparison.csv")

    return comparison_df

def run_stage6():
    X_train, X_test, y_train, y_test = load_data(
        split_seed=config.SPLIT_SEED
    )

    history_results = []
    final_results = []

    for mode in ["iid", "noniid"]:
        # 同一种划分下，所有噪声设置使用相同的客户端数据
        train_indices = partition_data(
            y_train,
            num_clients=config.NUM_CLIENTS,
            mode=mode,
            seed=config.PARTITION_SEED,
        )

        for noise_multiplier in config.NOISE_MULTIPLIERS:
            for seed in config.SEEDS:
                _, history = run_federated_training(
                    X_train=X_train,
                    X_test=X_test,
                    y_test=y_test,
                    train_indices=train_indices,
                    seed=seed,
                    model_class=Autoencoder,
                    clip_norm=config.CLIP_NORM,
                    noise_multiplier=noise_multiplier,
                )

                # 标记当前实验的设置
                setting = {
                    "Partition": mode,
                    "Model": "AE",
                    "Clip_norm": config.CLIP_NORM,
                    "Noise_multiplier": noise_multiplier,
                    "Local_epochs": config.LOCAL_EPOCHS,
                }

                # 保存每一轮的结果
                for row in history:
                    history_results.append({
                        **setting,
                        **row,
                    })

                # 最后一轮用于比较不同噪声设置
                final_results.append({
                    **setting,
                    **history[-1],
                })

    history_df = pd.DataFrame(history_results)
    final_df = pd.DataFrame(final_results)

    metric_columns = [
        "Reconstruction_loss",
        "ARI",
        "NMI",
        "Silhouette",
        "Norm_before_mean",
        "Norm_after_mean",
        "Training_time_seconds",
    ]

    # 每种划分、每档噪声：汇总不同种子的最终结果
    comparison_df = (
        final_df
        .groupby(
            [
                "Partition",
                "Model",
                "Clip_norm",
                "Noise_multiplier",
                "Local_epochs",
                "Round",
            ],
            sort=False,
        )[metric_columns]
        .agg(["mean"])
    )

    comparison_df.columns = [
        f"{metric}_{stat}"
        for metric, stat in comparison_df.columns
    ]
    comparison_df = comparison_df.reset_index()

    output_dir = Path(__file__).resolve().parent / "result"
    output_dir.mkdir(parents=True, exist_ok=True)

    history_df.to_csv(
        output_dir / "stage6_history.csv",
        index=False,
        float_format="%.4f",
    )

    comparison_df.to_csv(
        output_dir / "stage6_comparison.csv",
        index=False,
        float_format="%.4f",
    )

    print(
        "Stage 6's result: week5/result/stage6_history.csv, "
        "week5/result/stage6_comparison.csv"
    )

    return history_df, comparison_df

def run_stage7():
    X_train, X_test, y_train, y_test = load_data(
        split_seed=config.SPLIT_SEED
    )

    train_indices = partition_data(
        y_train,
        num_clients=config.NUM_CLIENTS,
        mode="noniid",
        seed=config.PARTITION_SEED,
    )

    ablation_clip_norm = 0.3
    # 固定实际噪声标准差，单独观察裁剪阈值变化的影响
    noise_std = 0.005

    settings = [
        ("Full method", ablation_clip_norm, noise_std),
        ("w/o clipping", None, noise_std),
        ("w/o noise", ablation_clip_norm, 0.0),
        ("w/o both", None, 0.0),
    ]

    history_results = []
    final_results = []

    for setting_name, clip_norm, setting_noise_std in settings:
        for seed in config.SEEDS:
            _, history = run_federated_training(
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                train_indices=train_indices,
                seed=seed,
                model_class=Autoencoder,
                clip_norm=clip_norm,
                noise_std=setting_noise_std,
            )
            setting = {
                "Setting": setting_name,
                "Partition": "noniid",
                "Model": "AE",
                "Clip_norm": clip_norm,
                "Noise_std": setting_noise_std,
                "Local_epochs": config.LOCAL_EPOCHS,
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
                "Noise_std",
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

    output_dir = Path(__file__).resolve().parent / "result"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        "Stage 7's result: week5/result/stage7_history.csv, "
        "week5/result/stage7_comparison.csv"
    )

    return history_df, comparison_df

def run_stage8():
    X_train, X_test, y_train, y_test = load_data(
        split_seed=config.SPLIT_SEED
    )
    # 预先固定种子和完整测试集，不根据标签或图形效果挑选样本。
    seed = config.SEEDS[0]
    # 与 Stage 7 的 Full method 一致，直接固定噪声标准差。
    ablation_clip_norm = 0.3
    noise_std = 0.005
    representations = {"Raw features": (X_train, X_test)}

    np.random.seed(seed)
    torch.manual_seed(seed)
    centralized_model = Autoencoder(latent_dim=config.LATENT_DIM)
    train_model(centralized_model, X_train, seed)
    representations["Centralized AE"] = (
        extract_features(centralized_model, X_train),
        extract_features(centralized_model, X_test),
    )

    for mode in ["iid", "noniid"]:
        # 训练标签仅用于构造客户端划分，不参与表示学习或聚类拟合。
        train_indices = partition_data(
            y_train,
            num_clients=config.NUM_CLIENTS,
            mode=mode,
            seed=config.PARTITION_SEED,
        )
        settings = [(None, None)]
        if mode == "noniid":
            settings.append((ablation_clip_norm, noise_std))

        for clip_norm, setting_noise_std in settings:
            # y_test 在已有训练函数中仅计算评价指标，不影响更新或模型选择。
            global_model, _ = run_federated_training(
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                train_indices=train_indices,
                seed=seed,
                model_class=Autoencoder,
                clip_norm=clip_norm,
                noise_std=setting_noise_std,
            )
            partition_label = "IID" if mode == "iid" else "non-IID"
            title = (
                f"Federated AE ({partition_label}, "
                f"{config.COMMUNICATION_ROUNDS} rounds)"
            )
            if clip_norm is not None:
                title += (
                    f"\nClipping C={clip_norm}, noise std={setting_noise_std}"
                )
            representations[title] = (
                extract_features(global_model, X_train),
                extract_features(global_model, X_test),
            )

    save_path = (
        Path(__file__).resolve().parent
        / "result"
        / "stage8_representations.png"
    )
    fig = plot_representations(
        representations,
        y_test,
        seed=seed,
        save_path=save_path,
    )
    plt.close(fig)
    print(f"Stage 8's result: week5/result/{save_path.name}")
    return fig


def run_stage9():
    X_train, X_test, y_train, y_test = load_data(
        split_seed=config.SPLIT_SEED
    )
    results = []

    for seed in config.SEEDS:
        for method in ["Raw", "PCA", "AE", "VAE"]:
            np.random.seed(seed)
            torch.manual_seed(seed)

            if method == "Raw":
                Z_train, Z_test = X_train, X_test
                latent_dim = X_train.shape[1]
            elif method == "PCA":
                pca = PCA(n_components=config.PCA_DIM, svd_solver="full")
                Z_train = pca.fit_transform(X_train)
                Z_test = pca.transform(X_test)
                latent_dim = config.PCA_DIM
            else:
                model_class = Autoencoder if method == "AE" else VAE
                model = model_class(latent_dim=config.LATENT_DIM)
                train_model(model, X_train, seed)
                Z_train = extract_features(model, X_train)
                Z_test = extract_features(model, X_test)
                latent_dim = config.LATENT_DIM

            y_pred, _ = run_kmeans(Z_train, Z_test, seed)
            metrics = evaluate_clustering(
                Z_test, y_test, y_pred, training_time=0.0,
            )
            results.append({
                "Method": f"{method} + K-means",
                "Partition": "Centralized",
                "Latent Dim.": latent_dim,
                "Privacy Setting": "None",
                "Communication Rounds": 0,
                "Seed": seed,
                **{metric: metrics[metric] for metric in ["ARI", "NMI", "Silhouette"]},
            })

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
        partition_label = "IID" if mode == "iid" else "Non-IID"

        if mode == "noniid":
            for seed in config.SEEDS:
                local_df = pd.DataFrame(run_local_only(
                    X_train=X_train,
                    X_test=X_test,
                    y_test=y_test,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    seed=seed,
                ))
                # 独立的本地潜在空间分别评价，再按测试样本数加权。
                results.append({
                    "Method": "Local AE + K-means",
                    "Partition": partition_label,
                    "Latent Dim.": config.LATENT_DIM,
                    "Privacy Setting": "None",
                    "Communication Rounds": 0,
                    "Seed": seed,
                    **{
                        metric: np.average(local_df[metric], weights=local_df["N_test"])
                        for metric in ["ARI", "NMI", "Silhouette"]
                    },
                })

        settings = [("Federated AE + K-means", Autoencoder, None, None)]
        if mode == "noniid":
            settings.extend([
                ("Federated VAE + K-means", VAE, None, None),
                # 沿用 Stage 7 / 8 的 Full method 设置。
                ("Federated AE + clipping/noise", Autoencoder, 0.3, 0.005),
            ])

        for method, model_class, clip_norm, noise_std in settings:
            for seed in config.SEEDS:
                _, history = run_federated_training(
                    X_train=X_train,
                    X_test=X_test,
                    y_test=y_test,
                    train_indices=train_indices,
                    seed=seed,
                    model_class=model_class,
                    clip_norm=clip_norm,
                    noise_std=noise_std,
                )
                # 联邦方法使用完整测试集指标，与 Stage 4 / 5 的口径一致。
                results.append({
                    "Method": method,
                    "Partition": partition_label,
                    "Latent Dim.": config.LATENT_DIM,
                    "Privacy Setting": (
                        "None" if clip_norm is None
                        else f"C={clip_norm}, noise std={noise_std} (simulation)"
                    ),
                    "Communication Rounds": history[-1]["Round"],
                    "Seed": seed,
                    **{
                        metric: history[-1][metric]
                        for metric in ["ARI", "NMI", "Silhouette"]
                    },
                })

    results_df = pd.DataFrame(results)
    setting_columns = [
        "Method", "Partition", "Latent Dim.", "Privacy Setting",
        "Communication Rounds",
    ]
    comparison_rows = []
    for setting, group in results_df.groupby(setting_columns, sort=False):
        row = dict(zip(setting_columns, setting))
        for metric in ["ARI", "NMI", "Silhouette"]:
            # 标准差反映不同种子的波动，不是置信区间。
            row[metric] = f"{group[metric].mean():.4f} ± {group[metric].std():.4f}"
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)[[
        "Method", "Partition", "Latent Dim.", "Privacy Setting",
        "ARI", "NMI", "Silhouette", "Communication Rounds",
    ]]
    output_dir = Path(__file__).resolve().parent / "result"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_dir / "stage9_comparison.csv", index=False)
    print("Stage 9's result: week5/result/stage9_comparison.csv")
    return results_df, comparison_df


def main():
    torch.set_num_threads(1)
    run_stage9()


if __name__ == "__main__":
    main()
