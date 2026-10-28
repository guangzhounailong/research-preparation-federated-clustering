"""Stage 9: 汇总逐种子结果，同时保留全局与客户端加权的评价口径。"""

import pandas as pd


def run_stage9(config, output_dir):
    metrics = ["ARI", "NMI", "Silhouette"]
    results = []
    run_dir = output_dir.parent
    centralized = pd.read_csv(run_dir / "stage1" / "stage1_results.csv")
    # 与原 Stage 9 一样，先按种子，再按方法排列。
    for seed in config.seeds:
        for _, row in centralized[centralized["Seed"] == seed].iterrows():
            method = row["Model"]
            latent_dim = {
                "Raw": config.input_dim,
                "PCA": config.pca_dim,
                "AE": config.latent_dim,
                "VAE": config.latent_dim,
            }[method]
            results.append({
                "Method": f"{method} + K-means",
                "Partition": "Centralized",
                "Latent Dim.": latent_dim,
                "Privacy Setting": "None",
                "Evaluation": "Global test set",
                "Communication Rounds": 0,
                "Seed": seed,
                **{metric: row[metric] for metric in metrics},
            })

    local = pd.read_csv(run_dir / "stage3" / "stage3_weighted_results.csv")
    federated = pd.read_csv(run_dir / "stage4" / "stage4_final.csv")
    vae = pd.read_csv(run_dir / "stage5" / "stage5_final.csv")
    privacy = pd.read_csv(run_dir / "stage7" / "stage7_final.csv")

    for mode, label in [("iid", "IID"), ("noniid", "Non-IID")]:
        if mode == "noniid":
            for _, row in local[local["Partition"] == mode].iterrows():
                results.append({
                    "Method": "Local AE + K-means",
                    "Partition": label,
                    "Latent Dim.": config.latent_dim,
                    "Privacy Setting": "None",
                    "Evaluation": "Client-weighted test metrics",
                    "Communication Rounds": 0,
                    "Seed": row["Seed"],
                    **{metric: row[f"Weighted_{metric}"] for metric in metrics},
                })

        settings = [
            ("Federated AE + K-means", federated, "None"),
            ("Federated VAE + K-means", vae[vae["Model"] == "VAE"], "None"),
        ]
        if mode == "noniid":
            settings.extend([
                (
                    "Federated AE + clipping/noise",
                    privacy[privacy["Setting"] == "Full method"],
                    f"C={config.ablation_clip_norm:.4f}, noise sigma={config.ablation_noise_std:.4f} (simulation)",
                ),
            ])
        for method, frame, privacy_setting in settings:
            for _, row in frame[frame["Partition"] == mode].iterrows():
                results.append({
                    "Method": method,
                    "Partition": label,
                    "Latent Dim.": config.latent_dim,
                    "Privacy Setting": privacy_setting,
                    "Evaluation": "Global test set",
                    "Communication Rounds": row["Round"],
                    "Seed": row["Seed"],
                    **{metric: row[metric] for metric in metrics},
                })

    results_df = pd.DataFrame(results)
    results_df["Seed"] = results_df["Seed"].astype(int)
    results_df["Communication Rounds"] = results_df["Communication Rounds"].astype(int)
    results_df.to_csv(output_dir / "stage9_results.csv", index=False, float_format="%.4f")
    setting_columns = [
        "Method", "Partition", "Latent Dim.", "Privacy Setting",
        "Evaluation", "Communication Rounds",
    ]
    summary = results_df.groupby(setting_columns, sort=False)[metrics].agg(["mean"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output_dir / "stage9_summary.csv", index=False, float_format="%.4f")

    comparison = summary[setting_columns].copy()
    for metric in metrics:
        comparison[metric] = summary[f"{metric}_mean"].map("{:.4f}".format)
    comparison = comparison[[
        "Method", "Partition", "Latent Dim.", "Privacy Setting", "Evaluation",
        *metrics, "Communication Rounds",
    ]]
    comparison.to_csv(output_dir / "stage9_comparison.csv", index=False, float_format="%.4f")
    print(f"Stage 9's result: {output_dir / 'stage9_comparison.csv'}")
    return results_df, comparison
