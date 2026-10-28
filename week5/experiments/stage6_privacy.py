"""Stage 6: evaluate federated AE with update clipping and Gaussian noise."""

import pandas as pd

from ..data import load_data
from ..federated_training import run_federated_training
from ..models import Autoencoder
from ..partition import partition_data


def run_stage6(config, output_dir):
    X_train, X_test, y_train, y_test = load_data(
        split_seed=config.split_seed,
        config=config,
    )

    history_results = []
    final_results = []

    for mode in ["iid", "noniid"]:
        # Use the same client data for all noise settings within each partition.
        train_indices = partition_data(
            y_train,
            num_clients=config.num_clients,
            mode=mode,
            seed=config.partition_seed,
        )

        for noise_multiplier in config.noise_multipliers:
            for seed in config.seeds:
                print(f"Stage 6: {mode}, noise={noise_multiplier}, seed={seed}", flush=True)
                _, history = run_federated_training(
                    X_train=X_train,
                    X_test=X_test,
                    y_test=y_test,
                    train_indices=train_indices,
                    seed=seed,
                    model_class=Autoencoder,
                    clip_norm=config.clip_norm,
                    noise_multiplier=noise_multiplier,
                    config=config,
                )

                # Record the current experiment settings.
                setting = {
                    "Partition": mode,
                    "Model": "AE",
                    "Clip_norm": config.clip_norm,
                    "Noise_multiplier": noise_multiplier,
                    "Local_epochs": config.local_epochs,
                }

                # Save the results from every round.
                for row in history:
                    history_results.append({
                        **setting,
                        **row,
                    })

                # Use the final round to compare noise settings.
                final_results.append({
                    **setting,
                    **history[-1],
                })

    history_df = pd.DataFrame(history_results)
    final_df = pd.DataFrame(final_results)
    final_df.to_csv(output_dir / "stage6_final.csv", index=False, float_format="%.4f")

    metric_columns = [
        "Reconstruction_loss",
        "ARI",
        "NMI",
        "Silhouette",
        "Norm_before_mean",
        "Norm_after_mean",
        "Training_time_seconds",
    ]

    # Aggregate final results across seeds for each partition and noise level.
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
        f"Stage 6's result: {output_dir}/stage6_history.csv, "
        f"{output_dir}/stage6_comparison.csv"
    )

    return history_df, comparison_df
