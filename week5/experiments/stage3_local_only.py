"""Stage 3: evaluate independent local-only AE clustering."""

import numpy as np
import pandas as pd

from ..client import run_local_only
from ..data import load_data
from ..partition import partition_data


def run_stage3(config, output_dir):
    X_train, X_test, y_train, y_test = load_data(
        split_seed=config.split_seed,
        config=config,
    )

    client_results = []
    weighted_results = []

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
            rows = run_local_only(
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                train_indices=train_indices,
                test_indices=test_indices,
                seed=seed,
                config=config,
            )

            seed_df = pd.DataFrame(rows)

            # For this seed, weight client metrics by their test sample counts.
            weighted_results.append({
                "Partition": mode,
                "Seed": seed,
                "Weighted_ARI": np.average(
                    seed_df["ARI"],
                    weights=seed_df["N_test"],
                ),
                "Weighted_NMI": np.average(
                    seed_df["NMI"],
                    weights=seed_df["N_test"],
                ),
                "Weighted_Silhouette": np.average(
                    seed_df["Silhouette"], weights=seed_df["N_test"],
                ),
            })

            for row in rows:
                client_results.append({
                    "Partition": mode,
                    **row,
                })

    pd.DataFrame(client_results).to_csv(
        output_dir / "stage3_client_results.csv", index=False,
        float_format="%.4f",
    )
    pd.DataFrame(weighted_results).to_csv(
        output_dir / "stage3_weighted_results.csv", index=False,
        float_format="%.4f",
    )

    # Average per-client ARI and NMI across seeds.
    client_df = (
        pd.DataFrame(client_results)
        .groupby(
            ["Partition", "Client"],
            as_index=False,
            sort=False,
        )[["ARI", "NMI"]]
        .mean()
    )

    # Compute client-weighted metrics per seed, then average across seeds.
    weighted_df = (
        pd.DataFrame(weighted_results)
        .groupby(
            "Partition",
            as_index=False,
            sort=False,
        )[["Weighted_ARI", "Weighted_NMI"]]
        .mean()
    )

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
        f"Stage 3's result: {output_dir}/stage3_clients.csv, "
        f"{output_dir}/stage3_weighted.csv"
    )

    return client_df, weighted_df
