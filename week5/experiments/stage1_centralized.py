"""Stage 1: compare centralized Raw, PCA, AE, and VAE clustering."""

from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from ..clustering import run_kmeans
from ..data import load_data
from ..evaluation import evaluate_clustering
from ..models import Autoencoder, VAE
from ..training import extract_features, train_model
from .common import save_features


def run_stage1(config, output_dir):
    X_train, X_test, _, y_test = load_data(config.split_seed, config=config)
    results = []

    for seed in config.seeds:
        for method in ["Raw", "PCA", "AE", "VAE"]:
            np.random.seed(seed)
            torch.manual_seed(seed)

            if method == "Raw":
                Z_train, Z_test = X_train, X_test
                representation_training_time = 0.0

            elif method == "PCA":
                pca = PCA(n_components=config.pca_dim, svd_solver="full")
                start = perf_counter()
                pca.fit(X_train)
                representation_training_time = perf_counter() - start

                Z_train = pca.transform(X_train)
                Z_test = pca.transform(X_test)

            else:
                model_class = Autoencoder if method == "AE" else VAE
                model = model_class(latent_dim=config.latent_dim, config=config)
                start = perf_counter()
                train_model(model, X_train, seed, config=config)
                representation_training_time = perf_counter() - start

                Z_train = extract_features(model, X_train)
                Z_test = extract_features(model, X_test)

            y_pred, kmeans_training_time = run_kmeans(Z_train, Z_test, seed, config=config)
            metrics = evaluate_clustering(
                Z_test, y_test, y_pred,
                training_time=representation_training_time + kmeans_training_time,
            )
            results.append({"Model": method, "Seed": seed, **metrics})
            if method == "AE" and seed == config.seeds[0]:
                save_features(output_dir, f"stage1_ae_seed{seed}", Z_train, Z_test)

    results = pd.DataFrame(results)
    results.to_csv(output_dir / "stage1_results.csv", index=False, float_format="%.4f")
    results = results.rename(columns={
        "Silhouette": "Silhouette score",
        "Training_time_seconds": "Training time (s)",
    })

    means = results.drop(columns="Seed").groupby("Model", sort=False).mean()
    means.to_csv(output_dir / "stage1.csv", float_format="%.4f")
    print(f"Stage 1's result: {output_dir}/stage1.csv")
    summary = (
        results.drop(columns="Seed").groupby("Model", sort=False).agg(["mean"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(output_dir / "stage1_summary.csv", float_format="%.4f")
    return results
