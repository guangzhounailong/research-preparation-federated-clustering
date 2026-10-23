from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

import config
from data import load_data
from evaluation import evaluate_clustering, run_kmeans
from models import Autoencoder, VAE, extract_features, train_model


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
    table = means.map(lambda value: f"{value:.4f}")
    print(table.to_string())
    return results


def main():
    torch.set_num_threads(1)
    run_stage1()


if __name__ == "__main__":
    main()
