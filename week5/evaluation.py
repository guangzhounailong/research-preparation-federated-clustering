from time import perf_counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

import config


def run_kmeans(Z_train, Z_test, seed):
    kmeans = KMeans(
        n_clusters=config.NUM_CLUSTERS,
        n_init=config.KMEANS_N_INIT,
        random_state=seed,
    )
    start = perf_counter()
    kmeans.fit(Z_train)
    training_time = perf_counter() - start

    y_pred = kmeans.predict(Z_test)
    return y_pred, training_time


def evaluate_clustering(features, y_true, y_pred, training_time):
    cluster_count = len(np.unique(y_pred))
    silhouette = (
        silhouette_score(features, y_pred)
        if 1 < cluster_count < len(features)
        else float("nan")
    )

    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "Silhouette": silhouette,
        "Training_time_seconds": training_time,
    }
