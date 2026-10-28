"""全局聚类指标与按客户端测试样本数加权的指标。"""

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


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


def evaluate_client_weighted(y_true, y_pred, test_indices):
    client_ari = []
    client_nmi = []
    client_sizes = []

    for indices in test_indices:
        local_true = y_true[indices]
        local_pred = y_pred[indices]

        client_ari.append(
            adjusted_rand_score(local_true, local_pred)
        )
        client_nmi.append(
            normalized_mutual_info_score(local_true, local_pred)
        )
        client_sizes.append(len(indices))

    return {
        "Weighted_ARI": np.average(
            client_ari,
            weights=client_sizes,
        ),
        "Weighted_NMI": np.average(
            client_nmi,
            weights=client_sizes,
        ),
    }
