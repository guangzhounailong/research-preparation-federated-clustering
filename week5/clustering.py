"""只在训练表示上拟合 K-means，在测试表示上预测。"""

from time import perf_counter

from sklearn.cluster import KMeans

from .config import DEFAULT_CONFIG


def run_kmeans(Z_train, Z_test, seed, *, config=DEFAULT_CONFIG):
    kmeans = KMeans(
        n_clusters=config.num_clusters,
        n_init=config.kmeans_n_init,
        random_state=seed,
    )
    start = perf_counter()
    kmeans.fit(Z_train)
    training_time = perf_counter() - start

    y_pred = kmeans.predict(Z_test)
    return y_pred, training_time
