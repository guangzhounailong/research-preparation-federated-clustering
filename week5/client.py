from time import perf_counter

import numpy as np
import torch

from evaluation import evaluate_clustering, run_kmeans
from models import Autoencoder, extract_features, train_model

from copy import deepcopy

import config

def run_local_only(
    X_train,
    X_test,
    y_test,
    train_indices,
    test_indices,
    seed,
):
    results = []

    for client_id, (train_idx, test_idx) in enumerate(
        zip(train_indices, test_indices)
    ):
        # 当前客户端的数据
        X_local_train = X_train[train_idx]
        X_local_test = X_test[test_idx]
        y_local_test = y_test[test_idx]

        # 每个客户端创建独立模型，使用相同初始化控制实验条件
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = Autoencoder()

        # 只在本地训练数据上训练 AE
        start = perf_counter()
        train_model(model, X_local_train, seed)
        ae_training_time = perf_counter() - start

        # 使用当前客户端的编码器提取特征
        Z_train = extract_features(model, X_local_train)
        Z_test = extract_features(model, X_local_test)

        # 在本地训练特征上拟合，在本地测试特征上预测
        y_pred, kmeans_training_time = run_kmeans(
            Z_train,
            Z_test,
            seed,
        )

        metrics = evaluate_clustering(
            Z_test,
            y_local_test,
            y_pred,
            training_time=ae_training_time + kmeans_training_time,
        )

        results.append({
            "Seed": seed,
            "Client": client_id,
            "N_train": len(train_idx),
            "N_test": len(test_idx),
            **metrics,
        })

    return results

def train_client(global_model, X_local_train, seed):
    local_model = deepcopy(global_model)

    train_model(
        local_model,
        X_local_train,
        seed,
        epochs=config.LOCAL_EPOCHS,
    )

    return local_model.state_dict(), len(X_local_train)
