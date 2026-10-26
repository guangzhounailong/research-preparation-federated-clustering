from time import perf_counter

import numpy as np
import torch

import config
from client import train_client
from evaluation import evaluate_clustering, run_kmeans
from models import Autoencoder, extract_features
from server import fedavg

def run_federated_training(
    X_train,
    X_test,
    y_test,
    train_indices,
    seed,
):
    #初始化全局模型
    np.random.seed(seed)
    torch.manual_seed(seed)
    global_model = Autoencoder()

    history = []
    training_time = 0.0

    for round_id in range(1, config.COMMUNICATION_ROUNDS + 1):
        client_states = []
        client_sizes = []

        start = perf_counter()

        for client_id, indices in enumerate(train_indices):
            X_local_train = X_train[indices]

            local_seed = (seed + (round_id - 1) * len(train_indices) + client_id)

            state, size = train_client(
                global_model,
                X_local_train,
                local_seed,
            )

            client_states.append(state)
            client_sizes.append(size)

        #更新全局模型
        global_state = fedavg(client_states, client_sizes)
        global_model.load_state_dict(global_state)
        training_time += perf_counter() - start

        #使用更新后的全局编码器提取特征 
        Z_train = extract_features(global_model, X_train)
        Z_test = extract_features(global_model, X_test)

        #训练集拟合 K-means，测试集预测与评价
        y_pred, _ = run_kmeans(Z_train, Z_test, seed)

        metrics = evaluate_clustering(
            Z_test,
            y_test,
            y_pred,
            training_time=training_time,
        )

        #计算全局模型在测试集上的重建误差
        global_model.eval()
        with torch.no_grad():
            x_test = torch.from_numpy(X_test)
            reconstructed = global_model(x_test)

            reconstruction_loss = (
                (reconstructed - x_test)
                .square()
                .sum(dim=1)
                .mean()
                .item()
            )
        history.append({
            "Seed": seed,
            "Round": round_id,
            "Reconstruction_loss": reconstruction_loss,
            **metrics,
        })

    return global_model, history