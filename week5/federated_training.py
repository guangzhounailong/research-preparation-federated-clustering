from time import perf_counter

import numpy as np
import torch

import config
from client import train_client
from evaluation import evaluate_clustering, run_kmeans
from models import Autoencoder, VAE, extract_features
from server import fedavg,fedavg_with_noise


def run_federated_training(
    X_train,
    X_test,
    y_test,
    train_indices,
    seed,
    model_class = Autoencoder,
    clip_norm = None,
    noise_multiplier=0.0
):
    #初始化全局模型
    np.random.seed(seed)
    torch.manual_seed(seed)
    global_model = model_class()

    history = []
    training_time = 0.0
    noise_generator = torch.Generator().manual_seed(seed + 1000)

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
        norm_metrics = {}

        if clip_norm is None:
            global_state = fedavg(client_states, client_sizes)
        else:
            global_state, norm_records = fedavg_with_noise(
                client_states=client_states,
                client_sizes=client_sizes,
                global_state=global_model.state_dict(),
                clip_norm=clip_norm,
                noise_multiplier=noise_multiplier,
                noise_generator=noise_generator,
            )
            norm_metrics = {
                "Norm_before_mean": np.mean([row["Norm_before"] for row in norm_records]),
                "Norm_after_mean": np.mean([row["Norm_after"] for row in norm_records]),
                }
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
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + round_id)
            x_test = torch.from_numpy(X_test)

            if isinstance(global_model, VAE):
                reconstructed, mu, logvar = global_model(x_test)
                kl_loss = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
            else:
                reconstructed = global_model(x_test)
                kl_loss = x_test.new_tensor(0.0)

            reconstruction_loss = (
                (reconstructed - x_test).square().sum(dim=1).mean().item()
            )
        
        history.append({
            "Seed": seed,
            "Round": round_id,
            "Reconstruction_loss": reconstruction_loss,
            "KL_loss": kl_loss.item(),
            **norm_metrics,
            **metrics,
        })

    return global_model, history
