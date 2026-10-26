import torch

def fedavg(client_states, client_sizes):
    #计算样本总数
    total_samples = sum(client_sizes)

    global_state = {name: torch.zeros_like(param) for name, param in client_states[0].items()}

    for state, size in zip(client_states, client_sizes):
        weight = size / total_samples

        for name in global_state:
            global_state[name] += state[name] * weight

    return global_state