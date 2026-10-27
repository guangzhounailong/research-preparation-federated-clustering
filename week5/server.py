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

def clip_client_update(local_state, global_state, clip_norm):
    # 计算本轮本地训练带来的参数变化
    update = {
        name: local_state[name] - global_state[name]
        for name in global_state
    }

    # 把所有层的更新合起来，计算一个总长度
    norm_before = torch.sqrt(sum(
        value.square().sum()
        for value in update.values()
    )).item()

    if norm_before > 0:
        scale = min(1.0, clip_norm / norm_before)
    else:
        scale = 1.0

    clipped_update = {
        name: value * scale
        for name, value in update.items()
    }

    norm_after = torch.sqrt(sum(
        value.square().sum()
        for value in clipped_update.values()
    )).item()

    return clipped_update, norm_before, norm_after

def fedavg_with_noise(
    client_states,
    client_sizes,
    global_state,
    clip_norm,
    noise_multiplier,
    noise_generator,
):
    clipped_updates = []
    norm_records = []

    # 1. 每个客户端分别计算、裁剪更新
    for state in client_states:
        update, norm_before, norm_after = clip_client_update(
            state,
            global_state,
            clip_norm,
        )

        clipped_updates.append(update)
        norm_records.append({
            "Norm_before": norm_before,
            "Norm_after": norm_after,
        })

    # 2. 按客户端样本数，对裁剪后的更新加权平均
    averaged_update = fedavg(clipped_updates, client_sizes)

    # 3. 噪声标准差 = 噪声倍数 × 裁剪上限
    noise_std =  noise_multiplier*clip_norm

    # 4. 旧参数 + 平均更新 + 噪声
    new_state = {}

    for name, update in averaged_update.items():
        noise = torch.randn(
            update.shape,
            generator=noise_generator,
            device=update.device,
            dtype=update.dtype,
        ) * noise_std

        new_state[name] = global_state[name] + update + noise

    return new_state, norm_records