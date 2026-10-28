"""更新裁剪和高斯噪声机制模拟，不提供正式 DP 会计。"""

import torch


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

    if clip_norm is not None and norm_before > 0:
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


def sample_gaussian_noise(update, noise_std, noise_generator):
    """保留原有逐层采样顺序，包括标准差为零时的采样。"""
    noise = torch.randn(
        update.shape,
        generator=noise_generator,
        device=update.device,
        dtype=update.dtype,
    ) * noise_std
    return noise
