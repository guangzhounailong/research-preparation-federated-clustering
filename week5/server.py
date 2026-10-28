"""Aggregate client model parameters or updates by sample count."""

import torch

from .privacy import clip_client_update, sample_gaussian_noise


def fedavg(client_states, client_sizes):
    # Compute the total sample count.
    total_samples = sum(client_sizes)

    global_state = {name: torch.zeros_like(param) for name, param in client_states[0].items()}

    for state, size in zip(client_states, client_sizes):
        weight = size / total_samples

        for name in global_state:
            global_state[name] += state[name] * weight

    return global_state


def fedavg_with_noise(
    client_states,
    client_sizes,
    global_state,
    clip_norm,
    noise_multiplier,
    noise_generator,
    noise_std = None
):
    clipped_updates = []
    norm_records = []

    # 1. Compute and clip each client update separately.
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

    # 2. Average clipped updates, weighted by client sample counts.
    averaged_update = fedavg(clipped_updates, client_sizes)

    # 3. Noise standard deviation = noise multiplier * clipping threshold.
    if noise_std is None:
        noise_std = noise_multiplier * clip_norm

    # 4. Previous parameters + average update + noise.
    new_state = {}

    for name, update in averaged_update.items():
        noise = sample_gaussian_noise(update, noise_std, noise_generator)

        new_state[name] = global_state[name] + update + noise

    return new_state, norm_records
