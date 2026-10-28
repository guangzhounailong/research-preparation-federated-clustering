"""Simulate update clipping and Gaussian noise without formal DP accounting."""

import torch


def clip_client_update(local_state, global_state, clip_norm):
    # Compute parameter changes from this round of local training.
    update = {
        name: local_state[name] - global_state[name]
        for name in global_state
    }

    # Compute the joint L2 norm of updates across all layers.
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
    """Preserve layer-wise sampling order, including draws with zero standard deviation."""
    noise = torch.randn(
        update.shape,
        generator=noise_generator,
        device=update.device,
        dtype=update.dtype,
    ) * noise_std
    return noise
