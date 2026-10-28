"""Train models and extract features for centralized and client workflows."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import DEFAULT_CONFIG
from .models import Autoencoder, VAE


def train_model(model, X_train, seed, epochs=None, *, config=DEFAULT_CONFIG):
    if epochs is None:
        epochs = config.epochs

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train)),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate
    )

    for _ in range(epochs):
        model.train()
        for (x,) in loader:
            optimizer.zero_grad()

            if isinstance(model, VAE):
                reconstructed, mu, logvar = model(x)
                kl_loss = -0.5 * (
                    1 + logvar - mu.square() - logvar.exp()
                ).sum(dim=1).mean()
            else:
                reconstructed = model(x)
                kl_loss = x.new_tensor(0.0)

            reconstruction_loss = (
                (reconstructed - x).square().sum(dim=1).mean()
            )
            loss = reconstruction_loss + config.beta * kl_loss
            loss.backward()
            optimizer.step()

    return model


def extract_features(model, X):
    model.eval()
    with torch.no_grad():
        features = model.encode(torch.from_numpy(X))
        # Apply per-sample L2 normalization to AE clustering features only; leave reconstruction training unchanged.
        if isinstance(model, Autoencoder):
            features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features.numpy()
