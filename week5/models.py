"""Define AE/VAE networks and use VAE posterior means for clustering."""

import torch
from torch import nn

from .config import DEFAULT_CONFIG


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=None, *, config=DEFAULT_CONFIG):
        if latent_dim is None:
            latent_dim = config.latent_dim
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self, latent_dim=None, *, config=DEFAULT_CONFIG):
        if latent_dim is None:
            latent_dim = config.latent_dim
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(config.hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
            nn.Sigmoid(),
        )

    def distribution(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def encode(self, x):
        mu, _ = self.distribution(x)
        return mu

    def forward(self, x):
        mu, logvar = self.distribution(x)

        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
