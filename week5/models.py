import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=config.LATENT_DIM):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, config.INPUT_DIM),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self, latent_dim=config.LATENT_DIM):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(config.HIDDEN_DIM, latent_dim)
        self.fc_logvar = nn.Linear(config.HIDDEN_DIM, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, config.INPUT_DIM),
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


def train_model(model, X_train, seed):
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train)),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE
    )

    for _ in range(config.EPOCHS):
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
            loss = reconstruction_loss + config.BETA * kl_loss
            loss.backward()
            optimizer.step()

    return model


def extract_features(model, X):
    model.eval()
    with torch.no_grad():
        features = model.encode(torch.from_numpy(X))
    return features.numpy()
