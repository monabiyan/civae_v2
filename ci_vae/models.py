import torch
from torch import nn
from typing import Tuple


def block(in_features: int, out_features: int, dropout_rate: float, momentum: float) -> nn.Sequential:
    """
    Creates a basic neural network block: Linear -> ReLU -> BatchNorm -> Dropout.
    """
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.BatchNorm1d(out_features, momentum=momentum),
        nn.Dropout(p=dropout_rate)
    )


class CI_VAE(nn.Module):
    """
    Class-Informed Variational Autoencoder (CI-VAE).

    This model consists of an encoder that maps input data to latent distribution parameters,
    a decoder that reconstructs the data from the latent space, and a classifier layer on the latent space
    to enforce class separation.
    """
    def __init__(self, input_size: int, n_classes: int, latent_size: int,
                 dropout_rate: float = 0.05, momentum: float = 0.2):
        super().__init__()
        self.latent_size = latent_size
        self.input_size = input_size

        # Intermediate layer sizes (adjustable)
        medium_layer2 = 20
        medium_layer = 20
        medium_layer3 = 10

        # Build Encoder: produces 2*latent_size outputs (for mu and logvar)
        encoder_layers = [
            block(self.input_size, medium_layer2, dropout_rate, momentum),
            block(medium_layer2, medium_layer, dropout_rate, momentum)
        ]
        for _ in range(6):
            encoder_layers.append(block(medium_layer, medium_layer, dropout_rate, momentum))
        encoder_layers.append(block(medium_layer, medium_layer3, dropout_rate, momentum))
        encoder_layers.append(nn.Linear(medium_layer3, latent_size * 2))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build Decoder: reconstructs input from latent vector
        decoder_layers = [
            block(latent_size, medium_layer3, dropout_rate, momentum),
            block(medium_layer3, medium_layer, dropout_rate, momentum)
        ]
        for _ in range(6):
            decoder_layers.append(block(medium_layer, medium_layer, dropout_rate, momentum))
        decoder_layers.extend([
            block(medium_layer, medium_layer2, dropout_rate, momentum),
            nn.Linear(medium_layer2, input_size)
        ])
        self.decoder = nn.Sequential(*decoder_layers)

        # Linear classifier on latent space for enforcing class separation
        self.classifier = nn.Sequential(
            nn.Linear(latent_size, n_classes),
            nn.Dropout(p=0.80)
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample from N(mu, exp(logvar)) using eps ~ N(0, 1).
        """
        if self.training:
            std = torch.exp(logvar / 2) + 1e-7
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes input x into latent mean and log variance.
        """
        x = x.view(-1, self.input_size)
        mu_logvar = self.encoder(x)
        mu_logvar = mu_logvar.view(-1, 2, self.latent_size)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent vector z to a reconstruction of x.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: returns reconstructed input, classifier output, latent mean, log variance, and latent vector.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y_hat = self.classifier(z)
        x_hat = self.decode(z)
        return x_hat, y_hat, mu, logvar, z

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Samples new data points from the latent prior and decodes them.
        """
        device = next(self.parameters()).device
        z = torch.randn((n_samples, self.latent_size), device=device)
        return self.decode(z)
