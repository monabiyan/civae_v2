"""
CI-VAE Package

This package provides an implementation of the Class-Informed Variational Autoencoder (CI-VAE)
with utilities for training, latent interpolation, synthetic data generation, and visualization.
"""

from .dataset import MyDataset
from .models import CI_VAE
from .trainer import Trainer
from .utils import (
    plot_residuals,
    save_residuals,
    load_residuals,
    latent_traversal,
    generate_synthetic_data,
    calculate_lower_dimensions,
    plot_lower_dimension,
)
