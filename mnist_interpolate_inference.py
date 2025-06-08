#!/usr/bin/env python3
"""Run inference with a trained CI-VAE on MNIST.

This script loads two digit '3' images from the MNIST test set, encodes them
with a trained CI-VAE model, linearly interpolates between the two latent
representations and saves the decoded transition as an animated GIF.
"""

import argparse
import os
import random
import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from ci_vae.models import CI_VAE
from ci_vae.utils import sample_data_on_line, save_gif


def load_model(model_path: str, device: torch.device) -> CI_VAE:
    """Load a CI-VAE model from ``model_path``."""
    state = torch.load(model_path, map_location=device)
    input_size = state["input_size"]
    n_classes = state["n_classes"]
    latent_size = state["latent_size"]
    model = CI_VAE(input_size=input_size, n_classes=n_classes, latent_size=latent_size)
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()
    return model


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = MNIST(root="./data", train=False, download=True, transform=ToTensor())
    indices = [i for i, (_, label) in enumerate(dataset) if label == 3]
    if len(indices) < 2:
        raise RuntimeError("Need at least two digit '3' images in the dataset")
    idx1, idx2 = random.sample(indices, 2)
    img1, _ = dataset[idx1]
    img2, _ = dataset[idx2]
    img1 = img1.view(-1).unsqueeze(0).to(device)
    img2 = img2.view(-1).unsqueeze(0).to(device)

    model = load_model(args.model, device)
    with torch.no_grad():
        mu1, _ = model.encode(img1)
        mu2, _ = model.encode(img2)
        line = sample_data_on_line(mu1.squeeze(), mu2.squeeze(), args.steps)
        decoded = model.decode(line.to(device)).cpu().numpy()

    frames = np.array([decoded[i].reshape(28, 28) for i in range(args.steps)])
    gif_path = os.path.join(args.output_dir, "digit3_transition.gif")
    save_gif(frames, file_path_root=os.path.join(args.output_dir, ""), indicator="digit3_transition", speed=3)
    print(f"Saved interpolation GIF to {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CI-VAE MNIST interpolation demo")
    parser.add_argument("--model", type=str, required=True, help="Path to trained CI-VAE model file")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory to save the GIF")
    parser.add_argument("--steps", type=int, default=20, help="Number of interpolation steps")
    main(parser.parse_args())

