#!/usr/bin/env python3
"""
Command-line interface for CI-VAE.

Usage:
    python -m ci_vae.cli --data_csv path/to/data.csv --train
"""

import argparse
import pandas as pd
import torch
from ci_vae.dataset import MyDataset
from ci_vae.models import CI_VAE
from ci_vae.trainer import Trainer
from ci_vae.utils import plot_residuals, save_residuals, load_residuals


def main(args):
    # Load data (expects a CSV file with features and a label column "Y")
    df = pd.read_csv(args.data_csv)
    train_dataset = MyDataset(df, y_label=["Y"], mode="train")
    test_dataset = MyDataset(df, y_label=["Y"], mode="train")
    
    input_size = df.shape[1] - 1  # one column is the label
    n_classes = df["Y"].nunique()
    
    model = CI_VAE(input_size=input_size, n_classes=n_classes, latent_size=args.latent_size)
    
    trainer = Trainer(
        model,
        train_dataset,
        test_dataset,
        batch_size=args.batch_size,
        reconst_coef=args.reconst_coef,
        kl_coef=args.kl_coef,
        classifier_coef=args.classifier_coef,
    )
    
    if args.train:
        trainer.train(epochs=args.epochs, learning_rate=args.learning_rate)
        torch.save(model.state_dict(), args.model_save_path)
        tracker = {
            "train_tracker": trainer.train_tracker,
            "test_tracker": trainer.test_tracker,
            "test_BCE_tracker": trainer.test_BCE_tracker,
            "test_KLD_tracker": trainer.test_KLD_tracker,
            "test_CEP_tracker": trainer.test_CEP_tracker,
        }
        save_residuals(tracker, filepath=args.residuals_path)
        plot_residuals(
            trainer.train_tracker,
            trainer.test_tracker,
            trainer.test_BCE_tracker,
            trainer.test_KLD_tracker,
            trainer.test_CEP_tracker,
            save_fig_address=args.residuals_fig,
        )
    else:
        model.load_state_dict(torch.load(args.model_save_path, map_location=torch.device("cpu")))
        tracker = load_residuals(filepath=args.residuals_path)
        plot_residuals(
            tracker["train_tracker"],
            tracker["test_tracker"],
            tracker["test_BCE_tracker"],
            tracker["test_KLD_tracker"],
            tracker["test_CEP_tracker"],
            save_fig_address=args.residuals_fig,
        )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CI-VAE Training and Inference")
    parser.add_argument("--data_csv", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--latent_size", type=int, default=20, help="Latent dimension size")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--reconst_coef", type=float, default=100000, help="Reconstruction loss coefficient")
    parser.add_argument("--kl_coef", type=float, default=0.001 * 512, help="KL divergence coefficient")
    parser.add_argument("--classifier_coef", type=float, default=1000, help="Classifier loss coefficient")
    parser.add_argument("--model_save_path", type=str, default="ci_vae_model.pt", help="Path to save the model")
    parser.add_argument("--residuals_path", type=str, default="ci_vae_residuals.pkl", help="Path to save residuals")
    parser.add_argument("--residuals_fig", type=str, default="residuals.pdf", help="Filename for residuals plot")
    parser.add_argument("--train", action="store_true", help="Flag to train the model")
    args = parser.parse_args()
    main(args)
