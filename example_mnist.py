#!/usr/bin/env python3
"""
Example: Using CI-VAE on MNIST for Class-Informed Interpolation and Synthetic Data Generation

This script demonstrates:
  - Loading MNIST and converting images into a pandas DataFrame.
  - Training a CI-VAE model.
  - Reconstructing test images.
  - Interpolating between two latent codes from the same class.
  - Generating synthetic data via latent traversal.
  - Visualizing latent embeddings using t-SNE, UMAP, and PCA.
  - Creating animated GIFs of the interpolation sequence and synthetic data samples.

All outputs are saved in an output folder automatically.
Ensure that the "ci_vae" package (with modules: models, dataset, trainer, utils) is available.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Import MNIST dataset and transform from torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# Import CI-VAE package components
from ci_vae import CI_VAE, MyDataset, Trainer
from ci_vae.utils import (
    plot_residuals,
    calculate_lower_dimensions,
    plot_lower_dimension,
    sample_data_on_line,
    generate_synthetic_data,
    save_gif,
)

# Set output directory (automatically created if not exists)
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_mnist_dataframe(train: bool = True, max_samples: int = None) -> pd.DataFrame:
    """
    Loads MNIST and returns a DataFrame with 784 pixel features and a label column "Y".
    """
    dataset = MNIST(root='./data', train=train, download=True, transform=ToTensor())
    images = []
    labels = []
    num_samples = len(dataset) if max_samples is None else max_samples
    for i in range(num_samples):
        img, label = dataset[i]
        img_np = img.view(-1).numpy()
        images.append(img_np)
        labels.append(label)
    images = np.array(images)
    df = pd.DataFrame(images, columns=[f"pixel_{i}" for i in range(images.shape[1])])
    df["Y"] = labels
    return df

def main():
    # -------------------------------
    # 1. Load MNIST and Create Datasets
    # -------------------------------
    print("Loading MNIST dataset...")
    train_df = load_mnist_dataframe(train=True, max_samples=10000)
    test_df  = load_mnist_dataframe(train=False, max_samples=2000)
    
    train_dataset = MyDataset(train_df, y_label=["Y"], mode="train")
    test_dataset  = MyDataset(test_df, y_label=["Y"], mode="train")
    
    # -------------------------------
    # 2. Initialize the CI-VAE Model and Trainer
    # -------------------------------
    input_size = train_df.shape[1] - 1  # subtract label column
    n_classes = 10  # MNIST has 10 classes (0-9)
    latent_size = 20

    model = CI_VAE(input_size=input_size, n_classes=n_classes, latent_size=latent_size)
    
    trainer = Trainer(
        model,
        train_dataset,
        test_dataset,
        batch_size=512,
        reconst_coef=100000,
        kl_coef=0.001 * 512,
        classifier_coef=10000
    )
    
    # -------------------------------
    # 3. Train the Model
    # -------------------------------
    print("Training CI-VAE model on MNIST...")
    trainer.train(epochs=1000, learning_rate=1e-5)
    
    plot_residuals(
        trainer.train_tracker,
        trainer.test_tracker,
        trainer.test_BCE_tracker,
        trainer.test_KLD_tracker,
        trainer.test_CEP_tracker,
        save_fig_address=os.path.join(OUTPUT_DIR, "mnist_residuals.pdf")
    )
    
    # -------------------------------
    # 4. Reconstruction on Test Data
    # -------------------------------
    print("Performing reconstruction on test data...")
    model.eval()
    with torch.no_grad():
        for x, y in trainer.testloader:
            x = x.to(trainer.device)
            x_hat, y_hat, mu, logvar, z = model(x)
            break  # use the first batch only
    
    x_np = x.cpu().numpy()
    x_hat_np = x_hat.cpu().numpy()
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axes[0, i].imshow(x_np[i].reshape(28, 28), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")
        axes[1, i].imshow(x_hat_np[i].reshape(28, 28), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title("Reconstructed")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mnist_reconstruction.pdf"))
    plt.show()
    
    # -------------------------------
    # 5. Latent Space Interpolation (Class-Informed) and GIF creation
    # -------------------------------
    print("Performing latent space interpolation for class 3...")
    indices_class3 = test_df.index[test_df["Y"] == 3].tolist()
    if len(indices_class3) < 2:
        print("Not enough samples of class 3 for interpolation.")
    else:
        idx1, idx2 = random.sample(indices_class3, 2)
        sample1 = torch.tensor(test_df.drop(columns=["Y"]).iloc[idx1].values, dtype=torch.float32).to(trainer.device).unsqueeze(0)
        sample2 = torch.tensor(test_df.drop(columns=["Y"]).iloc[idx2].values, dtype=torch.float32).to(trainer.device).unsqueeze(0)
        
        mu1, _ = model.encode(sample1)
        mu2, _ = model.encode(sample2)
        
        num_steps = 20  # Increase steps for smoother transition
        latent_line = sample_data_on_line(mu1.squeeze(), mu2.squeeze(), num_steps)
        decoded_line = model.decoder(latent_line.to(trainer.device)).detach().cpu().numpy()
        
        # Reshape each decoded image from (784,) to (28,28)
        interp_images = np.array([decoded_line[i].reshape(28, 28) for i in range(num_steps)])
        
        # Plot interpolation sequence
        fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.5, 2))
        for i in range(num_steps):
            axes[i].imshow(interp_images[i], cmap="gray")
            axes[i].axis("off")
        plt.suptitle("Latent Space Interpolation (Class 3)")
        plt.savefig(os.path.join(OUTPUT_DIR, "mnist_interpolation_class3.pdf"))
        plt.show()
        
        # Save GIF for the interpolation sequence
        save_gif(interp_images, file_path_root=OUTPUT_DIR + "/", indicator="interpolation_class3", speed=3)
        print("Interpolation GIF saved as 'interpolation_class3.gif' in the output folder.")
    
    # -------------------------------
    # 6. Synthetic Data Generation via Latent Traversal and GIF creation
    # -------------------------------
    print("Generating synthetic data via latent traversal...")
    synthetic_data = generate_synthetic_data(trainer, num_additional_data=200, images_per_traversal=10)
    
    # Plot a subset (first 10 images) of the synthetic data
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(10):
        axes[i // 5, i % 5].imshow(synthetic_data[i].reshape(28, 28), cmap="gray")
        axes[i // 5, i % 5].axis("off")
    plt.suptitle("Synthetic Data Samples")
    plt.savefig(os.path.join(OUTPUT_DIR, "mnist_synthetic_samples.pdf"))
    plt.show()
    
    # Save GIF for synthetic data using the first 10 samples as frames
    synthetic_subset = np.array([synthetic_data[i].reshape(28, 28) for i in range(10)])
    save_gif(synthetic_subset, file_path_root=OUTPUT_DIR + "/", indicator="synthetic_samples", speed=3)
    print("Synthetic data GIF saved as 'synthetic_samples.gif' in the output folder.")
    
    # -------------------------------
    # 7. Latent Space Visualization (Lower-Dimensional Projections)
    # -------------------------------
    print("Calculating and plotting lower-dimensional embeddings...")
    latent_all = []
    labels_all = []
    with torch.no_grad():
        for x, y in trainer.testloader:
            x = x.to(trainer.device)
            _, _, _, _, z = model(x)
            latent_all.append(z.cpu())
            labels_all.append(y.cpu().numpy())
    latent_all = torch.cat(latent_all, dim=0)
    labels_all = np.concatenate(labels_all, axis=0)
    
    tsne_proj, umap_proj, pca_proj, labels_subset = calculate_lower_dimensions(latent_all, labels_all, N=2000)
    
    plot_lower_dimension(tsne_proj, labels_subset, projection="2d", save_str=os.path.join(OUTPUT_DIR, "mnist_tsne_2d.pdf"))
    plot_lower_dimension(umap_proj, labels_subset, projection="2d", save_str=os.path.join(OUTPUT_DIR, "mnist_umap_2d.pdf"))
    plot_lower_dimension(pca_proj, labels_subset, projection="2d", save_str=os.path.join(OUTPUT_DIR, "mnist_pca_2d.pdf"))
    
    print("End-to-end CI-VAE demo on MNIST completed. All outputs are saved in the 'output' folder.")

if __name__ == "__main__":
    main()
