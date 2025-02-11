import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Tuple, List
import numpy as np


class Trainer:
    """
    Trainer for the CI-VAE model.

    Args:
        model (nn.Module): The CI-VAE model.
        train_dataset: Training dataset.
        test_dataset: Test dataset.
        batch_size (int): Batch size for training.
        device (torch.device): Device for training.
        reconst_coef (float): Weight for reconstruction loss.
        kl_coef (float): Weight for KL divergence loss.
        classifier_coef (float): Weight for classifier loss.
        random_seed (int): For reproducibility.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        test_dataset,
        batch_size: int = 512,
        device: torch.device = None,
        reconst_coef: float = 100000,
        kl_coef: float = 0.001 * 512,
        classifier_coef: float = 1000,
        random_seed: int = 0,
    ):
        self.model = model
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.testloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
        self.reconst_coef = reconst_coef
        self.kl_coef = kl_coef
        self.classifier_coef = classifier_coef
        self.random_seed = random_seed
        self.optimizer = Adam(self.model.parameters(), lr=1e-4, weight_decay=2e-5)

        self.mae_loss = nn.L1Loss().to(self.device)
        self.cross_entropy_loss = nn.CrossEntropyLoss().to(self.device)

        # Trackers for loss components
        self.train_tracker = []
        self.test_tracker = []
        self.test_BCE_tracker = []
        self.test_KLD_tracker = []
        self.test_CEP_tracker = []

    def loss_function(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes reconstruction (BCE), KL divergence, and classifier (cross-entropy) losses.
        """
        x = x.view(-1, self.model.input_size)
        BCE = self.mae_loss(x_hat, x)
        CEP = self.cross_entropy_loss(y_hat, y)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD / self.batch_size
        total_loss = BCE * self.reconst_coef + KLD * self.kl_coef + CEP * self.classifier_coef
        return BCE * self.reconst_coef, KLD * self.kl_coef, CEP * self.classifier_coef, total_loss

    def train_epoch(self) -> float:
        """
        Performs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        for x, y in self.trainloader:
            x = x.to(self.device)
            y = y.to(self.device).view(-1).long()
            self.optimizer.zero_grad()
            x_hat, y_hat, mu, logvar, _ = self.model(x)
            bce, kld, cep, loss = self.loss_function(x_hat, x, y_hat, y, mu, logvar)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.trainloader)
        self.train_tracker.append(avg_loss)
        return avg_loss

    def test_epoch(self) -> Tuple[float, float, float, float, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Evaluates the model on the test set.
        """
        self.model.eval()
        test_BCE_loss = 0.0
        test_KLD_loss = 0.0
        test_CEP_loss = 0.0
        test_total_loss = 0.0

        means, logvars = [], []
        true_labels, inputs = [], []
        pred_labels, reconstructed = [], []
        zs = []

        with torch.no_grad():
            for x, y in self.testloader:
                x = x.to(self.device)
                y = y.to(self.device).view(-1).long()
                x_hat, y_hat, mu, logvar, z = self.model(x)
                bce, kld, cep, loss = self.loss_function(x_hat, x, y_hat, y, mu, logvar)
                test_total_loss += loss.item()
                test_BCE_loss += bce.item()
                test_KLD_loss += kld.item()
                test_CEP_loss += cep.item()

                means.append(mu.detach())
                logvars.append(logvar.detach())
                true_labels.append(y.detach())
                inputs.append(x.detach())
                pred_labels.append(y_hat.detach())
                reconstructed.append(x_hat.detach())
                zs.append(z.detach())

        n_batches = len(self.testloader)
        test_total_loss /= n_batches
        test_BCE_loss /= n_batches
        test_KLD_loss /= n_batches
        test_CEP_loss /= n_batches

        self.test_tracker.append(test_total_loss)
        self.test_BCE_tracker.append(test_BCE_loss)
        self.test_KLD_tracker.append(test_KLD_loss)
        self.test_CEP_tracker.append(test_CEP_loss)

        return (
            test_BCE_loss,
            test_KLD_loss,
            test_CEP_loss,
            test_total_loss,
            means,
            logvars,
            true_labels,
            inputs,
            pred_labels,
            reconstructed,
            zs,
        )

    def train(self, epochs: int, learning_rate: float = 1e-4) -> None:
        """
        Trains the model for a specified number of epochs.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            bce_loss, kld_loss, cep_loss, test_loss, *_ = self.test_epoch()
            print(
                f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}, "
                f"BCE = {bce_loss:.6f}, KLD = {kld_loss:.6f}, CEP = {cep_loss:.6f}"
            )
