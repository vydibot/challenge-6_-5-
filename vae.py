import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VAE(nn.Module):
    """
    Continuous Variational AutoEncoder (beta-VAE) for tabular socioeconomic metrics.
    Maps inputs dynamically to a localized normal distribution space.
    """
    def __init__(self, input_dim: int = 5, hidden_dims: List[int] = [128, 64], latent_dim: int = 16):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        
        # Encoder Body
        enc_layers = []
        in_features = input_dim
        for h_dim in hidden_dims:
            enc_layers.append(nn.Linear(in_features, h_dim))
            enc_layers.append(nn.ReLU())
            in_features = h_dim
        self.encoder_body = nn.Sequential(*enc_layers)
        
        # Parallel Distribution Spaces
        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)
        
        # Decoder Engine
        dec_layers = []
        in_features = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_features, h_dim))
            dec_layers.append(nn.ReLU())
            in_features = h_dim
        dec_layers.append(nn.Linear(in_features, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Stochastic trick: z = mu + standard_deviation * noise_epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.size(1) != self.input_dim:
            raise ValueError(f"Dimensionality mismatch: Expected {self.input_dim} features, got {x.size(1)}.")
        x = x.to(torch.float32)
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar

def vae_loss_fn(x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes specialized beta-VAE objective matrix optimization target."""
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='none').sum(dim=1).mean()
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, kld_loss

def train_vae(model: VAE, train_loader: torch.utils.data.DataLoader, 
              epochs: int = 100, lr: float = 1e-3, beta: float = 1.0, 
              device: str = 'cpu') -> Dict[str, List[float]]:
    """Trains VAE while logging both Reconstruction and Structural Divergence objectives."""
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'total_loss': [], 'recon_loss': [], 'kld_loss': []}
    
    for epoch in range(epochs):
        epoch_total, epoch_recon, epoch_kld = 0.0, 0.0, 0.0
        for batch in train_loader:
            batch_x = batch[0].to(device, dtype=torch.float32)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(batch_x)
            total_loss, recon_loss, kld_loss = vae_loss_fn(x_hat, batch_x, mu, logvar, beta)
            total_loss.backward()
            optimizer.step()
            
            epoch_total += total_loss.item() * batch_x.size(0)
            epoch_recon += recon_loss.item() * batch_x.size(0)
            epoch_kld += kld_loss.item() * batch_x.size(0)
            
        n_samples = len(train_loader.dataset)
        history['total_loss'].append(epoch_total / n_samples)
        history['recon_loss'].append(epoch_recon / n_samples)
        history['kld_loss'].append(epoch_kld / n_samples)
        
        if (epoch_kld / n_samples) < 1e-4:
            logging.warning(f"Epoch {epoch+1}: Low KL detected. Threat of full posterior collapse.")
            
    return history