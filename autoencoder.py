import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, List

# Configure internal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed: int = 42) -> None:
    """Ensures reproducible model initialization and operations."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AutoEncoder(nn.Module):
    """
    Symmetric Bottleneck AutoEncoder for tabular data.
    Architecture: Input (5 features) -> 128 -> 64 -> 16 (latent) -> 64 -> 128 -> Output (5)
    """
    def __init__(self, input_dim: int = 5, hidden_dims: List[int] = [128, 64], latent_dim: int = 16):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        
        # Encoder Construction
        enc_layers = []
        in_features = input_dim
        for h_dim in hidden_dims:
            enc_layers.append(nn.Linear(in_features, h_dim))
            enc_layers.append(nn.ReLU())
            in_features = h_dim
        enc_layers.append(nn.Linear(in_features, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        
        # Decoder Construction
        dec_layers = []
        in_features = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_features, h_dim))
            dec_layers.append(nn.ReLU())
            in_features = h_dim
        dec_layers.append(nn.Linear(in_features, input_dim)) # Identity activation for regression
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.size(1) != self.input_dim:
            raise ValueError(f"Dimensionality mismatch: Expected {self.input_dim} features, got {x.size(1)}.")
        
        x = x.to(torch.float32)
        z = self.encoder(x)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, z

def train_autoencoder(model: AutoEncoder, train_loader: torch.utils.data.DataLoader, 
                      epochs: int = 100, lr: float = 1e-3, device: str = 'cpu') -> List[float]:
    """Trains the AutoEncoder using Adam and MSELoss, returning historical losses."""


    import torch.optim as optim
    
    model.to(device)
    model.train()
    
    # Use the local import
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    criterion = nn.MSELoss()
    history = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch_x = batch[0].to(device, dtype=torch.float32)
            optimizer.zero_grad()
            x_hat, _ = model(batch_x)
            loss = criterion(x_hat, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        history.append(epoch_loss / len(train_loader.dataset))
    return history

def evaluate_anomalies(model: AutoEncoder, data_tensor: torch.Tensor, 
                       threshold_strategy: str = 'percentile', percentile_val: float = 95.0, 
                       device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray, float]:
    """Calculates sample reconstruction errors, establishes threshold, and flags anomalies."""
    model.to(device)
    model.eval()
    if data_tensor.size(1) != model.input_dim:
         raise ValueError(f"Dimensionality mismatch: Expected {model.input_dim} features, got {data_tensor.size(1)}.")
         
    data_tensor = data_tensor.to(device, dtype=torch.float32)
    with torch.no_grad():
        x_hat, _ = model(data_tensor)
        errors = torch.mean((data_tensor - x_hat) ** 2, dim=1).cpu().numpy()
        
    if threshold_strategy == 'percentile':
        threshold = np.percentile(errors, percentile_val)
    elif threshold_strategy == '3sigma':
        threshold = np.mean(errors) + 3 * np.std(errors)
    else:
        raise ValueError("Invalid threshold_strategy. Use 'percentile' or '3sigma'.")
        
    anomalies = errors > threshold
    return anomalies, errors, threshold