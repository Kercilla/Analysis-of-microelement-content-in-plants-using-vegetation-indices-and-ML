
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch не установлен. Установите: pip install torch\n"
            "Для CPU: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )

if TORCH_AVAILABLE:

    class SpectralCNN1D(nn.Module):
        
        def __init__(self, n_bands: int, dropout: float = 0.3):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )

            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            f = self.features(x)
            return self.regressor(f).squeeze(-1)

    class SpectralAttention(nn.Module):
        
        def __init__(self, n_bands: int):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(n_bands, n_bands // 4),
                nn.ReLU(),
                nn.Linear(n_bands // 4, n_bands),
                nn.Sigmoid(),
            )

        def forward(self, x):
            weights = self.attention(x.squeeze(1))  # (batch, n_bands)
            return x * weights.unsqueeze(1)          # (batch, 1, n_bands)

    class SpectralCNN1DAttention(nn.Module):
        
        def __init__(self, n_bands: int, dropout: float = 0.3):
            super().__init__()

            self.attention = SpectralAttention(n_bands)

            self.features = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )

            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            x = self.attention(x)
            f = self.features(x)
            return self.regressor(f).squeeze(-1)

        def get_attention_weights(self, x):
            
            with torch.no_grad():
                weights = self.attention.attention(x.squeeze(1))
            return weights.cpu().numpy()

    class SpectralAutoencoder(nn.Module):
        
        def __init__(self, n_bands: int, latent_dim: int = 16):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(n_bands, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, n_bands),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

        def encode(self, x):
            return self.encoder(x)

class CNN1DRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(
        self,
        n_bands: int = 300,
        use_attention: bool = True,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 0.001,
        dropout: float = 0.3,
        patience: int = 30,
        device: str = "auto",
    ):
        self.n_bands = n_bands
        self.use_attention = use_attention
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.patience = patience
        self.device = device

    def _get_device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def fit(self, X, y):
        check_torch()
        device = self._get_device()

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        mask = np.isfinite(y)
        X_t = torch.FloatTensor(X_scaled[mask]).unsqueeze(1).to(device)  # (n, 1, bands)
        y_t = torch.FloatTensor(y[mask]).to(device)

        n_bands = X_t.shape[2]
        if self.use_attention:
            self.model_ = SpectralCNN1DAttention(n_bands, self.dropout).to(device)
        else:
            self.model_ = SpectralCNN1D(n_bands, self.dropout).to(device)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_loss = np.inf
        patience_counter = 0
        best_state = None

        self.model_.train()
        self.train_losses_ = []

        for epoch in range(self.epochs):
            epoch_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = self.model_(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            self.train_losses_.append(avg_loss)
            scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state:
            self.model_.load_state_dict(best_state)

        self.model_.eval()
        return self

    def predict(self, X):
        check_torch()
        device = self._get_device()

        X_scaled = self.scaler_.transform(X)
        X_t = torch.FloatTensor(X_scaled).unsqueeze(1).to(device)

        with torch.no_grad():
            y_pred = self.model_(X_t).cpu().numpy()

        return y_pred

    def get_attention_weights(self, X):
        
        if not self.use_attention:
            raise ValueError("Attention недоступен (use_attention=False)")

        check_torch()
        device = self._get_device()

        X_scaled = self.scaler_.transform(X)
        X_t = torch.FloatTensor(X_scaled).unsqueeze(1).to(device)

        return self.model_.get_attention_weights(X_t)

class AutoencoderFeatureExtractor(BaseEstimator):
    
    def __init__(
        self,
        n_bands: int = 300,
        latent_dim: int = 16,
        epochs: int = 200,
        lr: float = 0.001,
        batch_size: int = 32,
    ):
        self.n_bands = n_bands
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

    def fit(self, X, y=None):
        check_torch()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        X_t = torch.FloatTensor(X_scaled).to(device)
        dataset = TensorDataset(X_t, X_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = SpectralAutoencoder(X.shape[1], self.latent_dim).to(device)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model_.train()
        for epoch in range(self.epochs):
            for X_batch, _ in loader:
                optimizer.zero_grad()
                X_recon = self.model_(X_batch)
                loss = criterion(X_recon, X_batch)
                loss.backward()
                optimizer.step()

        self.model_.eval()
        self.device_ = device
        return self

    def transform(self, X):
        check_torch()
        X_scaled = self.scaler_.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(self.device_)

        with torch.no_grad():
            features = self.model_.encode(X_t).cpu().numpy()

        return features

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
