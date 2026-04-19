"""Conditional WGAN-GP for generating synthetic spectra with target nutrient labels."""
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch():
    if not HAS_TORCH:
        raise ImportError("pip install torch")


class _Gen(nn.Module):
    def __init__(self, z_dim, cond_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + cond_dim, 256), nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, out_dim),
        )

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=1))


class _Disc(nn.Module):
    def __init__(self, in_dim, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, 512), nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x, c):
        return self.net(torch.cat([x, c], dim=1))


def _gp(disc, real, fake, cond, device):
    alpha = torch.rand(real.size(0), 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = disc(interp, cond)
    grads = torch.autograd.grad(d_interp, interp,
                                grad_outputs=torch.ones_like(d_interp),
                                create_graph=True, retain_graph=True)[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def train_cwgan(X, y, n_gen=None, z_dim=64, epochs=500, batch_size=32,
                lr=1e-4, gp_weight=10, n_critic=5, seed=42):
    """
    Train conditional WGAN-GP and generate synthetic spectra.

    X: (n, features) spectra
    y: (n,) target nutrient values
    n_gen: how many synthetic samples to generate (default = len(X))

    Returns (X_syn, y_syn)
    """
    _check_torch()
    torch.manual_seed(seed)
    if n_gen is None:
        n_gen = X.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize
    x_mu, x_sd = X.mean(0), X.std(0) + 1e-8
    y_mu, y_sd = y.mean(), y.std() + 1e-8
    Xn = (X - x_mu) / x_sd
    yn = (y - y_mu) / y_sd

    feat_dim = X.shape[1]
    cond_dim = 1

    gen = _Gen(z_dim, cond_dim, feat_dim).to(device)
    disc = _Disc(feat_dim, cond_dim).to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.9))

    Xt = torch.tensor(Xn, dtype=torch.float32, device=device)
    yt = torch.tensor(yn.reshape(-1, 1), dtype=torch.float32, device=device)
    n = Xt.shape[0]

    for epoch in range(epochs):
        for _ in range(n_critic):
            idx = torch.randint(0, n, (min(batch_size, n),))
            real_x, real_c = Xt[idx], yt[idx]
            z = torch.randn(real_x.size(0), z_dim, device=device)
            fake_x = gen(z, real_c)

            d_real = disc(real_x, real_c).mean()
            d_fake = disc(fake_x.detach(), real_c).mean()
            gp_loss = _gp(disc, real_x, fake_x.detach(), real_c, device)
            d_loss = d_fake - d_real + gp_weight * gp_loss

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

        z = torch.randn(min(batch_size, n), z_dim, device=device)
        idx = torch.randint(0, n, (min(batch_size, n),))
        fake_x = gen(z, yt[idx])
        g_loss = -disc(fake_x, yt[idx]).mean()

        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

    # Generate
    gen.eval()
    with torch.no_grad():
        y_syn_n = torch.FloatTensor(n_gen, 1).uniform_(-2, 2).to(device)
        z = torch.randn(n_gen, z_dim, device=device)
        X_syn_n = gen(z, y_syn_n).cpu().numpy()

    X_syn = X_syn_n * x_sd + x_mu
    y_syn = y_syn_n.cpu().numpy().ravel() * y_sd + y_mu

    return X_syn, y_syn
