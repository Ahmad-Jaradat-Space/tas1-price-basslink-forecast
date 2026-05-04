"""Small Temporal Convolutional Network in PyTorch with quantile heads.

Architecture: stack of dilated 1D convolutions, ReLU, residual; final
linear projection to (n_quantiles,) outputs. Causal padding so the
network sees only past timesteps. Trained with summed pinball loss
across the three quantiles.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))   # left pad only — causal
        return self.conv(x)


class TCN(nn.Module):
    def __init__(self, in_features, hidden=32, n_layers=4, kernel=3,
                 n_quantiles=3, dropout=0.1):
        super().__init__()
        layers = []
        ch_in = in_features
        for i in range(n_layers):
            layers.append(nn.Sequential(
                CausalConv1d(ch_in, hidden, kernel, dilation=2 ** i),
                nn.ReLU(),
                nn.Dropout(dropout),
                CausalConv1d(hidden, hidden, kernel, dilation=2 ** i),
                nn.ReLU(),
            ))
            ch_in = hidden
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(hidden, n_quantiles)
        self.in_features = in_features

    def forward(self, x):
        # x shape (batch, time, features) -> (batch, features, time)
        h = x.transpose(1, 2)
        for layer in self.layers:
            res = layer(h)
            if res.shape == h.shape:
                h = h + res
            else:
                h = res
        h_last = h[:, :, -1]                     # (batch, hidden)
        return self.head(h_last)                 # (batch, n_quantiles)


def pinball_loss_torch(y_pred, y_true, quantiles):
    """y_pred shape (B, K). y_true shape (B, 1). quantiles tensor (K,)."""
    e = y_true - y_pred
    q = quantiles.view(1, -1)
    return torch.maximum(q * e, (q - 1) * e).mean()


def make_windows(X, y, window=12):
    """Sliding windows. Returns (Xw, yw) where Xw shape (n, window, features)
    and yw shape (n, 1) is the target at the end of each window."""
    n = len(X) - window
    Xw = np.stack([X[i:i + window] for i in range(n)], axis=0).astype(np.float32)
    yw = y[window:window + n].astype(np.float32).reshape(-1, 1)
    return Xw, yw


def train_tcn(X_tr, y_tr, X_va, y_va, window=12, hidden=32, epochs=15,
              batch_size=256, lr=1e-3, quantiles=(0.1, 0.5, 0.9),
              seed=0, device="cpu"):
    """Fit a TCN with quantile heads. Returns the fitted model + a
    history dict and the validation quantile predictions."""
    torch.manual_seed(seed)
    Xw_tr, yw_tr = make_windows(X_tr, y_tr, window=window)
    Xw_va, yw_va = make_windows(X_va, y_va, window=window)

    model = TCN(in_features=X_tr.shape[1], hidden=hidden,
                n_layers=4, kernel=3, n_quantiles=len(quantiles))
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    qten = torch.tensor(quantiles, dtype=torch.float32, device=device)

    Xw_tr_t = torch.from_numpy(Xw_tr).to(device)
    yw_tr_t = torch.from_numpy(yw_tr).to(device)
    Xw_va_t = torch.from_numpy(Xw_va).to(device)
    yw_va_t = torch.from_numpy(yw_va).to(device)

    history = {"train": [], "val": []}
    n = len(Xw_tr_t)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        idx = rng.permutation(n)
        running = 0.0
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            xb = Xw_tr_t[b]; yb = yw_tr_t[b]
            opt.zero_grad()
            yh = model(xb)
            loss = pinball_loss_torch(yh, yb, qten)
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(b)
        train_loss = running / n
        model.eval()
        with torch.no_grad():
            yh_v = model(Xw_va_t)
            val_loss = float(pinball_loss_torch(yh_v, yw_va_t, qten).item())
        history["train"].append(train_loss)
        history["val"].append(val_loss)
    model.eval()
    with torch.no_grad():
        Xw_va_pred = model(Xw_va_t).cpu().numpy()
    return model, history, Xw_va_pred


def predict_tcn(model, X, window=12, device="cpu"):
    Xw, _ = make_windows(X, np.zeros(len(X), dtype=np.float32), window=window)
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(Xw).to(device)).cpu().numpy()
    return out
