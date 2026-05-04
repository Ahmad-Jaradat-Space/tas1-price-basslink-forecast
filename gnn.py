"""Graph neural network over the NEM regional graph.

Five nodes (NSW1, QLD1, SA1, TAS1, VIC1) connected by interconnectors.
At each timestep we have a node feature vector (recent prices + demand
+ time-of-day) per region. The model is a two-layer GCN: each layer
aggregates neighbouring node features through a normalised adjacency,
then a per-node MLP head predicts next-period TAS1 RRP from the TAS1
node embedding.

For a 5-node graph there's no point pulling in PyTorch Geometric —
explicit dense matrix multiplications are clearer and just as fast.
"""

import numpy as np
import torch
import torch.nn as nn


def normalise_adj(A):
    """Symmetric normalisation: D^{-1/2} A D^{-1/2}. Self-loops should be
    in A already."""
    d = A.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-9))
    return d_inv_sqrt @ A @ d_inv_sqrt


class GCNRegional(nn.Module):
    def __init__(self, n_nodes, in_features, hidden=24, out_features=1,
                 target_node=3):
        super().__init__()
        self.n_nodes = n_nodes
        self.target_node = target_node
        self.W1 = nn.Linear(in_features, hidden)
        self.W2 = nn.Linear(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, X, A_norm):
        # X shape (batch, n_nodes, features); A_norm shape (n_nodes, n_nodes)
        h = torch.relu(self.W1(X))
        h = torch.einsum("ij,bjf->bif", A_norm, h)
        h = torch.relu(self.W2(h))
        h = torch.einsum("ij,bjf->bif", A_norm, h)
        # take the embedding at the target node and project
        return self.head(h[:, self.target_node, :])


def build_node_features(graph_df, regions, lags=(1, 2, 48)):
    """Per-region feature blocks: for each region, [rrp_t, rrp_{t-1},
    rrp_{t-2}, rrp_{t-48}]. Returns a tensor shape (T, n_nodes, n_feats)
    and a target vector of next-period TAS1 RRP."""
    cols = [f"rrp_{r.lower()}" for r in regions]
    P = graph_df[cols].values.astype(np.float32)   # (T, n_nodes)
    T, N = P.shape
    F = 1 + len(lags)
    X = np.zeros((T, N, F), dtype=np.float32)
    X[:, :, 0] = P
    for k, L in enumerate(lags, start=1):
        X[L:, :, k] = P[:-L]
    # target = next-period TAS1
    tas_idx = regions.index("TAS1")
    y = np.zeros(T, dtype=np.float32)
    y[:-1] = P[1:, tas_idx]
    valid = max(lags)
    return X[valid:-1], y[valid:-1]


def train_gcn(X, y, A_norm, tr, va, hidden=24, epochs=30,
              batch_size=512, lr=1e-3, seed=0, device="cpu"):
    torch.manual_seed(seed)
    model = GCNRegional(
        n_nodes=A_norm.shape[0], in_features=X.shape[2],
        hidden=hidden, out_features=1, target_node=3,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    A = torch.from_numpy(A_norm.astype(np.float32)).to(device)
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y.reshape(-1, 1)).to(device)

    history = {"train": [], "val": []}
    n_tr = tr.stop - tr.start
    rng = np.random.default_rng(seed)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        idx = rng.permutation(n_tr) + tr.start
        running = 0.0
        for s in range(0, n_tr, batch_size):
            b = idx[s:s + batch_size]
            yh = model(Xt[b], A)
            loss = loss_fn(yh, yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss.item()) * len(b)
        model.eval()
        with torch.no_grad():
            yh_v = model(Xt[va], A)
            val_loss = float(loss_fn(yh_v, yt[va]).item())
        history["train"].append(running / n_tr)
        history["val"].append(val_loss)

    model.eval()
    with torch.no_grad():
        all_pred = model(Xt, A).cpu().numpy().ravel()
    return model, history, all_pred
