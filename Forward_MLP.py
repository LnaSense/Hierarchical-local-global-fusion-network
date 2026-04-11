"""
================================
Training of the forward MLP model
================================
"""

import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# ======================
# CONFIG (edit here)
# ======================
INPUT_CSV = "data_in.csv"          # Input feature CSV (columns: div_x,	div_y, rot_x, rot_y, harm_x, harm_y, z)
TARGET_CSV = "data_out.csv"       # Target force CSV (columns: fx, fy, fz)
TORCHSCRIPT_PATH = "forwardMLP.ts"  # Exported TorchScript file

BATCH_SIZE = 1024
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN = [128, 128, 64]
DROPOUT = 0.1
USE_CUDA = True
MAX_SAMPLES = None   # Set an integer for debugging; None uses all samples
SEED = 42
# ======================


def set_seed(seed: int = 42):
    # Set random seeds for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256, 128), dropout=0.0,
                 dom_idx: int = None, alpha: float = 1.0):
        super().__init__()
        # Optionally amplify one selected input dimension.
        self.dom_idx = dom_idx
        self.alpha = alpha

        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.dom_idx is not None and self.alpha != 1.0:
            # Clone the tensor to avoid in-place modification of the input.
            x = x.clone()
            x[..., self.dom_idx] *= self.alpha

        return self.net(x)


class InferenceWrapper(nn.Module):
    """
    Self-contained inference module:
    - Input: raw features x_raw [N, in_dim]
    - Output: de-normalized force y [N, out_dim] in N
    """
    def __init__(self, core_mlp: nn.Module,
                 x_mean: torch.Tensor, x_std: torch.Tensor,
                 y_mean: torch.Tensor, y_std: torch.Tensor):
        super().__init__()
        self.core = core_mlp
        self.register_buffer("x_mean", x_mean.clone().detach())
        self.register_buffer("x_std",  x_std.clone().detach())
        self.register_buffer("y_mean", y_mean.clone().detach())
        self.register_buffer("y_std",  y_std.clone().detach())

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        # Normalize inputs using training statistics.
        x_n = (x_raw - self.x_mean) / (self.x_std + 1e-8)
        y_n = self.core(x_n)
        y = y_n * (self.y_std + 1e-8) + self.y_mean
        return y


def main():
    set_seed(SEED)

    # Load input and target data.
    X = pd.read_csv(INPUT_CSV, header=0).values.astype(np.float32)
    y = pd.read_csv(TARGET_CSV, header=0).values.astype(np.float32)
    n = min(len(X), len(y))
    X, y = X[:n], y[:n]

    if MAX_SAMPLES is not None and n > MAX_SAMPLES:
        # Randomly subsample the dataset when debugging.
        rng = np.random.default_rng(SEED)
        idx = rng.choice(n, size=MAX_SAMPLES, replace=False)
        X, y = X[idx], y[idx]
        n = len(X)

    assert y.shape[1] >= 3, "Target CSV must contain at least three columns: [fx, fy, fz]"

    print(X.shape)

    # Split the dataset into 85% training and 15% validation.
    ds_raw = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.float32))
    n_train = int(0.85 * n)
    n_val = n - n_train
    g = torch.Generator().manual_seed(SEED)
    train_raw, val_raw = random_split(ds_raw, [n_train, n_val], generator=g)

    # Compute normalization statistics from the training subset only.
    X_train_raw = train_raw.dataset.tensors[0][train_raw.indices].numpy()
    y_train_raw = train_raw.dataset.tensors[1][train_raw.indices].numpy()
    x_mean = X_train_raw.mean(0)
    x_std = X_train_raw.std(0) + 1e-8
    y_mean = y_train_raw.mean(0)
    y_std = y_train_raw.std(0) + 1e-8

    def norm_xy(subset):
        # Normalize one dataset subset using training statistics.
        X_sub = subset.dataset.tensors[0][subset.indices].numpy()
        y_sub = subset.dataset.tensors[1][subset.indices].numpy()
        Xn = (X_sub - x_mean) / x_std
        yn = (y_sub - y_mean) / y_std
        return torch.tensor(Xn, dtype=torch.float32), torch.tensor(yn, dtype=torch.float32)

    X_train_n, y_train_n = norm_xy(train_raw)
    X_val_n, y_val_n = norm_xy(val_raw)

    train_loader = DataLoader(TensorDataset(X_train_n, y_train_n), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_n, y_val_n), batch_size=BATCH_SIZE, shuffle=False)

    # Build the model and optimizer.
    in_dim, out_dim = X.shape[1], y.shape[1]
    model = MLP(in_dim, out_dim, hidden=tuple(HIDDEN), dropout=DROPOUT, dom_idx=6, alpha=50)
    device = torch.device("cuda" if (torch.cuda.is_available() and USE_CUDA) else "cpu")
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    # Track the best checkpoint using validation MSE.
    best_val = float("inf")
    best_state = None

    x_mean_t = torch.tensor(x_mean, dtype=torch.float32)
    x_std_t = torch.tensor(x_std, dtype=torch.float32)
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32)
    y_std_t = torch.tensor(y_std, dtype=torch.float32)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tr_loss += loss.detach().cpu().item() * len(xb)
        tr_loss /= max(1, len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(model(xb), yb).detach().cpu().item() * len(xb)
        val_loss /= max(1, len(val_loader.dataset))

        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch {epoch:03d}/{EPOCHS} | train MSE: {tr_loss:.6f} | val MSE: {val_loss:.6f}")

        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_state = {"model": copy.deepcopy(model.state_dict())}

    # Load the best validation checkpoint and report the final validation MSE.
    model.load_state_dict(best_state["model"])
    model.eval()

    final_val_loss = 0.0
    with torch.inference_mode():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            final_val_loss += loss_fn(model(xb), yb).detach().cpu().item() * len(xb)
    final_val_loss /= max(1, len(val_loader.dataset))
    print(f"Best validation MSE: {final_val_loss:.6f}")

    # Export a self-contained TorchScript inference module.
    core = MLP(in_dim, out_dim, hidden=tuple(HIDDEN), dropout=DROPOUT, dom_idx=6, alpha=50)
    core.load_state_dict(best_state["model"])
    core.eval()

    wrapper = InferenceWrapper(core, x_mean_t, x_std_t, y_mean_t, y_std_t).eval()
    scripted = torch.jit.script(wrapper)
    torch.jit.save(scripted, TORCHSCRIPT_PATH)
    print(f"TorchScript export completed: {TORCHSCRIPT_PATH}")


if __name__ == "__main__":
    main()
