import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


# ---------------------------
# Config
# ---------------------------
DATA_DIR = "dataset"
TRAIN_RATIO = 0.85
BATCH_SIZE = 8
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
SEED = 42
DEVICE = "cuda"
NUM_WORKERS = 0
PIN_MEMORY = True
MODEL_PATH = "FMR.pth"

GRID_DX = 5.6e-4
GRID_DY = 5.6e-4
DEPTH_SCALE = 1.0e-3
Y_POSITIVE_UP = True

TAU_CONTACT = 1e-3
W_MAP = 1.0
W_FORCE = 0.5
W_TORQUE = 0.2
W_SPARSE = 0.1


# ---------------------------
# Basic utils
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_xy_grid(h, w, dx, dy, y_positive_up=True):
    x = (np.arange(w, dtype=np.float32) - (w - 1) / 2.0) * dx
    y = (np.arange(h, dtype=np.float32) - (h - 1) / 2.0) * dy
    if y_positive_up:
        y = y[::-1].copy()
    xx, yy = np.meshgrid(x, y)
    return xx.astype(np.float32), yy.astype(np.float32)


def reshape_mlp_input(arr, h, w):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[0] == h * w:
            return arr.reshape(h, w, arr.shape[1]).transpose(2, 0, 1)
        return arr.reshape(arr.shape[0], h, w)
    if arr.ndim == 3:
        if arr.shape[0] == h and arr.shape[1] == w:
            return arr.transpose(2, 0, 1)
        return arr
    raise ValueError(f"Unsupported MLP input shape: {arr.shape}")


def numeric_key(path: Path):
    try:
        return (0, int(path.name))
    except ValueError:
        return (1, path.name)


# ---------------------------
# Dataset
# ---------------------------
class ForceFieldDataset(Dataset):
    def __init__(self, data_dir):
        self.sample_dirs = sorted([p for p in Path(data_dir).iterdir() if p.is_dir()], key=numeric_key)

        depth0 = np.load(self.sample_dirs[0] / "uz_simu.npy").astype(np.float32)
        self.h, self.w = depth0.shape
        self.xx, self.yy = make_xy_grid(self.h, self.w, GRID_DX, GRID_DY, Y_POSITIVE_UP)

        feat0 = np.load(self.sample_dirs[0] / "MLP_data.npy").astype(np.float32)
        feat0 = reshape_mlp_input(feat0, self.h, self.w)
        self.deform_channels = feat0.shape[0]

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        d = self.sample_dirs[idx]

        # initial force map estimated by the forward MLP
        fx = np.load(d / "fx0.npy").astype(np.float32)
        fy = np.load(d / "fy0.npy").astype(np.float32)
        fz = np.load(d / "fz0.npy").astype(np.float32)

        # force-map labels
        fx_re = np.load(d / "fx.npy").astype(np.float32)
        fy_re = np.load(d / "fy.npy").astype(np.float32)
        fz_re = np.load(d / "fz.npy").astype(np.float32)

        depth = np.load(d / "uz.npy").astype(np.float32)

        # initial input for the forward MLP
        deform_feat = np.load(d / "MLP_data.npy").astype(np.float32)
        deform_feat = reshape_mlp_input(deform_feat, self.h, self.w)

        # global reference
        g = np.load(d / "Global.npy").astype(np.float32)
        if g.shape == (2, 3):
            g = g.reshape(-1)

        f_init = np.stack([fx, fy, fz], axis=0)
        f_gt = np.stack([fx_re, fy_re, fz_re], axis=0)
        coords = np.stack([self.xx, self.yy, depth * DEPTH_SCALE], axis=0)

        return (
            torch.from_numpy(f_init),
            torch.from_numpy(deform_feat),
            torch.from_numpy(coords),
            torch.from_numpy(f_gt),
            torch.from_numpy(g.astype(np.float32)),
        )


# ---------------------------
# Model blocks
# ---------------------------
def compute_force_and_torque(coords, forces):
    total_force = forces.sum(dim=2)
    total_torque = torch.cross(coords, forces, dim=1).sum(dim=2)
    return torch.cat([total_force, total_torque], dim=1)


class DWConv(nn.Module):
    def __init__(self, c, k=7):
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size=k, padding=k // 2, groups=c)

    def forward(self, x):
        return self.dw(x)


class ConvNeXtBlockLite(nn.Module):
    def __init__(self, c, expansion=4):
        super().__init__()
        self.dw = DWConv(c, k=7)
        self.pw1 = nn.Conv2d(c, c * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(c * expansion, c, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 1e-6)

    def forward(self, x):
        y = self.dw(x)
        y = self.pw1(y)
        y = self.act(y)
        y = self.pw2(y)
        return x + self.gamma * y


class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=2, padding=1),
            nn.GroupNorm(8, c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.up(x))


class FiLM(nn.Module):
    def __init__(self, c, cond_dim):
        super().__init__()
        self.c = c
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, c * 2),
            nn.SiLU(inplace=True),
            nn.Linear(c * 2, c * 2),
        )

    def forward(self, x, cond):
        h = self.mlp(cond)
        gamma, beta = h[:, :self.c], h[:, self.c:]
        gamma = gamma.view(-1, self.c, 1, 1)
        beta = beta.view(-1, self.c, 1, 1)
        return x * (1.0 + gamma) + beta


class LiteCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, kv_stride=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.kv_stride = kv_stride
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, q_feat, kv_feat, extra_kv_tokens=None):
        b, c, h, w = q_feat.shape
        q = q_feat.permute(0, 2, 3, 1).reshape(b, h * w, c)
        kv = kv_feat
        if self.kv_stride > 1:
            kv = F.avg_pool2d(kv, kernel_size=self.kv_stride, stride=self.kv_stride)
        kv = kv.permute(0, 2, 3, 1).reshape(b, -1, c)
        if extra_kv_tokens is not None:
            kv = torch.cat([kv, extra_kv_tokens], dim=1)
        out, _ = self.attn(self.q_norm(q), self.kv_norm(kv), self.kv_norm(kv), need_weights=False)
        out = self.proj(out)
        y = q + self.alpha * out
        return y.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class DualBranchBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(ConvNeXtBlockLite(c), ConvNeXtBlockLite(c))

    def forward(self, x):
        return self.block(x)


class ForceRefineNet(nn.Module):
    def __init__(self, deform_in_ch, global_dim=6, base=48, num_global_tokens=4):
        super().__init__()
        g_dim = base * 4

        self.force_stem = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(inplace=True),
        )
        self.force_enc1 = DualBranchBlock(base)
        self.force_down1 = Down(base, base * 2)
        self.force_enc2 = DualBranchBlock(base * 2)
        self.force_down2 = Down(base * 2, base * 4)
        self.force_enc3 = DualBranchBlock(base * 4)
        self.force_film = FiLM(base * 4, g_dim)
        self.force_bot = DualBranchBlock(base * 4)
        self.force_up2 = Up(base * 4, base * 2)
        self.force_dec2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(inplace=True),
            ConvNeXtBlockLite(base * 2),
        )
        self.force_up1 = Up(base * 2, base)
        self.force_dec1 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(inplace=True),
            ConvNeXtBlockLite(base),
        )

        self.deform_stem = nn.Sequential(
            nn.Conv2d(deform_in_ch, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(inplace=True),
        )
        self.deform_enc1 = DualBranchBlock(base)
        self.deform_down1 = Down(base, base * 2)
        self.deform_enc2 = DualBranchBlock(base * 2)
        self.deform_down2 = Down(base * 2, base * 4)
        self.deform_enc3 = DualBranchBlock(base * 4)
        self.deform_bot = DualBranchBlock(base * 4)
        self.deform_up2 = Up(base * 4, base * 2)
        self.deform_dec2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(inplace=True),
            ConvNeXtBlockLite(base * 2),
        )
        self.deform_up1 = Up(base * 2, base)
        self.deform_dec1 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(inplace=True),
            ConvNeXtBlockLite(base),
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, g_dim),
            nn.SiLU(inplace=True),
            nn.Linear(g_dim, g_dim),
        )
        self.global_token_head = nn.Sequential(
            nn.Linear(g_dim, g_dim),
            nn.SiLU(inplace=True),
            nn.Linear(g_dim, num_global_tokens * base * 4),
        )
        self.bottleneck_attn = LiteCrossAttention(dim=base * 4)

        self.fuse_proj = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(inplace=True),
            ConvNeXtBlockLite(base),
        )

        self.head_pool = nn.AdaptiveAvgPool2d(1)
        self.head_local = nn.Conv2d(base, 3, 1)
        self.head_g_proj = nn.Sequential(
            nn.Linear(g_dim, base),
            nn.SiLU(inplace=True),
            nn.Linear(base, base),
        )
        self.head_global_out = nn.Linear(base, 3)
        self.num_global_tokens = num_global_tokens
        self.base = base

    def resize_like(self, x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, f_init, deform_feat, coords, g_in):
        b, _, h, w = f_init.shape
        g = self.global_encoder(g_in)

        f1 = self.force_enc1(self.force_stem(f_init))
        f2 = self.force_enc2(self.force_down1(f1))
        f3 = self.force_enc3(self.force_down2(f2))
        f3 = self.force_film(f3, g)

        d1 = self.deform_enc1(self.deform_stem(deform_feat))
        d2 = self.deform_enc2(self.deform_down1(d1))
        d3 = self.deform_enc3(self.deform_down2(d2))

        g_tokens = self.global_token_head(g).view(b, self.num_global_tokens, self.base * 4)
        f3 = self.bottleneck_attn(f3, d3, extra_kv_tokens=g_tokens)

        fu2 = self.force_dec2(torch.cat([self.resize_like(self.force_up2(self.force_bot(f3)), f2), f2], dim=1))
        fu1 = self.force_dec1(torch.cat([self.resize_like(self.force_up1(fu2), f1), f1], dim=1))

        du2 = self.deform_dec2(torch.cat([self.resize_like(self.deform_up2(self.deform_bot(d3)), d2), d2], dim=1))
        du1 = self.deform_dec1(torch.cat([self.resize_like(self.deform_up1(du2), d1), d1], dim=1))

        fused = self.fuse_proj(torch.cat([fu1, du1], dim=1))
        delta_local = self.head_local(fused)
        pooled = self.head_pool(fused).flatten(1)
        delta_global = self.head_global_out(pooled * self.head_g_proj(g)).view(b, 3, 1, 1)

        f_out = f_init + delta_local + delta_global
        t_pred = compute_force_and_torque(coords.reshape(b, 3, h * w), f_out.reshape(b, 3, h * w))
        return f_out, t_pred


# ---------------------------
# Loss and train
# ---------------------------
def compute_loss(f_out, f_gt, t_pred, t_gt):
    contact = (f_gt.abs().sum(dim=1, keepdim=True) > TAU_CONTACT).float()
    noncontact = (1.0 - contact).expand(-1, 3, -1, -1)

    map_loss = F.mse_loss(f_out, f_gt)
    force_loss = F.mse_loss(t_pred[:, :3], t_gt[:, :3])
    torque_loss = F.mse_loss(t_pred[:, 3:6], t_gt[:, 3:6])
    sparse_loss = (f_out.abs() * noncontact).sum() / noncontact.sum().clamp_min(1.0)

    return W_MAP * map_loss + W_FORCE * force_loss + W_TORQUE * torque_loss + W_SPARSE * sparse_loss


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_num = 0

    with torch.no_grad():
        for f_init, deform_feat, coords, f_gt, g_tgt in loader:
            f_init = f_init.to(device)
            deform_feat = deform_feat.to(device)
            coords = coords.to(device)
            f_gt = f_gt.to(device)
            g_tgt = g_tgt.to(device)

            f_out, t_pred = model(f_init, deform_feat, coords, g_tgt)
            loss = compute_loss(f_out, f_gt, t_pred, g_tgt)

            bs = f_init.shape[0]
            total_loss += loss.item() * bs
            total_num += bs

    return total_loss / total_num


def train():
    set_seed(SEED)

    device = torch.device("cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
    dataset = ForceFieldDataset(DATA_DIR)

    n_total = len(dataset)
    n_train = int(round(n_total * TRAIN_RATIO))
    n_val = n_total - n_train
    split_gen = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=split_gen)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = ForceRefineNet(deform_in_ch=dataset.deform_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_num = 0

        for f_init, deform_feat, coords, f_gt, g_tgt in train_loader:
            f_init = f_init.to(device)
            deform_feat = deform_feat.to(device)
            coords = coords.to(device)
            f_gt = f_gt.to(device)
            g_tgt = g_tgt.to(device)

            optimizer.zero_grad(set_to_none=True)
            f_out, t_pred = model(f_init, deform_feat, coords, g_tgt)
            loss = compute_loss(f_out, f_gt, t_pred, g_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            bs = f_init.shape[0]
            total_loss += loss.item() * bs
            total_num += bs

        train_loss = total_loss / total_num
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()
