import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

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

