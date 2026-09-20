"""Window-based scale-aware prior attention module."""

import torch
from torch import nn
from torch.nn import functional as F


def window_partition(x, window_size):
    (B, C, H, W) = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = (
        x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    C = windows.shape[1]
    x = windows.view(B, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
    return x


class SA_PAM(nn.Module):
    def __init__(
        self,
        rgb_channels=3,
        prior_channels=1,
        embed_dim=64,
        num_scales=4,
        num_heads=8,
        mlp_ratio=2,
        window_size=8,
    ):
        super(SA_PAM, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        self.scale_factor = self.head_dim ** (-0.5)
        self.window_size = window_size
        self.rgb_projector = nn.Sequential(
            nn.Conv2d(rgb_channels, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.scale_embedding_dict = nn.Embedding(num_scales, embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.see_mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, 2 * embed_dim),
        )
        self.to_q = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(prior_channels, embed_dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(prior_channels, embed_dim, kernel_size=1, bias=False)
        self.proj_out = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x_rgb, x_prior, scale_idx):
        (B, C_rgb, H, W) = x_rgb.shape
        assert H % self.window_size == 0 and W % self.window_size == 0, (
            f"H({H}) and W({W}) must be divisible by window_size({self.window_size})"
        )
        f_rgb = self.rgb_projector(x_rgb)
        if isinstance(scale_idx, int):
            scale_idx = torch.full(
                (B,), scale_idx, device=x_rgb.device, dtype=torch.long
            )
        e_i = self.scale_embedding_dict(scale_idx)
        gamma_beta = self.see_mlp(e_i)
        (gamma, beta) = gamma_beta.chunk(2, dim=1)
        gamma = gamma.view(B, self.embed_dim, 1, 1)
        beta = beta.view(B, self.embed_dim, 1, 1)
        f_hat_rgb = gamma * f_rgb + beta
        q_full = self.to_q(f_hat_rgb)
        k_full = self.to_k(x_prior)
        v_full = self.to_v(x_prior)
        ws = self.window_size
        q_windows = window_partition(q_full, ws)
        k_windows = window_partition(k_full, ws)
        v_windows = window_partition(v_full, ws)
        BW = q_windows.shape[0]
        N = ws * ws
        q = q_windows.view(BW, self.num_heads, self.head_dim, N)
        k = k_windows.view(BW, self.num_heads, self.head_dim, N)
        v = v_windows.view(BW, self.num_heads, self.head_dim, N)
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor
        attn_map = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_map, v)
        attn_out = (
            attn_out.transpose(-2, -1).contiguous().view(BW, self.embed_dim, ws, ws)
        )
        attn_out = window_reverse(attn_out, ws, H, W)
        f_attn_projected = self.proj_out(attn_out)
        f_out = f_attn_projected + f_hat_rgb
        return f_out
