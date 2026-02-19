
import torch
import torch.nn as nn
import torch.nn.functional as F
from prithvi_teacher.prithvi_mae import PrithviViT

class RegimeAggregator(nn.Module):
    def __init__(self, embed_dim=256, K=4, num_heads=4, dropout=0.0):
        super().__init__()
        self.K = K
        self.tokens = nn.Parameter(torch.randn(1, K, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, patch_tokens):
        B, N, D = patch_tokens.shape
        q = self.tokens.expand(B, -1, -1)
        q = self.ln_q(q)
        kv = self.ln_kv(patch_tokens)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = out + self.ff(out)
        pooled = out.mean(dim=1)
        reg_scores = torch.norm(out, dim=-1)
        reg_prob = torch.softmax(reg_scores, dim=-1)
        return pooled, reg_prob

class StudentRegime(nn.Module):
    def __init__(self, img_size=256, num_frames=3, embed_dim=256, depth=8, heads=4, K_reg=4):
        super().__init__()
        self.enc = PrithviViT(
            img_size=img_size,
            num_frames=num_frames,
            patch_size=(1,16,16),
            in_chans=6,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=heads,
            mlp_ratio=4.0,
            coords_encoding=["time","location"],
            coords_scale_learn=True,
        )
        self.reg = RegimeAggregator(embed_dim=embed_dim, K=K_reg, num_heads=heads)
        self.proj_cls = nn.Linear(2 * embed_dim, 1024)
        self.proj_pm  = nn.Linear(2 * embed_dim, 1024)

    def forward_once(self, x, tc, lc):
        feats = self.enc.forward_features(x, tc, lc)
        last = feats[-1]
        cls = last[:, 0, :]
        patches = last[:, 1:, :]
        pm = patches.mean(dim=1)
        reg_pool, reg_prob = self.reg(patches)
        cls_fused = torch.cat([cls, reg_pool], dim=-1)
        pm_fused  = torch.cat([pm,  reg_pool], dim=-1)
        cls1024 = self.proj_cls(cls_fused)
        pm1024  = self.proj_pm(pm_fused)
        return cls1024, pm1024, reg_prob

