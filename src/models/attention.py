"""Cross-modal attention fusion module, per thesis Table 5.

PM2.5 branch output = source of Query.
Meteorological branch output = source of Key and Value.

NOTE: `d_model` is NOT assigned a numeric value anywhere in the thesis
(Table 5 uses the symbol only). The default of 128 here is a documented
ASSUMPTION -- see implementation guide Section 16/19 -- not a thesis fact.
Confirm with your adviser before treating this as final.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAttentionFusion(nn.Module):
    def __init__(
        self,
        branch_dim: int = 128,
        d_model: int = 128,
        num_heads: int = 4,
        ffn_dim: int = 256,
        fused_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.q_proj = nn.Linear(branch_dim, d_model)
        self.k_proj = nn.Linear(branch_dim, d_model)
        self.v_proj = nn.Linear(branch_dim, d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.residual_proj = (
            nn.Identity() if branch_dim == d_model else nn.Linear(branch_dim, d_model)
        )
        self.output_proj = nn.Linear(d_model, fused_dim)

    def forward(self, h_pm25: torch.Tensor, h_met: torch.Tensor):
        """h_pm25, h_met: (B, L, branch_dim) -> fused (B, L, fused_dim), attn (B, h, L, L)."""
        q = self.q_proj(h_pm25)
        k = self.k_proj(h_met)
        v = self.v_proj(h_met)
        attn_out, attn_weights = self.mha(
            q, k, v, need_weights=True, average_attn_weights=False
        )
        z = self.norm1(attn_out + self.residual_proj(h_pm25))
        z = self.norm2(self.ffn(z) + z)
        fused = self.output_proj(z)
        return fused, attn_weights


if __name__ == "__main__":
    fusion = CrossModalAttentionFusion()
    h_pm25 = torch.randn(4, 48, 128)
    h_met = torch.randn(4, 48, 128)
    fused, w = fusion(h_pm25, h_met)
    assert fused.shape == (4, 48, 256)
    print("Cross-modal attention shape check passed:", fused.shape, w.shape)
