"""Four ablation variants, per thesis Table 1 / Data Generation Step 9.

Variant A: single-branch TCN, unified encoding (PM2.5 + met concatenated
           BEFORE encoding), concatenation fusion (trivial, one stream).
Variant B: dual-branch TCN, modality-specific encoding, concatenation fusion.
Variant C: dual-branch TCN, modality-specific encoding, cross-modal
           attention fusion. This is AIRPRED.
Variant D: single-branch TCN, PM2.5-only input, no meteorology at all.

Everything not listed above (TCN block design, forecast head design,
training hyperparameters) is IDENTICAL across all four, by design --
see implementation guide Section 24 (fair-comparison checklist).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.tcn import TCNEncoder, MultiScaleTCNEncoder
from src.models.attention import CrossModalAttentionFusion, CrossModalAttentionFusionPlus


class ForecastHead(nn.Module):
    """Shared by all four variants (thesis Table 6)."""

    def __init__(self, in_dim: int, hidden: int = 128, horizon: int = 24, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, in_dim)
        x = x.mean(dim=1)             # Global Average Pooling over L -> (B, in_dim)
        x = self.act(self.fc1(x))     # (B, hidden)
        x = self.drop(x)
        return self.fc2(x)            # (B, horizon)


class VariantA_SingleBranchUnified(nn.Module):
    """RQ1 baseline."""

    def __init__(self, n_met_features: int = 7, horizon: int = 24):
        super().__init__()
        self.encoder = TCNEncoder(in_channels=1 + n_met_features)
        self.head = ForecastHead(in_dim=128, horizon=horizon)

    def forward(self, x_pm25: torch.Tensor, x_met: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_pm25, x_met], dim=-1)
        h = self.encoder(x)
        return self.head(h)


class VariantB_DualBranchConcat(nn.Module):
    """RQ1 proposed model / RQ2 baseline."""

    def __init__(self, n_met_features: int = 7, horizon: int = 24):
        super().__init__()
        self.pm25_encoder = TCNEncoder(in_channels=1)
        self.met_encoder = TCNEncoder(in_channels=n_met_features)
        self.head = ForecastHead(in_dim=256, horizon=horizon)

    def forward(self, x_pm25: torch.Tensor, x_met: torch.Tensor) -> torch.Tensor:
        h_pm25 = self.pm25_encoder(x_pm25)
        h_met = self.met_encoder(x_met)
        fused = torch.cat([h_pm25, h_met], dim=-1)
        return self.head(fused)


class VariantC_AIRPRED(nn.Module):
    """RQ2 & RQ3 proposed model. This is AIRPRED."""

    def __init__(self, n_met_features: int = 7, horizon: int = 24,
                 d_model: int = 128, num_heads: int = 4):
        super().__init__()
        self.pm25_encoder = TCNEncoder(in_channels=1)
        self.met_encoder = TCNEncoder(in_channels=n_met_features)
        self.fusion = CrossModalAttentionFusion(
            branch_dim=128, d_model=d_model, num_heads=num_heads, fused_dim=256
        )
        self.head = ForecastHead(in_dim=256, horizon=horizon)

    def forward(self, x_pm25: torch.Tensor, x_met: torch.Tensor, return_attention: bool = False):
        h_pm25 = self.pm25_encoder(x_pm25)
        h_met = self.met_encoder(x_met)
        fused, attn_weights = self.fusion(h_pm25, h_met)
        out = self.head(fused)
        return (out, attn_weights) if return_attention else out


class VariantD_PM25Only(nn.Module):
    """RQ3 baseline. x_met accepted-and-ignored for a uniform call signature
    across all four variants (simplifies the shared training loop)."""

    def __init__(self, horizon: int = 24):
        super().__init__()
        self.encoder = TCNEncoder(in_channels=1)
        self.head = ForecastHead(in_dim=128, horizon=horizon)

    def forward(self, x_pm25: torch.Tensor, x_met: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder(x_pm25)
        return self.head(h)


# ---------------------------------------------------------------------------
# C+ (improved AIRPRED). VariantC_AIRPRED above is NOT modified, so the original
# model and its checkpoints stay reproducible. See docs/variant_c_plus.md.
# ---------------------------------------------------------------------------

C_PLUS_DEFAULTS = dict(
    tcn_channels=[64, 64, 128, 128], dilations=[1, 2, 4, 8], kernel_size=3, tcn_block_dropout=0.0,
    d_model=128, num_attention_heads=4, ffn_dim=256, fused_dim=256,
    multi_scale=True, ms_taps=None,
    input_norm=True, attention_dropout=0.1, ffn_dropout=0.1, bidirectional_attention=False,
)

# Ablation presets. A preset pins ONLY the keys it lists; every other value comes from
# config.yaml (model.c_plus) -- so CPLUS is "whatever the config says".
C_PLUS_PRESETS = {
    "CMS":     dict(multi_scale=True, input_norm=False, attention_dropout=0.0, ffn_dropout=0.0,
                    bidirectional_attention=False),                      # C + multi-scale TCN only
    "CATT":    dict(multi_scale=False, bidirectional_attention=False),   # C + improved attention only
    "CPLUS":   dict(),                                                   # multi-scale + improved attention
    "CPLUSBI": dict(bidirectional_attention=True),                       # CPLUS + bidirectional attention
}

VARIANT_KEYS = ["A", "B", "C", "D"] + list(C_PLUS_PRESETS)


class VariantC_Plus(nn.Module):
    """Improved AIRPRED (CMA-TCN+). Same interface as VariantC_AIRPRED:
    forward(x_pm25, x_met, return_attention=False) -> (B, horizon).

    Two separate modality branches (PM2.5 / meteorology) -> optional multi-scale TCN
    readout -> cross-modal attention (PM2.5 = Query, meteorology = Key/Value; optional
    reverse direction) -> same ForecastHead as every other variant."""

    def __init__(self, n_met_features: int = 7, horizon: int = 24, **kw):
        super().__init__()
        unknown = set(kw) - set(C_PLUS_DEFAULTS)
        assert not unknown, f"unknown C+ options: {sorted(unknown)}"
        c = {**C_PLUS_DEFAULTS, **kw}
        self.hparams = c
        d = c["d_model"]

        def enc(in_ch):
            return MultiScaleTCNEncoder(
                in_ch, channels=c["tcn_channels"], dilations=c["dilations"], kernel_size=c["kernel_size"],
                block_dropout=c["tcn_block_dropout"], multi_scale=c["multi_scale"], taps=c["ms_taps"], out_dim=d)

        self.pm25_encoder = enc(1)
        self.met_encoder = enc(n_met_features)
        assert self.pm25_encoder.out_channels == d, (
            f"branch output ({self.pm25_encoder.out_channels}) must equal d_model ({d}): "
            f"use multi_scale=true, or set tcn_channels[-1] = d_model")
        self.fusion = CrossModalAttentionFusionPlus(
            d_model=d, num_heads=c["num_attention_heads"], ffn_dim=c["ffn_dim"], fused_dim=c["fused_dim"],
            attention_dropout=c["attention_dropout"], ffn_dropout=c["ffn_dropout"],
            input_norm=c["input_norm"], bidirectional=c["bidirectional_attention"])
        self.head = ForecastHead(in_dim=c["fused_dim"], horizon=horizon)

    def forward(self, x_pm25: torch.Tensor, x_met: torch.Tensor, return_attention: bool = False):
        h_pm25 = self.pm25_encoder(x_pm25)
        h_met = self.met_encoder(x_met)
        fused, attn = self.fusion(h_pm25, h_met, need_weights=return_attention)
        out = self.head(fused)
        return (out, attn) if return_attention else out


class ResidualForecast(nn.Module):
    """Optional residual forecasting: out = last observed (scaled) PM2.5 + base(x).
    The head's output layer is zero-initialised so training starts at persistence.
    state_dict()/load_state_dict() delegate to the base model, so checkpoints keep the
    ordinary format (residual-ness is code, not weights -- record it in the file name)."""

    def __init__(self, base: nn.Module, zero_init: bool = True):
        super().__init__()
        self.base = base
        if zero_init:
            nn.init.zeros_(base.head.fc2.weight)
            nn.init.zeros_(base.head.fc2.bias)

    def forward(self, x_pm25, x_met=None, **kw):
        out = self.base(x_pm25, x_met, **kw)
        if isinstance(out, tuple):
            return (out[0] + x_pm25[:, -1, :],) + tuple(out[1:])
        return out + x_pm25[:, -1, :]

    def state_dict(self, *a, **k):
        return self.base.state_dict(*a, **k)

    def load_state_dict(self, sd, *a, **k):
        return self.base.load_state_dict(sd, *a, **k)


def build_variant(name: str, horizon: int, cfg: dict, residual: bool = False, overrides: dict | None = None):
    """Single place that turns a variant key (A, B, C, D, CMS, CATT, CPLUS, CPLUSBI) into a model.
    A, B, D and the original C are constructed exactly as before. `overrides` (e.g.
    {"attention_dropout": 0.2}) are applied to the config values BEFORE the preset's pins."""
    name = name.upper()
    mcfg = cfg["model"]
    if name == "A":
        model = VariantA_SingleBranchUnified(horizon=horizon)
    elif name == "B":
        model = VariantB_DualBranchConcat(horizon=horizon)
    elif name == "D":
        model = VariantD_PM25Only(horizon=horizon)
    elif name == "C":
        model = VariantC_AIRPRED(horizon=horizon, d_model=mcfg["d_model"], num_heads=mcfg["num_attention_heads"])
    elif name in C_PLUS_PRESETS:
        opts = {**C_PLUS_DEFAULTS, **(mcfg.get("c_plus") or {}), **(overrides or {}), **C_PLUS_PRESETS[name]}
        model = VariantC_Plus(horizon=horizon, **opts)
    else:
        raise ValueError(f"unknown variant {name!r}; choose from {VARIANT_KEYS}")
    return ResidualForecast(model) if residual else model


if __name__ == "__main__":
    B, L, H = 4, 48, 24
    x_pm25, x_met = torch.randn(B, L, 1), torch.randn(B, L, 7)
    for Model in (
        VariantA_SingleBranchUnified,
        VariantB_DualBranchConcat,
        VariantC_AIRPRED,
        VariantD_PM25Only,
    ):
        model = Model(horizon=H)
        out = model(x_pm25, x_met)
        assert out.shape == (B, H), f"{Model.__name__} shape mismatch: {out.shape}"
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{Model.__name__}: output {tuple(out.shape)}, {n_params:,} parameters")
    print("All four variants: forward pass + shape checks passed.")