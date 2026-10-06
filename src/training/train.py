"""TCN residual block + dual-branch encoder, per thesis Table 4.

NOTE (see implementation guide Section 17):
  - The thesis's stated cumulative receptive fields (3, 7, 15, 31) arithmetically
    match ONE dilated conv per block. The block's own textual description
    ("two dilated causal convolution sub-layers") matches the standard
    Bai et al. (2018) design, which would give cumulative RF 5/13/29/61.
    This file implements the two-conv-per-block (Bai et al.) version, since
    that is what Figure 8's block diagram shows. Flag this discrepancy to
    your adviser -- do not treat this file as having silently resolved it.
  - TCN block dropout is NOT specified in the thesis text (only the
    forecast head has dropout, p=0.1). Defaults to 0.0 here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Module):
    """Dilated causal 1D conv: left-pads so output[t] only sees input[<=t]."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.pad, dilation=dilation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, L)
        out = self.conv(x)
        return out[:, :, : -self.pad] if self.pad > 0 else out


class TCNResidualBlock(nn.Module):
    """Dilated Causal Conv -> WeightNorm -> ReLU -> Dilated Causal Conv ->
    WeightNorm -> Residual Add -> ReLU (thesis Table 4 Operations column).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int,
                 dropout: float = 0.0):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C_in, L)
        out = self.relu1(self.conv1(x))
        out = self.drop(self.conv2(out))
        res = x if self.downsample is None else self.downsample(x)
        assert out.shape == res.shape, (
            f"TCNResidualBlock shape mismatch: out={out.shape}, res={res.shape}"
        )
        return self.relu_out(out + res)


class TCNEncoder(nn.Module):
    """4-block encoder (thesis Table 4). Shared design for both branches;
    only `in_channels` differs (1 for PM2.5, 7 for meteorology)."""

    DILATIONS = [1, 2, 4, 8]
    CHANNELS = [64, 64, 128, 128]

    def __init__(self, in_channels: int, kernel_size: int = 3, block_dropout: float = 0.0):
        super().__init__()
        blocks = []
        prev_ch = in_channels
        for ch, d in zip(self.CHANNELS, self.DILATIONS):
            blocks.append(TCNResidualBlock(prev_ch, ch, kernel_size, d, dropout=block_dropout))
            prev_ch = ch
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = self.CHANNELS[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C_in) -> (B, L, 128)
        x = x.transpose(1, 2)  # (B, C_in, L) for Conv1d
        for block in self.blocks:
            x = block(x)
        return x.transpose(1, 2)  # (B, L, 128)


class MultiScaleTCNEncoder(nn.Module):
    """Configurable TCN encoder with optional multi-scale readout (used by C+).

    Same blocks as TCNEncoder (two dilated causal convs per block, thesis Table 4),
    but channels / dilations / kernel size come from config instead of class
    constants. With multi_scale=False it behaves like TCNEncoder (last block only).

    With multi_scale=True the output of every tapped block is read out, projected by a
    1x1 conv to out_dim // n_taps channels and concatenated. Block k has cumulative
    receptive field 1 + 2*(K-1)*sum(d_1..d_k) = 5, 13, 29, 61 steps for K=3 and
    dilations 1, 2, 4, 8, so the readout mixes short-, medium- and long-range features
    from ONE network instead of running several parallel TCNs. The 1x1 conv acts on the
    channel axis only, so causality is preserved.
    """

    def __init__(self, in_channels: int, channels=(64, 64, 128, 128), dilations=(1, 2, 4, 8),
                 kernel_size: int = 3, block_dropout: float = 0.0, multi_scale: bool = False,
                 taps=None, out_dim: int | None = None):
        super().__init__()
        assert len(channels) == len(dilations), "channels and dilations must have the same length"
        n_blocks = len(channels)
        self.taps = list(range(n_blocks)) if (multi_scale and taps is None) else (list(taps) if multi_scale else [n_blocks - 1])
        assert all(0 <= t < n_blocks for t in self.taps), f"taps {self.taps} out of range for {n_blocks} blocks"
        n_use = max(self.taps) + 1                       # blocks after the last tap would be unused
        blocks, prev = [], in_channels
        for ch, d in list(zip(channels, dilations))[:n_use]:
            blocks.append(TCNResidualBlock(prev, ch, kernel_size, d, dropout=block_dropout))
            prev = ch
        self.blocks = nn.ModuleList(blocks)
        self.multi_scale = multi_scale
        if multi_scale:
            out_dim = out_dim or channels[-1]
            assert out_dim % len(self.taps) == 0, f"out_dim {out_dim} must be divisible by the {len(self.taps)} taps"
            self.tap_proj = nn.ModuleList(
                [nn.Conv1d(channels[t], out_dim // len(self.taps), 1) for t in self.taps]
            )
            self.out_channels = out_dim
        else:
            self.tap_proj = None
            self.out_channels = channels[self.taps[0]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C_in) -> (B, L, out_channels)
        x = x.transpose(1, 2)
        feats = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.multi_scale and i in self.taps:
                feats.append(x)
        if self.multi_scale:
            x = torch.cat([proj(f) for proj, f in zip(self.tap_proj, feats)], dim=1)
        return x.transpose(1, 2)


if __name__ == "__main__":
    enc_pm25 = TCNEncoder(in_channels=1)
    enc_met = TCNEncoder(in_channels=7)
    x_pm25 = torch.randn(4, 48, 1)
    x_met = torch.randn(4, 48, 7)
    assert enc_pm25(x_pm25).shape == (4, 48, 128)
    assert enc_met(x_met).shape == (4, 48, 128)
    print("TCN encoder shape checks passed.")