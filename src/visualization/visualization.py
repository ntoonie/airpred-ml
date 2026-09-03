"""Plotting helpers for Chapter 4 figures (implementation guide Section 28)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_loss_curves(train_losses, val_losses, title, out_path):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE loss (scaled units)")
    plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_actual_vs_predicted(timestamps, y_true, y_pred, city, out_path):
    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, y_true, label="actual")
    plt.plot(timestamps, y_pred, label="predicted")
    plt.xlabel("time"); plt.ylabel("PM2.5 (µg/m³)")
    plt.title(f"Actual vs. predicted PM2.5 — {city}")
    plt.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def plot_horizon_degradation(rmse_per_step, out_path):
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(rmse_per_step) + 1), rmse_per_step, marker="o")
    plt.xlabel("forecast horizon step"); plt.ylabel("RMSE (µg/m³)")
    plt.title("Forecast error vs. horizon"); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()


def plot_attention_heatmap(attn_weights_single_sample, out_path):
    """attn_weights_single_sample: (L, L) averaged over heads for ONE test
    sample. Rows = PM2.5 query timestep, cols = meteorological key timestep."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(attn_weights_single_sample, cmap="viridis")
    plt.xlabel("meteorological input timestep (t-47…t)")
    plt.ylabel("PM2.5 query timestep (t-47…t)")
    plt.title("Cross-modal attention weights")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
