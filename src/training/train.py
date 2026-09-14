"""Shared training loop for all four variants (thesis Data Generation Step 10: identical hyperparameters across all variants to ensure fair comparison)."""
from __future__ import annotations

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    """Makes weight init, dropout, and data shuffling reproducible across runs.
    Call once per variant, right before building the model, so A/B/C/D all
    start from the same reproducible random state rather than an arbitrary one."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op safely if no GPU present
    # torch.manual_seed alone does NOT make CUDA convolutions deterministic --
    # cuDNN can pick non-deterministic algorithms (e.g. atomic-add-based
    # backward passes) whose tiny per-step floating-point differences compound
    # over thousands of optimizer steps into a visibly different result by
    # epoch 2+, even with the "same" seed. These two flags fix that.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(X_pm25, X_met, Y, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.as_tensor(X_pm25), torch.as_tensor(X_met), torch.as_tensor(Y)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_variant(model, train_data, val_data, cfg: dict, ckpt_path: str) -> float:
    """train_data / val_data: (X_pm25, X_met, Y) numpy tuples.
    NOTE: call set_seed() in the CALLING code, before constructing `model` --
    seeding here is too late, since weight initialization already happened
    by the time this function receives the model object."""
    device = cfg["training"]["device"]
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    loss_fn = torch.nn.MSELoss()

    train_loader = make_loader(*train_data, cfg["training"]["batch_size"], shuffle=True)
    val_loader = make_loader(*val_data, cfg["training"]["batch_size"], shuffle=False)

    best_val_rmse = float("inf")
    patience_left = cfg["training"]["early_stopping_patience"]

    for epoch in range(cfg["training"]["max_epochs"]):
        model.train()
        train_sq_err, train_n = 0.0, 0
        for x_pm25, x_met, y in train_loader:
            x_pm25, x_met, y = x_pm25.to(device), x_met.to(device), y.to(device)
            opt.zero_grad()
            y_hat = model(x_pm25, x_met)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()

            # Accumulated from each batch's forward pass BEFORE that batch's
            # weight update -- the standard "running train loss" approximation.
            # Cheap: reuses y_hat already computed for the backward pass, no
            # second pass over the training set. Good enough to distinguish
            # "train loss keeps falling, val rises" (overfitting) from
            # "train loss itself is unstable" (LR/architecture issue) --
            # not an exact end-of-epoch train RMSE at fixed weights.
            train_sq_err += ((y_hat.detach() - y) ** 2).sum().item()
            train_n += y.numel()
        train_rmse = (train_sq_err / train_n) ** 0.5

        model.eval()
        sq_err, n = 0.0, 0
        with torch.no_grad():
            for x_pm25, x_met, y in val_loader:
                x_pm25, x_met, y = x_pm25.to(device), x_met.to(device), y.to(device)
                y_hat = model(x_pm25, x_met)
                sq_err += ((y_hat - y) ** 2).sum().item()
                n += y.numel()
        val_rmse = (sq_err / n) ** 0.5
        print(
            f"epoch {epoch:3d}  train_rmse={train_rmse:.4f}  "
            f"val_rmse={val_rmse:.4f}  patience_left={patience_left}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_left = cfg["training"]["early_stopping_patience"]
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}, best val RMSE={best_val_rmse:.4f}")
                break

    return best_val_rmse