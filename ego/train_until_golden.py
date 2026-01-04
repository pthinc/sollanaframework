# train_until_golden.py
"""Utility to train until validation loss reaches a golden threshold (torch backend)."""

from typing import Optional, Tuple

from backends import ensure_backend

GOLDEN_THRESHOLD = 0.1


def train_until_golden_ratio(model, train_loader, val_loader, loss_fn, max_epochs: int = 1000,
                             golden_threshold: float = GOLDEN_THRESHOLD, patience: int = 10,
                             device: Optional[str] = None, backend_name: Optional[str] = None) -> Tuple[object, float]:
    backend = ensure_backend(backend_name)
    if not backend.name.startswith("torch"):
        raise RuntimeError("train_until_golden_ratio currently requires the torch backend.")

    import torch
    import torch.optim as optim

    dev = torch.device(device or backend.device or "cpu")
    model.to(dev)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    best_val = float("inf")
    wait = 0
    for _ in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                out = model(xb)
                val_losses.append(float(loss_fn(out, yb).item()))
        L_val = sum(val_losses) / max(1, len(val_losses))
        if L_val < golden_threshold:
            break
        if L_val < best_val - 1e-6:
            best_val = L_val
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return model, best_val


__all__ = ["train_until_golden_ratio", "GOLDEN_THRESHOLD"]
