"""
Evaluation Metrics for Bottleneck-Triggered AWN Experiments.

Calculates test accuracy, ELBO loss, active width, post-hoc prefix truncation curves,
area under the accuracy-versus-width curve (AUC), and intervention statistics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from metric import ELBO_Classification


def evaluate_model_performance(
    model: nn.Module,
    data_loader: DataLoader,
    elbo_fn: ELBO_Classification,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Computes mean loss and classification accuracy over a DataLoader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for x_b, y_b in data_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            if y_b.ndim == 1:
                y_b = y_b.unsqueeze(1)

            outs = model(x_b, skip_update=True)
            preds, tgt = elbo_fn.get_predictions_and_targets(y_b, *outs)
            loss = elbo_fn.compute_metric(tgt, preds)

            logits = outs[0]
            pred_labels = logits.argmax(dim=-1)
            correct += (pred_labels == y_b.squeeze()).sum().item()

            total_loss += loss.item() * len(x_b)
            total_samples += len(x_b)

    mean_loss = total_loss / max(1, total_samples)
    accuracy = correct / max(1, total_samples)
    return mean_loss, accuracy


def evaluate_post_hoc_prefix_curve(
    model: nn.Module,
    test_loader: DataLoader,
    active_width: int,
    device: torch.device = torch.device("cpu"),
    fractions: List[float] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
) -> Dict[str, float]:
    """Evaluates prefix truncation accuracy across fractions p in [1.0 ... 0.1].

    In AWN, active neurons are indexed 0..active_width-1.
    For fraction p, the first ceil(p * active_width) neurons are retained,
    and remaining neurons are masked out.
    """
    model.eval()
    results: Dict[str, float] = {}
    W = active_width

    dyn = model.dyn_model
    layer0 = dyn.get_layer(0)
    layer1 = dyn.get_layer(1)

    for frac in fractions:
        k = max(1, int(round(frac * W)))
        k = min(k, W)

        correct = 0
        total = 0

        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b = x_b.to(device)
                y_b = y_b.to(device).squeeze()

                pre_h = layer0(x_b)
                h = dyn.activation(pre_h)
                physical_w = h.shape[1]
                eff_w = min(W, physical_w)

                q_probs = model.qW_probs[0][:eff_w]

                # Prefix mask: 1 for index < k, 0 for index >= k
                mask = torch.zeros(eff_w, device=device)
                mask[:min(k, eff_w)] = 1.0

                h_active = h[:, :eff_w] * q_probs.unsqueeze(0) * mask.unsqueeze(0)
                w1_eff = layer1.weight[:, :eff_w]
                out = h_active @ w1_eff.T
                if layer1.bias is not None:
                    out = out + layer1.bias

                pred = out.argmax(dim=-1)
                correct += (pred == y_b).sum().item()
                total += len(x_b)

        acc = correct / max(1, total)
        results[f"{int(round(frac * 100))}%"] = round(acc, 4)

    return results


def compute_auc_from_prefix_curve(prefix_curve: Dict[str, float]) -> float:
    """Computes Area Under the Accuracy-versus-Width Curve via trapezoidal integration."""
    fracs = []
    accs = []
    for key, val in prefix_curve.items():
        pct = float(key.replace("%", "")) / 100.0
        fracs.append(pct)
        accs.append(val)

    # Sort in ascending order of fraction
    sorted_pairs = sorted(zip(fracs, accs), key=lambda p: p[0])
    x = [p[0] for p in sorted_pairs]
    y = [p[1] for p in sorted_pairs]

    auc = float(np.trapezoid(y, x))
    return round(auc, 4)
