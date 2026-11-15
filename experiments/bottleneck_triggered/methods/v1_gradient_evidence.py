"""
V1-BT: Bottleneck-Triggered Gradient-Based Neuron Reordering.

Derives neuron importance scores from first-order Taylor expansion / gradient-activation product:
    S_j^{V1} = E_{(x,y) ~ B} [ | h_j(x) * (d L / d h_j(x)) | ]

Aggregates over a configurable window of training batches without modifying model parameters
or leaking optimizer updates.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .evidence_base import BaseEvidenceProvider
from ..accounting.resource_ledger import ResourceLedger


class V1GradientEvidence(BaseEvidenceProvider):
    """Computes first-order gradient-activation Taylor evidence over multiple training batches."""

    def __init__(self, num_batches: int = 5):
        super().__init__(name="V1-BT")
        self.num_batches = max(1, num_batches)

    def compute_neuron_scores(
        self,
        model: nn.Module,
        layer_id: int,
        width: int,
        dataloader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu"),
        ledger: Optional[ResourceLedger] = None,
    ) -> Tensor:
        """Computes S_j^{V1} averaged over `num_batches` training batches."""
        if dataloader is None:
            raise ValueError("V1GradientEvidence requires a training dataloader.")

        t0 = time.perf_counter()
        model.eval()  # Do not update BatchNorm / Dropout during evidence gathering

        # Preserve any existing parameter gradients so we don't destroy active state
        saved_grads = {p: p.grad.clone() if p.grad is not None else None for p in model.parameters()}
        model.zero_grad()

        accumulated_scores = torch.zeros(width, device=device)
        batches_processed = 0

        # Loss function for gradient evaluation on training batches
        loss_fn = nn.CrossEntropyLoss()

        for batch_idx, (x_b, y_b) in enumerate(dataloader):
            if batch_idx >= self.num_batches:
                break

            x_b = x_b.to(device)
            y_b = y_b.to(device).squeeze().long()
            bsz = x_b.shape[0]

            t_pass0 = time.perf_counter()

            # Hook into activations of layer_id
            act_container = {}

            if hasattr(model, "dyn_model"):
                curr_layer = model.dyn_model.get_layer(layer_id)
                # Forward pass tracking activations
                pre_h = curr_layer(x_b)
                h = model.dyn_model.activation(pre_h)
            elif hasattr(model, "fc1") and layer_id == 0:
                pre_h = model.fc1(x_b)
                h = model.activation(pre_h)
            else:
                raise ValueError(f"Unsupported model architecture: {type(model)}")

            W_eff = min(width, h.shape[1])
            h_sub = h[:, :W_eff]
            h_sub.retain_grad()

            # Forward remaining layers
            if hasattr(model, "dyn_model"):
                next_layer = model.dyn_model.get_layer(layer_id + 1)
                w_next = next_layer.weight[:, :W_eff]
                logits = h_sub @ w_next.T
                if next_layer.bias is not None:
                    logits = logits + next_layer.bias
            elif hasattr(model, "fc2") and layer_id == 0:
                w_next = model.fc2.weight[:, :W_eff]
                logits = h_sub @ w_next.T
                if model.fc2.bias is not None:
                    logits = logits + model.fc2.bias
            else:
                raise ValueError("Architecture forwarding not supported")

            loss = loss_fn(logits, y_b)
            loss.backward()

            t_pass = time.perf_counter() - t_pass0

            # Collect |h * grad|
            if h_sub.grad is not None:
                grad = h_sub.grad.detach()
                taylor_signal = (h_sub.detach() * grad).abs().mean(dim=0)  # [W_eff]
                accumulated_scores[:W_eff] += taylor_signal
            else:
                # Fallback if gradient is missing
                accumulated_scores[:W_eff] += 1e-4

            batches_processed += 1

            if ledger is not None:
                param_count = sum(p.numel() for p in model.parameters())
                ledger.record_evidence_pass(batch_size=bsz, param_count=param_count, is_backward=False, duration_sec=t_pass / 2.0)
                ledger.record_evidence_pass(batch_size=bsz, param_count=param_count, is_backward=True, duration_sec=t_pass / 2.0)

            # Ensure gradients are wiped between evidence batches
            model.zero_grad()

        # Restore original parameter gradients
        for p, g in saved_grads.items():
            p.grad = g

        if batches_processed > 0:
            scores = accumulated_scores / float(batches_processed)
        else:
            scores = accumulated_scores

        # Clean non-finite entries
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)

        # Layer normalization for numerical stability
        std = scores.std() + 1e-8
        mean = scores.mean()
        normalized_scores = (scores - mean) / std
        # Shift to positive domain
        normalized_scores = normalized_scores - normalized_scores.min() + 1e-6

        return normalized_scores
