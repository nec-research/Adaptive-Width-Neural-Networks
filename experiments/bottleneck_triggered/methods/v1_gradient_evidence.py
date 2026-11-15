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
        was_training = model.training
        # Capture the exact hidden representation consumed by the next
        # linear layer. In AWN this includes its position-dependent qW gate,
        # unlike a surrogate fc1 -> activation -> fc2 computation.
        if hasattr(model, "dyn_model"):
            next_layer = model.dyn_model.get_layer(layer_id + 1)
            is_awn = True
        elif hasattr(model, "fc1") and hasattr(model, "fc2") and layer_id == 0:
            next_layer = model.fc2
            is_awn = False
        else:
            raise ValueError(f"Unsupported model architecture: {type(model)}")

        captured = {}

        def capture_input(_module, inputs):
            captured["activation"] = inputs[0]

        hook = next_layer.register_forward_pre_hook(capture_input)
        accumulated_scores = torch.zeros(width, device=device)
        samples_processed = 0
        loss_fn = nn.CrossEntropyLoss()
        try:
            model.eval()
            with torch.enable_grad():
                for batch_idx, (x_b, y_b) in enumerate(dataloader):
                    if batch_idx >= self.num_batches:
                        break

                    x_b = x_b.to(device)
                    y_b = y_b.to(device).reshape(-1).long()
                    bsz = x_b.shape[0]

                    t_pass0 = time.perf_counter()
                    captured.clear()
                    # AWN needs skip_update to avoid resizing parameters while
                    # measuring evidence; its forward still applies qW gating.
                    if is_awn:
                        outs = model(x_b, skip_update=True)
                        logits = outs[0]
                    else:
                        logits = model(x_b)

                    if "activation" not in captured:
                        raise RuntimeError("Failed to capture hidden activation")
                    h = captured["activation"]
                    w_eff = min(width, h.shape[-1])
                    if w_eff <= 0:
                        raise ValueError("Cannot rank neurons in an empty layer")

                    loss = loss_fn(logits, y_b)
                    (grad_h,) = torch.autograd.grad(loss, h)
                    # CE is batch-averaged; undo that factor to aggregate
                    # consistently by the number of examples across batches.
                    contribution = (h.detach()[:, :w_eff] * grad_h.detach()[:, :w_eff]).abs()
                    accumulated_scores[:w_eff] += contribution.sum(dim=0) * bsz
                    samples_processed += bsz

                    if ledger is not None:
                        duration = time.perf_counter() - t_pass0
                        param_count = sum(p.numel() for p in model.parameters())
                        ledger.record_evidence_pass(
                            batch_size=bsz, param_count=param_count,
                            is_backward=False, duration_sec=duration / 2.0,
                        )
                        ledger.record_evidence_pass(
                            batch_size=bsz, param_count=param_count,
                            is_backward=True, duration_sec=duration / 2.0,
                        )
        finally:
            hook.remove()
            model.train(was_training)

        scores = accumulated_scores / max(1, samples_processed)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)

        # Population std avoids NaN for a one-neuron layer.
        std = scores.std(unbiased=False) + 1e-8
        normalized = (scores - scores.mean()) / std
        return normalized - normalized.min() + 1e-6
