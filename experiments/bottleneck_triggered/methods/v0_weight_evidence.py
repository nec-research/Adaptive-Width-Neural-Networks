"""
V0-BT: Bottleneck-Triggered Weight-Based Neuron Reordering.

Derives neuron importance scores from the product of incoming and outgoing weight L2 norms:
    S_j^{V0} = ||W_j^{in}||_2 * ||W_j^{out}||_2

Adheres strictly to the historical magnitude evidence definition from Section 4.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from .evidence_base import BaseEvidenceProvider
from ..accounting.resource_ledger import ResourceLedger


class V0WeightEvidence(BaseEvidenceProvider):
    """Computes weight-magnitude evidence for neurons in a specified layer."""

    def __init__(self):
        super().__init__(name="V0-BT")

    def compute_neuron_scores(
        self,
        model: nn.Module,
        layer_id: int,
        width: int,
        dataloader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu"),
        ledger: Optional[ResourceLedger] = None,
    ) -> Tensor:
        """Computes S_j^{V0} = ||W_j^{in}||_2 * ||W_j^{out}||_2 for j in 0..width-1."""
        t0 = time.perf_counter()

        # Locate current layer and next layer
        if hasattr(model, "dyn_model"):
            curr_layer = model.dyn_model.get_layer(layer_id)
            next_layer = model.dyn_model.get_layer(layer_id + 1)
        elif hasattr(model, "fc1") and hasattr(model, "fc2") and layer_id == 0:
            curr_layer = model.fc1
            next_layer = model.fc2
        else:
            raise ValueError(f"Unsupported model architecture: {type(model)}")

        w_curr = curr_layer.weight.detach()
        w_next = next_layer.weight.detach()

        W = min(width, w_curr.shape[0])

        # Incoming weight norm: ||W_j^{in}||_2 (L2 norm over incoming dimension)
        # w_curr[:W, :] has shape [W, d_in] -> norm(dim=1) gives shape [W]
        norm_in = torch.norm(w_curr[:W, :], p=2, dim=1)

        # Outgoing weight norm: ||W_j^{out}||_2 (L2 norm over outgoing dimension)
        # w_next[:, :W] has shape [d_out, W] -> norm(dim=0) gives shape [W]
        norm_out = torch.norm(w_next[:, :W], p=2, dim=0)

        scores = norm_in * norm_out

        # Handle non-finite values safely
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)

        duration = time.perf_counter() - t0
        if ledger is not None:
            # Flops for norms: 2 * elements in w_curr[:W] + 2 * elements in w_next[:, :W] + W
            est_flops = 2.0 * w_curr[:W, :].numel() + 2.0 * w_next[:, :W].numel() + float(W)
            ledger.record_evidence_pass(batch_size=1, param_count=int(est_flops // 2), is_backward=False, duration_sec=duration)

        return scores
