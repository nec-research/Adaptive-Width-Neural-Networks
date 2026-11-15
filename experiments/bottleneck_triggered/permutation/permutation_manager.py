"""
Permutation Manager for Function-Preserving Physical Neuron Reordering.

Applies physical permutations to weights, biases, outgoing projections,
and associated optimizer states while guaranteeing full-width mathematical equivalence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Optimizer


@dataclass
class InterventionRecord:
    """Detailed record of a single reordering intervention."""
    intervention_id: int
    epoch: int
    layer_id: int
    method_name: str
    scores: List[float]
    old_permutation: List[int]
    new_permutation: List[int]
    max_output_diff: float
    loss_diff: float
    awn_output_diff: float
    awn_loss_diff: float
    reordering_duration_sec: float
    optimizer_migration_duration_sec: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "epoch": self.epoch,
            "layer_id": self.layer_id,
            "method_name": self.method_name,
            "scores": [round(s, 6) for s in self.scores],
            "old_permutation": self.old_permutation,
            "new_permutation": self.new_permutation,
            "max_output_diff": float(self.max_output_diff),
            "loss_diff": float(self.loss_diff),
            "awn_output_diff": float(self.awn_output_diff),
            "awn_loss_diff": float(self.awn_loss_diff),
            "reordering_duration_sec": round(self.reordering_duration_sec, 6),
            "optimizer_migration_duration_sec": round(self.optimizer_migration_duration_sec, 6),
            "timestamp": self.timestamp,
        }


class PermutationManager:
    """Manages physical neuron permutation and optimizer state migration."""

    def __init__(self, tolerance: float = 1e-4):
        self.tolerance = tolerance
        self.intervention_history: List[InterventionRecord] = []
        self._intervention_counter: int = 0

    @staticmethod
    def compute_permutation_from_scores(scores: Tensor) -> Tensor:
        """Computes a deterministic descending permutation from neuron scores.

        Higher scores receive lower rank indices (0, 1, 2, ...).
        Ties are broken deterministically using the canonical neuron index.
        """
        w = scores.shape[0]
        # Stable sorting preserves the original index for exactly equal scores,
        # including float32 scores for which 1e-9 tie breakers are ineffective.
        perm = torch.argsort(scores, descending=True, stable=True)
        return perm

    @staticmethod
    def validate_permutation(perm: Tensor, expected_size: int) -> bool:
        """Validates that a permutation is a bijection over [0, expected_size - 1]."""
        if perm.numel() != expected_size:
            return False
        unique_elems = torch.unique(perm)
        if unique_elems.numel() != expected_size:
            return False
        if unique_elems.min().item() != 0 or unique_elems.max().item() != expected_size - 1:
            return False
        return True

    def permute_layer(
        self,
        model: nn.Module,
        layer_id: int,
        scores: Tensor,
        optimizer: Optional[Optimizer] = None,
        epoch: int = 0,
        method_name: str = "unknown",
        val_sample_x: Optional[Tensor] = None,
        val_sample_y: Optional[Tensor] = None,
        criterion: Optional[Any] = None,
    ) -> InterventionRecord:
        """Physically permutes neuron weights, biases, outgoing weights, and optimizer states.

        Args:
            model: AWN model instance (or PyTorch module containing layers).
            layer_id: Index of the hidden layer to permute.
            scores: Tensor of shape [W_eff] with neuron importance scores.
            optimizer: Optimizer instance whose momentum/variance states must be permuted.
            epoch: Current training epoch.
            method_name: Name of the evidence method triggering this reordering.
            val_sample_x: Optional verification input tensor.
            val_sample_y: Optional verification target tensor.
            criterion: Optional criterion to verify loss equivalence.

        Returns:
            InterventionRecord with diagnostic details and equivalence checks.
        """
        # Determine layers to permute
        # In AWN with DynamicMLP:
        if hasattr(model, "dyn_model"):
            curr_layer = model.dyn_model.get_layer(layer_id)
            next_layer = model.dyn_model.get_layer(layer_id + 1)
        elif hasattr(model, "fc1") and hasattr(model, "fc2") and layer_id == 0:
            curr_layer = model.fc1
            next_layer = model.fc2
        else:
            raise ValueError(f"Unsupported model architecture for physical permutation: {type(model)}")

        W_curr = curr_layer.weight
        b_curr = curr_layer.bias
        W_next = next_layer.weight

        w_dim = min(scores.shape[0], W_curr.shape[0])
        scores = scores[:w_dim]

        # Record both the function-preserving no-gate equivalence and the
        # actual AWN output, whose index-dependent qW gate changes on reorder.
        was_training = model.training
        out_before = None
        awn_out_before = None
        loss_before = 0.0
        awn_loss_before = 0.0
        if val_sample_x is not None:
            model.eval()
            with torch.no_grad():
                if hasattr(model, "dyn_model"):
                    out_before, _, _ = model.dyn_model(
                        val_sample_x,
                        [torch.ones(curr_layer.weight.shape[0], device=val_sample_x.device)],
                        return_activations=False,
                    )
                else:
                    out_before = model(val_sample_x)
                if val_sample_y is not None and criterion is not None:
                    loss_before = float(criterion(out_before, val_sample_y).item())
                if hasattr(model, "dyn_model"):
                    awn_out_before = model(val_sample_x, skip_update=True)[0].detach().clone()
                    if val_sample_y is not None and criterion is not None:
                        awn_loss_before = float(criterion(awn_out_before, val_sample_y).item())

        t0_reorder = time.perf_counter()

        # Compute permutation
        perm = self.compute_permutation_from_scores(scores)
        assert self.validate_permutation(perm, w_dim), f"Invalid permutation generated: {perm}"

        old_perm_list = list(range(w_dim))
        new_perm_list = perm.cpu().tolist()

        # 1. Permute incoming weights W_curr[perm, :]
        with torch.no_grad():
            W_curr_slice = W_curr[:w_dim, :].clone()
            W_curr[:w_dim, :] = W_curr_slice[perm, :]

            # 2. Permute bias b_curr[perm]
            if b_curr is not None:
                b_curr_slice = b_curr[:w_dim].clone()
                b_curr[:w_dim] = b_curr_slice[perm]

            # 3. Permute outgoing weights W_next[:, perm]
            W_next_slice = W_next[:, :w_dim].clone()
            W_next[:, :w_dim] = W_next_slice[:, perm]

        t_reorder = time.perf_counter() - t0_reorder

        # 4. Permute optimizer state
        t0_opt = time.perf_counter()
        if optimizer is not None:
            self._permute_optimizer_states(optimizer, W_curr, b_curr, W_next, perm, w_dim)
        t_opt = time.perf_counter() - t0_opt

        # Verify equivalence
        max_diff = 0.0
        loss_diff = 0.0
        awn_output_diff = 0.0
        awn_loss_diff = 0.0
        if val_sample_x is not None:
            model.eval()
            with torch.no_grad():
                if hasattr(model, "dyn_model"):
                    out_after, _, _ = model.dyn_model(
                        val_sample_x,
                        [torch.ones(curr_layer.weight.shape[0], device=val_sample_x.device)],
                        return_activations=False,
                    )
                else:
                    out_after = model(val_sample_x)
                max_diff = float((out_after - out_before).abs().max().item())
                if val_sample_y is not None and criterion is not None:
                    loss_after = float(criterion(out_after, val_sample_y).item())
                    loss_diff = abs(loss_after - loss_before)
                if awn_out_before is not None:
                    awn_out_after = model(val_sample_x, skip_update=True)[0]
                    awn_output_diff = float((awn_out_after - awn_out_before).abs().max().item())
                    if val_sample_y is not None and criterion is not None:
                        awn_loss_diff = abs(float(criterion(awn_out_after, val_sample_y).item()) - awn_loss_before)

            if max_diff > self.tolerance:
                model.train(was_training)
                raise RuntimeError(
                    f"Full-width functional equivalence violation! Max diff {max_diff:.8f} > tolerance {self.tolerance}"
                )

        self._intervention_counter += 1
        record = InterventionRecord(
            intervention_id=self._intervention_counter,
            epoch=epoch,
            layer_id=layer_id,
            method_name=method_name,
            scores=scores.detach().cpu().tolist(),
            old_permutation=old_perm_list,
            new_permutation=new_perm_list,
            max_output_diff=max_diff,
            loss_diff=loss_diff,
            awn_output_diff=awn_output_diff,
            awn_loss_diff=awn_loss_diff,
            reordering_duration_sec=t_reorder,
            optimizer_migration_duration_sec=t_opt,
        )
        self.intervention_history.append(record)
        model.train(was_training)
        return record

    def _permute_optimizer_states(
        self,
        optimizer: Optimizer,
        W_curr: Tensor,
        b_curr: Optional[Tensor],
        W_next: Tensor,
        perm: Tensor,
        w_dim: int,
    ) -> None:
        """Permutes optimizer state tensors in-place to follow permuted neuron parameters."""
        device = perm.device

        # Permute W_curr state (rows)
        if W_curr in optimizer.state:
            state = optimizer.state[W_curr]
            for key, val in state.items():
                if isinstance(val, Tensor) and val.shape == W_curr.shape:
                    val_slice = val[:w_dim, :].clone()
                    val[:w_dim, :] = val_slice[perm, :]

        # Permute b_curr state (elements)
        if b_curr is not None and b_curr in optimizer.state:
            state = optimizer.state[b_curr]
            for key, val in state.items():
                if isinstance(val, Tensor) and val.shape == b_curr.shape:
                    val_slice = val[:w_dim].clone()
                    val[:w_dim] = val_slice[perm]

        # Permute W_next state (columns)
        if W_next in optimizer.state:
            state = optimizer.state[W_next]
            for key, val in state.items():
                if isinstance(val, Tensor) and val.shape == W_next.shape:
                    val_slice = val[:, :w_dim].clone()
                    val[:, :w_dim] = val_slice[:, perm]
