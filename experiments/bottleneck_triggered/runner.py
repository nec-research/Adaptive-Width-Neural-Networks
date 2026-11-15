"""
Unified Runner for Bottleneck-Triggered Evidence-Based Neuron Reordering.

Executes controlled training of AWN models with bottleneck-triggered interventions,
enforces strict resource budgets, logs diagnostic histories, and evaluates post-hoc curves.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .common_utils import get_awn_config, load_dataset_splits
from metric import ELBO_Classification
from model import AWN

from .accounting.resource_ledger import ResourceLedger
from .evaluation.metrics import (
    compute_auc_from_prefix_curve,
    evaluate_model_performance,
    evaluate_post_hoc_prefix_curve,
)
from .methods.bottleneck_detector import BottleneckDetector
from .methods.evidence_base import BaseEvidenceProvider
from .methods.v0_weight_evidence import V0WeightEvidence
from .methods.v1_gradient_evidence import V1GradientEvidence
from .methods.v2_graph_evidence import V2GraphEvidence
from .permutation.permutation_manager import PermutationManager


def create_evidence_provider(method_name: str, **kwargs) -> Optional[BaseEvidenceProvider]:
    """Factory to instantiate the appropriate evidence provider."""
    if method_name == "V0-BT":
        return V0WeightEvidence()
    elif method_name == "V1-BT":
        num_batches = kwargs.get("num_batches", 5)
        return V1GradientEvidence(num_batches=num_batches)
    elif method_name == "V2-BT":
        return V2GraphEvidence(
            edge_weight_threshold=kwargs.get("edge_weight_threshold", 0.05),
            w_struct=kwargs.get("w_struct", 0.4),
            w_in=kwargs.get("w_in", 0.3),
            w_out=kwargs.get("w_out", 0.3),
        )
    elif method_name in ("AWN_Original", "Baseline"):
        return None
    else:
        raise ValueError(f"Unknown method name: {method_name}")


def run_single_experiment(
    method_name: str,
    dataset_name: str,
    seed: int,
    epochs: int = 50,
    lr: float = 0.01,
    bottleneck_config: Optional[Dict[str, Any]] = None,
    evidence_config: Optional[Dict[str, Any]] = None,
    reference_ledger: Optional[ResourceLedger] = None,
    device: torch.device = torch.device("cpu"),
    tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """Runs a single controlled experiment for AWN_Original, V0-BT, V1-BT, or V2-BT.

    Args:
        method_name: 'AWN_Original', 'V0-BT', 'V1-BT', or 'V2-BT'.
        dataset_name: 'DoubleMoon', 'Spiral', or 'SpiralHard'.
        seed: Random seed for initialization and data loading.
        epochs: Maximum allowed training epochs.
        lr: Learning rate for Adam optimizer.
        bottleneck_config: Hyperparameters for BottleneckDetector.
        evidence_config: Hyperparameters for EvidenceProvider.
        reference_ledger: Resource budget from baseline AWN run to enforce strict fairness.
        device: Torch device (CPU or CUDA).
        tolerance: Equivalence tolerance for full-width model output check.

    Returns:
        Structured experiment result dictionary.
    """
    # 1. Deterministic seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Load dataset splits
    train_loader, val_loader, test_loader, n_obs = load_dataset_splits(dataset_name)

    # 3. Instantiate original AWN model (with learned ordering disabled)
    cfg = get_awn_config(n_obs, enable_learned_ordering=False)
    model = AWN(dim_input_features=2, dim_target=2, config=cfg)
    model.to(device)

    # 4. Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.torch_optimizer = optimizer
    elbo_fn = ELBO_Classification(n_obs)

    # 5. Resource ledger setup
    ledger = ResourceLedger(name=f"{method_name}_{dataset_name}_seed{seed}")
    if reference_ledger is not None:
        # Match the baseline's training updates and examples exactly.
        # Evidence has genuine extra FLOPs; account for that cost rather than
        # cutting a method's training short at a synthetic FLOP ceiling.
        ledger.max_optimizer_steps = reference_ledger.optimizer_steps
        ledger.max_examples_processed = reference_ledger.examples_processed_train

    # 6. Bottleneck detector & Permutation manager
    bt_cfg = bottleneck_config or {}
    detector = BottleneckDetector(
        plateau_window=bt_cfg.get("plateau_window", 5),
        min_relative_improvement=bt_cfg.get("min_relative_improvement", 0.005),
        patience=bt_cfg.get("patience", 3),
        min_progress_epochs=bt_cfg.get("min_progress_epochs", 10),
        min_interval_between_interventions=bt_cfg.get("min_interval_between_interventions", 10),
    )
    perm_mgr = PermutationManager(tolerance=tolerance)
    evidence_provider = create_evidence_provider(method_name, **(evidence_config or {}))
    # An independent, non-shuffled loader makes evidence deterministic without
    # consuming the RNG used for the main training loader's batch order.
    evidence_loader = (
        DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False)
        if isinstance(evidence_provider, V1GradientEvidence) else None
    )

    # Tracking variables
    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    val_acc_history: List[float] = []
    width_history: List[int] = []
    stopped_early_due_to_budget: bool = False

    # Get sample validation batch for functional equivalence check
    sample_val_x, sample_val_y = next(iter(val_loader))
    sample_val_x = sample_val_x.to(device)
    sample_val_y = sample_val_y.to(device).squeeze().long()

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    start_wall_clock = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # A. Check budget before epoch
        if reference_ledger is not None and ledger.is_budget_exhausted():
            stopped_early_due_to_budget = True
            break

        model.train()
        epoch_train_loss = 0.0
        total_train_samples = 0

        for x_b, y_b in train_loader:
            if reference_ledger is not None and ledger.is_budget_exhausted():
                stopped_early_due_to_budget = True
                break

            x_b = x_b.to(device)
            y_b = y_b.to(device)
            if y_b.ndim == 1:
                y_b = y_b.unsqueeze(1)

            bsz = x_b.shape[0]
            param_count = sum(p.numel() for p in model.parameters())

            t0_fwd = time.perf_counter()
            optimizer.zero_grad()
            outs = model(x_b)
            preds, tgt = elbo_fn.get_predictions_and_targets(y_b, *outs)
            loss = elbo_fn.compute_metric(tgt, preds)
            t_fwd = time.perf_counter() - t0_fwd

            t0_bwd = time.perf_counter()
            loss.backward()
            optimizer.step()
            t_bwd = time.perf_counter() - t0_bwd

            ledger.record_forward_train(batch_size=bsz, param_count=param_count, duration_sec=t_fwd)
            ledger.record_backward_train(batch_size=bsz, param_count=param_count, duration_sec=t_bwd)
            ledger.record_optimizer_step()

            epoch_train_loss += loss.item() * bsz
            total_train_samples += bsz

        if stopped_early_due_to_budget:
            break

        mean_train_loss = epoch_train_loss / max(1, total_train_samples)
        train_loss_history.append(mean_train_loss)

        # B. Validation evaluation
        t0_val = time.perf_counter()
        val_loss, val_acc = evaluate_model_performance(model, val_loader, elbo_fn, device=device)
        t_val = time.perf_counter() - t0_val
        val_samples = len(val_loader.dataset)
        ledger.record_val_eval(batch_size=val_samples, duration_sec=t_val)

        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        curr_width = model.variational_Ws[0].compute_truncation_number()
        width_history.append(curr_width)

        # C. Bottleneck Evaluation and Intervention
        if evidence_provider is not None:
            should_trigger, event = detector.record_and_evaluate(epoch, val_loss)
            if should_trigger:
                physical_w = model.dyn_model.get_layer(0).weight.shape[0] if hasattr(model, "dyn_model") else curr_width
                eff_w = min(curr_width, physical_w)
                # 1. Compute evidence scores
                scores = evidence_provider.compute_neuron_scores(
                    model=model,
                    layer_id=0,
                    width=eff_w,
                    dataloader=evidence_loader if evidence_loader is not None else train_loader,
                    device=device,
                    ledger=ledger,
                )

                # 2. Permute layer with verification of full-width equivalence
                intervention_rec = perm_mgr.permute_layer(
                    model=model,
                    layer_id=0,
                    scores=scores,
                    optimizer=optimizer,
                    epoch=epoch,
                    method_name=method_name,
                    val_sample_x=sample_val_x,
                    val_sample_y=sample_val_y,
                    criterion=nn.CrossEntropyLoss(),
                )
                ledger.record_reordering(
                    duration_sec=intervention_rec.reordering_duration_sec,
                    migration_duration_sec=intervention_rec.optimizer_migration_duration_sec,
                )

    ledger.stop_and_finalize()

    # -------------------------------------------------------------------------
    # Final Evaluation & Post-Hoc Truncation
    # -------------------------------------------------------------------------
    final_width = model.variational_Ws[0].compute_truncation_number()
    final_rate = model.variational_Ws[0].discretized_distribution.base_distribution.rate.item()
    test_loss, test_acc = evaluate_model_performance(model, test_loader, elbo_fn, device=device)

    # Post-hoc prefix curves
    prefix_curve = evaluate_post_hoc_prefix_curve(
        model=model,
        test_loader=test_loader,
        active_width=final_width,
        device=device,
    )
    auc = compute_auc_from_prefix_curve(prefix_curve)

    # Latency test
    sample_x = torch.randn(100, 2, device=device)
    t0_lat = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            _ = model(sample_x, skip_update=True)
    latency_ms = (time.perf_counter() - t0_lat) / 50.0 * 1000.0

    param_count = sum(p.numel() for p in model.parameters())

    return {
        "method_name": method_name,
        "dataset_name": dataset_name,
        "seed": seed,
        "epochs_completed": len(train_loss_history),
        "stopped_early_due_to_budget": stopped_early_due_to_budget,
        "final_test_acc": round(test_acc, 4),
        "final_test_loss": round(test_loss, 4),
        "final_width": final_width,
        "final_rate": round(final_rate, 6),
        "param_count": param_count,
        "latency_ms": round(latency_ms, 3),
        "auc_prefix_curve": auc,
        "prefix_curve": prefix_curve,
        "total_interventions": detector.total_interventions,
        "interventions": [rec.to_dict() for rec in perm_mgr.intervention_history],
        "diagnostics": detector.get_diagnostic_summary(),
        "resource_ledger": ledger.to_dict(),
        "train_loss_history": [round(l, 4) for l in train_loss_history],
        "val_loss_history": [round(l, 4) for l in val_loss_history],
        "val_acc_history": [round(a, 4) for a in val_acc_history],
        "width_history": width_history,
    }
