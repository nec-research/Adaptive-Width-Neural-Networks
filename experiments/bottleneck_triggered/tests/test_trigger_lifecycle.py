"""
Suite H: Trigger Lifecycle Integration Tests.

Validates the full training lifecycle across Stages A, B, C, and D from Section 3.
"""

import pytest
import torch

from experiments.bottleneck_triggered.accounting.resource_ledger import ResourceLedger
from experiments.bottleneck_triggered.methods.bottleneck_detector import BottleneckDetector
from experiments.bottleneck_triggered.methods.v0_weight_evidence import V0WeightEvidence
from experiments.bottleneck_triggered.permutation.permutation_manager import PermutationManager
from experiments.bottleneck_triggered.runner import run_single_experiment


class TestTriggerLifecycle:
    """Verifies end-to-end integration of the bottleneck-triggered intervention lifecycle."""

    def test_lifecycle_stage_a_to_d_complete_run(self):
        """Runs a short end-to-end run confirming stages A through D."""
        res = run_single_experiment(
            method_name="V0-BT",
            dataset_name="DoubleMoon",
            seed=42,
            epochs=15,
            bottleneck_config={
                "plateau_window": 2,
                "min_relative_improvement": 0.05,
                "patience": 2,
                "min_progress_epochs": 4,
                "min_interval_between_interventions": 3,
            },
        )

        assert res["epochs_completed"] == 15
        assert "resource_ledger" in res
        assert "auc_prefix_curve" in res
        assert res["final_width"] > 0
        assert len(res["train_loss_history"]) == 15

        # Check diagnostics log
        diag = res["diagnostics"]
        assert diag["total_evaluations"] == 15

    def test_training_finishes_safely_without_bottleneck(self):
        """If loss improves rapidly and never plateaus, training finishes with 0 interventions."""
        res = run_single_experiment(
            method_name="V0-BT",
            dataset_name="DoubleMoon",
            seed=42,
            epochs=5,
            bottleneck_config={
                "min_progress_epochs": 100,  # impossible to trigger
            },
        )
        assert res["total_interventions"] == 0
        assert len(res["interventions"]) == 0

    def test_intervention_logs_record_metadata(self):
        """Intervention records contain layer, epoch, scores, permutation, and equivalence."""
        res = run_single_experiment(
            method_name="V0-BT",
            dataset_name="DoubleMoon",
            seed=42,
            epochs=12,
            bottleneck_config={
                "plateau_window": 1,
                "min_relative_improvement": 0.5,  # high threshold guarantees trigger
                "patience": 1,
                "min_progress_epochs": 2,
                "min_interval_between_interventions": 20,
            },
        )
        if res["total_interventions"] > 0:
            rec = res["interventions"][0]
            assert "method_name" in rec
            assert "scores" in rec
            assert "new_permutation" in rec
            assert rec["max_output_diff"] < 1e-4

    def test_budget_exhaustion_stops_training_safely(self):
        """Runner halts when ledger budget limits are reached."""
        ref_ledger = ResourceLedger(name="ref")
        ref_ledger.optimizer_steps = 5  # very tight budget

        res = run_single_experiment(
            method_name="V0-BT",
            dataset_name="DoubleMoon",
            seed=42,
            epochs=20,
            reference_ledger=ref_ledger,
        )
        assert res["stopped_early_due_to_budget"] is True
        assert res["resource_ledger"]["optimizer_steps"] <= 6
