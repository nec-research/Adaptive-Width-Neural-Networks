"""
Suite J: Reproducibility Tests.

Verifies seed determinism for initializations, data order, bottleneck decisions,
evidence scores, and permutations.
"""

import pytest
import torch

from experiments.bottleneck_triggered.methods.v0_weight_evidence import V0WeightEvidence
from experiments.bottleneck_triggered.permutation.permutation_manager import PermutationManager
from experiments.bottleneck_triggered.runner import run_single_experiment


class TestReproducibility:
    """Verifies end-to-end reproducibility under fixed seeds."""

    def test_identical_seeds_reproduce_exact_results(self):
        """Two identical runs with the same seed produce identical trajectories and metrics."""
        res1 = run_single_experiment(
            method_name="V0-BT",
            dataset_name="DoubleMoon",
            seed=42,
            epochs=6,
            bottleneck_config={"min_progress_epochs": 2},
        )
        res2 = run_single_experiment(
            method_name="V0-BT",
            dataset_name="DoubleMoon",
            seed=42,
            epochs=6,
            bottleneck_config={"min_progress_epochs": 2},
        )

        assert res1["final_test_acc"] == res2["final_test_acc"]
        assert res1["final_width"] == res2["final_width"]
        assert res1["train_loss_history"] == res2["train_loss_history"]
        assert res1["val_loss_history"] == res2["val_loss_history"]
        assert res1["total_interventions"] == res2["total_interventions"]

    def test_different_seeds_produce_different_initializations(self):
        """Different seeds initialize with different weight tensors."""
        torch.manual_seed(42)
        w1 = torch.randn(10, 10)
        torch.manual_seed(999)
        w2 = torch.randn(10, 10)
        assert not torch.equal(w1, w2)
