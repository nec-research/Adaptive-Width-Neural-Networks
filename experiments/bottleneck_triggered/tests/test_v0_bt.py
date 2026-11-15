"""
Suite C: V0-BT Unit Tests.

Verifies mathematical correctness and invariants for weight-magnitude evidence.
"""

import pytest
import torch
import torch.nn as nn

from experiments.bottleneck_triggered.methods.v0_weight_evidence import V0WeightEvidence
from experiments.bottleneck_triggered.permutation.permutation_manager import PermutationManager


class DummyTwoLayerNet(nn.Module):
    def __init__(self, in_dim=2, hidden=4, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)


class TestV0BT:
    """Tests for V0-BT weight-based evidence."""

    def test_score_matches_formula_exact(self):
        """S_j^{V0} = ||W_j^{in}||_2 * ||W_j^{out}||_2 exactly."""
        model = DummyTwoLayerNet(in_dim=2, hidden=3, out_dim=2)
        # Custom weights
        # fc1: 3 neurons, 2 inputs
        model.fc1.weight.data = torch.tensor([
            [3.0, 4.0],   # norm = 5.0
            [1.0, 0.0],   # norm = 1.0
            [0.0, 2.0],   # norm = 2.0
        ])
        # fc2: 2 outputs, 3 inputs
        model.fc2.weight.data = torch.tensor([
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
        ])
        # Col 0: [2, 0] -> norm = 2.0 -> Score 0 = 5.0 * 2.0 = 10.0
        # Col 1: [0, 3] -> norm = 3.0 -> Score 1 = 1.0 * 3.0 = 3.0
        # Col 2: [1, 0] -> norm = 1.0 -> Score 2 = 2.0 * 1.0 = 2.0

        provider = V0WeightEvidence()
        scores = provider.compute_neuron_scores(model, layer_id=0, width=3)

        expected = torch.tensor([10.0, 3.0, 2.0])
        assert torch.allclose(scores, expected, atol=1e-5)

    def test_sorting_direction_descending(self):
        """Neuron with highest norm product must be permuted to rank index 0."""
        scores = torch.tensor([2.5, 9.8, 1.1, 7.4])
        perm = PermutationManager.compute_permutation_from_scores(scores)
        # Expected order: index 1 (9.8), index 3 (7.4), index 0 (2.5), index 2 (1.1)
        expected_perm = torch.tensor([1, 3, 0, 2])
        assert torch.equal(perm, expected_perm)

    def test_ties_handled_deterministically(self):
        """Identical scores must produce deterministic permutation order."""
        scores = torch.tensor([5.0, 5.0, 5.0, 5.0])
        perm1 = PermutationManager.compute_permutation_from_scores(scores)
        perm2 = PermutationManager.compute_permutation_from_scores(scores)
        assert torch.equal(perm1, perm2)

    def test_non_finite_weights_handled_safely(self):
        """NaN or Inf weights must not produce crashing permutations."""
        model = DummyTwoLayerNet(in_dim=2, hidden=3, out_dim=2)
        model.fc1.weight.data[0, 0] = float("nan")
        model.fc2.weight.data[0, 1] = float("inf")

        provider = V0WeightEvidence()
        scores = provider.compute_neuron_scores(model, layer_id=0, width=3)
        assert torch.all(torch.isfinite(scores))

        perm = PermutationManager.compute_permutation_from_scores(scores)
        assert PermutationManager.validate_permutation(perm, 3)
