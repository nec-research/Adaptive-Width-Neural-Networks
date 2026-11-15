"""
Suite D: V1-BT Unit Tests.

Verifies gradient-activation Taylor evidence collection, aggregation, parameter isolation,
and resilience against zero or non-finite gradient states.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.bottleneck_triggered.methods.v1_gradient_evidence import V1GradientEvidence
from experiments.bottleneck_triggered.permutation.permutation_manager import PermutationManager


class DummyTwoLayerNet(nn.Module):
    def __init__(self, in_dim=2, hidden=4, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class TestV1BT:
    """Tests for V1-BT gradient-activation Taylor evidence."""

    def test_gradient_evidence_computation_and_mapping(self):
        """Derives positive scores for active neurons matching hidden width."""
        model = DummyTwoLayerNet(in_dim=2, hidden=5, out_dim=2)
        x = torch.randn(20, 2)
        y = torch.randint(0, 2, (20,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        provider = V1GradientEvidence(num_batches=2)
        scores = provider.compute_neuron_scores(model, layer_id=0, width=5, dataloader=loader)

        assert scores.shape == (5,)
        assert torch.all(torch.isfinite(scores))
        assert torch.all(scores >= 0)

    def test_evidence_collection_does_not_modify_parameters(self):
        """Model parameters must remain strictly identical before and after evidence collection."""
        model = DummyTwoLayerNet(in_dim=2, hidden=4, out_dim=2)
        w1_before = model.fc1.weight.clone()
        w2_before = model.fc2.weight.clone()

        x = torch.randn(10, 2)
        y = torch.randint(0, 2, (10,))
        loader = DataLoader(TensorDataset(x, y), batch_size=5)

        provider = V1GradientEvidence(num_batches=2)
        _ = provider.compute_neuron_scores(model, layer_id=0, width=4, dataloader=loader)

        assert torch.equal(model.fc1.weight, w1_before)
        assert torch.equal(model.fc2.weight, w2_before)

    def test_evidence_collection_does_not_leak_parameter_gradients(self):
        """Parameter gradients must be restored or zeroed after collection."""
        model = DummyTwoLayerNet(in_dim=2, hidden=4, out_dim=2)
        x = torch.randn(10, 2)
        y = torch.randint(0, 2, (10,))
        loader = DataLoader(TensorDataset(x, y), batch_size=5)

        provider = V1GradientEvidence(num_batches=1)
        _ = provider.compute_neuron_scores(model, layer_id=0, width=4, dataloader=loader)

        for p in model.parameters():
            assert p.grad is None or torch.all(p.grad == 0)

    def test_zero_and_missing_gradients_handled_safely(self):
        """Zero or unpropagated gradients produce safe non-crashing scores."""
        model = DummyTwoLayerNet(in_dim=2, hidden=4, out_dim=2)
        # Zero out outgoing weights for neuron 0 so it receives zero gradient
        model.fc2.weight.data[:, 0] = 0.0

        x = torch.randn(10, 2)
        y = torch.randint(0, 2, (10,))
        loader = DataLoader(TensorDataset(x, y), batch_size=5)

        provider = V1GradientEvidence(num_batches=1)
        scores = provider.compute_neuron_scores(model, layer_id=0, width=4, dataloader=loader)

        assert torch.all(torch.isfinite(scores))
        perm = PermutationManager.compute_permutation_from_scores(scores)
        assert PermutationManager.validate_permutation(perm, 4)

    def test_evidence_restores_original_training_mode(self):
        model = DummyTwoLayerNet()
        model.train()
        x = torch.randn(8, 2)
        y = torch.randint(0, 2, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        _ = V1GradientEvidence(num_batches=1).compute_neuron_scores(
            model, layer_id=0, width=4, dataloader=loader
        )
        assert model.training

    def test_reproducibility_with_identical_inputs(self):
        """Same model state and data batches produce identical scores."""
        torch.manual_seed(42)
        model = DummyTwoLayerNet(in_dim=2, hidden=4, out_dim=2)
        x = torch.randn(10, 2)
        y = torch.randint(0, 2, (10,))
        loader = DataLoader(TensorDataset(x, y), batch_size=5, shuffle=False)

        provider = V1GradientEvidence(num_batches=2)
        scores1 = provider.compute_neuron_scores(model, layer_id=0, width=4, dataloader=loader)
        scores2 = provider.compute_neuron_scores(model, layer_id=0, width=4, dataloader=loader)

        assert torch.allclose(scores1, scores2, atol=1e-5)
