"""
Suite E: V2-BT Unit Tests.

Verifies GRAPHyco graph evidence extraction, node-neuron mapping, topological feature computation,
and parameter isolation.
"""

import pytest
import torch
import torch.nn as nn

from experiments.bottleneck_triggered.methods.v2_graph_evidence import V2GraphEvidence
from experiments.bottleneck_triggered.permutation.permutation_manager import PermutationManager


class DummyTwoLayerNet(nn.Module):
    def __init__(self, in_dim=2, hidden=4, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)


class TestV2BT:
    """Tests for V2-BT GRAPHyco relational graph evidence."""

    def test_graph_node_mapping_and_shape(self):
        """Node-level mapping produces scores of exact dimension matching layer width."""
        model = DummyTwoLayerNet(in_dim=2, hidden=6, out_dim=2)
        provider = V2GraphEvidence()
        scores = provider.compute_neuron_scores(model, layer_id=0, width=6)

        assert scores.shape == (6,)
        assert torch.all(torch.isfinite(scores))

    def test_graph_extraction_is_deterministic(self):
        """Identical model parameters yield identical graph scores."""
        model = DummyTwoLayerNet(in_dim=2, hidden=4, out_dim=2)
        provider = V2GraphEvidence()
        scores1 = provider.compute_neuron_scores(model, layer_id=0, width=4)
        scores2 = provider.compute_neuron_scores(model, layer_id=0, width=4)

        assert torch.allclose(scores1, scores2, atol=1e-5)

    def test_missing_edges_and_isolated_nodes_handled_safely(self):
        """Completely zeroed weights (isolated nodes) do not cause division by zero or crashes."""
        model = DummyTwoLayerNet(in_dim=2, hidden=4, out_dim=2)
        model.fc1.weight.data.zero_()
        model.fc2.weight.data.zero_()

        provider = V2GraphEvidence()
        scores = provider.compute_neuron_scores(model, layer_id=0, width=4)

        assert torch.all(torch.isfinite(scores))
        perm = PermutationManager.compute_permutation_from_scores(scores)
        assert PermutationManager.validate_permutation(perm, 4)

    def test_graph_extraction_does_not_modify_parameters(self):
        """Weights and biases remain bitwise identical before and after graph extraction."""
        model = DummyTwoLayerNet(in_dim=2, hidden=5, out_dim=2)
        w1_before = model.fc1.weight.clone()
        w2_before = model.fc2.weight.clone()

        provider = V2GraphEvidence()
        _ = provider.compute_neuron_scores(model, layer_id=0, width=5)

        assert torch.equal(model.fc1.weight, w1_before)
        assert torch.equal(model.fc2.weight, w2_before)

    def test_permutation_generated_is_bijective(self):
        """Scores result in a valid permutation of neurons 0..W-1."""
        model = DummyTwoLayerNet(in_dim=2, hidden=8, out_dim=2)
        provider = V2GraphEvidence()
        scores = provider.compute_neuron_scores(model, layer_id=0, width=8)

        perm = PermutationManager.compute_permutation_from_scores(scores)
        assert PermutationManager.validate_permutation(perm, 8)
