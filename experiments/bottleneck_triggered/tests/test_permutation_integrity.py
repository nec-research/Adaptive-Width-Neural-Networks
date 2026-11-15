"""
Suite F: Permutation and Model Integrity Tests.

Covers all 14 mandatory requirements from Section 8.F, verifying bijectivity,
weight/bias/outgoing permutation alignment, optimizer state tracking, and full-width equivalence.
"""

import copy
import io
import pytest
import torch
import torch.nn as nn

from experiments.bottleneck_triggered.common_utils import get_awn_config
from experiments.bottleneck_triggered.permutation.permutation_manager import PermutationManager
from model import AWN


class SimpleMLP(nn.Module):
    def __init__(self, in_features=2, hidden=8, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestPermutationIntegrity:
    """Verifies physical reordering integrity, optimizer state migration, and equivalence."""

    def test_1_2_3_permutation_is_bijective(self):
        """Permutation must be bijective with no duplicated or lost neurons."""
        scores = torch.tensor([1.2, 5.4, 0.3, 9.1, 4.4])
        perm = PermutationManager.compute_permutation_from_scores(scores)
        assert PermutationManager.validate_permutation(perm, 5)
        assert len(perm) == len(set(perm.tolist()))

    def test_4_5_6_weights_biases_outgoing_permuted_consistently(self):
        """Incoming rows, biases, and outgoing columns follow exact permutation."""
        model = SimpleMLP(in_features=2, hidden=4, out_features=2)
        perm_mgr = PermutationManager()

        w_in_orig = model.fc1.weight.clone()
        b_orig = model.fc1.bias.clone()
        w_out_orig = model.fc2.weight.clone()

        scores = torch.tensor([10.0, 40.0, 20.0, 30.0])
        rec = perm_mgr.permute_layer(model, layer_id=0, scores=scores)
        perm = rec.new_permutation  # [1, 3, 2, 0]

        # Verify incoming rows
        assert torch.equal(model.fc1.weight, w_in_orig[perm, :])
        # Verify bias elements
        assert torch.equal(model.fc1.bias, b_orig[perm])
        # Verify outgoing columns
        assert torch.equal(model.fc2.weight, w_out_orig[:, perm])

    def test_7_optimizer_states_follow_neuron_identities(self):
        """Adam momentum and second moments follow physical neuron permutations."""
        model = SimpleMLP(in_features=2, hidden=4, out_features=2)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)

        # Run 1 optimizer step to populate state tensors
        x = torch.randn(5, 2)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        # Capture old optimizer states
        w1_exp_avg = opt.state[model.fc1.weight]["exp_avg"].clone()
        b1_exp_avg = opt.state[model.fc1.bias]["exp_avg"].clone()
        w2_exp_avg = opt.state[model.fc2.weight]["exp_avg"].clone()

        scores = torch.tensor([1.0, 4.0, 2.0, 3.0])
        perm_mgr = PermutationManager()
        rec = perm_mgr.permute_layer(model, layer_id=0, scores=scores, optimizer=opt)
        perm = rec.new_permutation

        # Verify optimizer states were permuted identically to weights
        assert torch.equal(opt.state[model.fc1.weight]["exp_avg"], w1_exp_avg[perm, :])
        assert torch.equal(opt.state[model.fc1.bias]["exp_avg"], b1_exp_avg[perm])
        assert torch.equal(opt.state[model.fc2.weight]["exp_avg"], w2_exp_avg[:, perm])

    def test_8_9_full_width_output_and_loss_mathematical_equivalence(self):
        """Full-width model output and loss must be identical within 1e-5 before and after reordering."""
        torch.manual_seed(42)
        cfg = get_awn_config(n_obs=100, enable_learned_ordering=False)
        model = AWN(dim_input_features=2, dim_target=2, config=cfg)

        x_val = torch.randn(20, 2)
        y_val = torch.randint(0, 2, (20,))

        curr_layer = model.dyn_model.get_layer(0)
        W = curr_layer.weight.shape[0]

        scores = torch.rand(W)
        perm_mgr = PermutationManager(tolerance=1e-5)
        rec = perm_mgr.permute_layer(
            model=model,
            layer_id=0,
            scores=scores,
            val_sample_x=x_val,
            val_sample_y=y_val,
            criterion=nn.CrossEntropyLoss(),
        )

        assert rec.max_output_diff < 1e-5
        assert rec.loss_diff < 1e-5
        # AWN's real index-dependent qW gate is *not* generally invariant.
        assert rec.awn_output_diff >= 0.0
        assert model.training  # permutation must restore the caller's mode

    def test_awn_real_gating_change_is_reported(self):
        """Physical reordering is not a pure symmetry for AWN's positional gates."""
        torch.manual_seed(37)
        cfg = get_awn_config(n_obs=100, enable_learned_ordering=False)
        model = AWN(dim_input_features=2, dim_target=2, config=cfg)
        x = torch.randn(16, 2)
        width = model.dyn_model.get_layer(0).weight.shape[0]
        scores = torch.arange(width, dtype=torch.float32)
        rec = PermutationManager().permute_layer(
            model, layer_id=0, scores=scores, val_sample_x=x
        )
        assert rec.max_output_diff < 1e-4
        assert rec.awn_output_diff > 1e-7

    def test_ties_follow_original_indices_in_float32(self):
        scores = torch.tensor([2.0, 2.0, 1.0, 2.0], dtype=torch.float32)
        perm = PermutationManager.compute_permutation_from_scores(scores)
        assert perm.tolist() == [0, 1, 3, 2]

    def test_10_prefix_masks_select_intended_neurons_after_permutation(self):
        """Higher scoring neurons are placed in prefix indices 0..k-1."""
        model = SimpleMLP(in_features=2, hidden=5, out_features=2)
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        perm_mgr = PermutationManager()
        rec = perm_mgr.permute_layer(model, layer_id=0, scores=scores)
        # Top 2 scores are index 1 (0.9) and index 3 (0.8)
        assert rec.new_permutation[0] == 1
        assert rec.new_permutation[1] == 3

    def test_11_repeated_interventions_preserve_valid_permutations(self):
        """Successive interventions across training epochs preserve bijective permutations."""
        model = SimpleMLP(in_features=2, hidden=4, out_features=2)
        perm_mgr = PermutationManager()

        for ep in range(3):
            scores = torch.rand(4)
            rec = perm_mgr.permute_layer(model, layer_id=0, scores=scores, epoch=ep)
            assert PermutationManager.validate_permutation(torch.tensor(rec.new_permutation), 4)

    def test_12_reordering_layer_0_does_not_corrupt_other_layers(self):
        """Output layer bias is unaffected when hidden layer is permuted."""
        model = SimpleMLP(in_features=2, hidden=4, out_features=2)
        out_bias_before = model.fc2.bias.clone()

        perm_mgr = PermutationManager()
        scores = torch.rand(4)
        _ = perm_mgr.permute_layer(model, layer_id=0, scores=scores)

        assert torch.equal(model.fc2.bias, out_bias_before)

    def test_13_14_checkpoints_preserve_state_and_behavior(self):
        """Saving and loading state dict reproduces identical model output."""
        model = SimpleMLP(in_features=2, hidden=4, out_features=2)
        perm_mgr = PermutationManager()
        scores = torch.tensor([1.0, 5.0, 2.0, 4.0])
        _ = perm_mgr.permute_layer(model, layer_id=0, scores=scores)

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)

        model2 = SimpleMLP(in_features=2, hidden=4, out_features=2)
        model2.load_state_dict(torch.load(buffer))

        x = torch.randn(5, 2)
        assert torch.allclose(model(x), model2(x), atol=1e-6)
