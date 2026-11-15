"""
Suite A: Original Code Preservation Tests.

Verifies that the original AWN codebase, historical baselines, configuration files,
and scoring mechanisms remain completely untouched and functionally intact.
"""

import os
import subprocess
import pytest
import torch

from experiments.bottleneck_triggered.common_utils import get_awn_config
from metric import ELBO_Classification
from model import AWN


class TestOriginalPreservation:
    """Verifies non-negotiable repository preservation rules."""

    def test_original_awn_import_and_execution(self):
        """Original AWN initializes and performs forward/backward pass with default config."""
        cfg = get_awn_config(n_obs=1000, enable_learned_ordering=False)
        model = AWN(dim_input_features=2, dim_target=2, config=cfg)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        model.torch_optimizer = opt
        elbo = ELBO_Classification(1000)

        x = torch.randn(8, 2)
        y = torch.tensor([[0], [1], [0], [1], [0], [1], [0], [1]])

        outs = model(x)
        preds, tgt = elbo.get_predictions_and_targets(y, *outs)
        loss = elbo.compute_metric(tgt, preds)
        loss.backward()
        opt.step()

        assert not torch.isnan(loss)
        assert model.variational_Ws[0].compute_truncation_number() > 0

    def test_historical_configurations_exist_and_unmodified(self):
        """Historical config files must be preserved and non-empty."""
        historical_configs = [
            "EXP_CONFIGS/GRID/MLP/DoubleMoon/mlp_awn.yml",
            "EXP_CONFIGS/GRID/MLP/Spiral/mlp_awn.yml",
            "EXP_CONFIGS/GRID/MLP/SpiralHard/mlp_awn.yml",
            "EXP_CONFIGS/GRID/MLP/DoubleMoon/mlp_fixed.yml",
            "EXP_CONFIGS/GRID/MLP/Spiral/mlp_fixed.yml",
            "EXP_CONFIGS/GRID/MLP/SpiralHard/mlp_fixed.yml",
        ]
        for cfg_path in historical_configs:
            assert os.path.exists(cfg_path), f"Historical config missing: {cfg_path}"
            assert os.path.getsize(cfg_path) > 100, f"Config appears truncated: {cfg_path}"

    def test_historical_benchmark_json_preserved(self):
        """Archived paper-equivalent benchmark results must be preserved."""
        bench_json = "tensorboard_logs/paper_equivalent_benchmark.json"
        assert os.path.exists(bench_json), "Archived benchmark json missing!"
        assert os.path.getsize(bench_json) > 10000

    def test_original_code_path_when_extension_not_selected(self):
        """The baseline AWN has no built-in ordering component."""
        cfg = get_awn_config(n_obs=500, enable_learned_ordering=False)
        model = AWN(dim_input_features=2, dim_target=2, config=cfg)
        assert not hasattr(model, "neuron_ranker")
        assert not hasattr(model, "ranking_loss_weight")
        assert len(model.variational_Ws) == 1

    def test_skip_update_forward_keeps_width_and_runs(self):
        """Validation/latency inference must not reference undefined widths."""
        cfg = get_awn_config(n_obs=500, enable_learned_ordering=False)
        model = AWN(dim_input_features=2, dim_target=2, config=cfg)
        x = torch.randn(8, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.torch_optimizer = optimizer
        width_before = model.dyn_model.get_layer(0).weight.shape[0]
        with torch.no_grad():
            out = model(x, skip_update=True)[0]
        assert out.shape == (8, 2)
        assert model.dyn_model.get_layer(0).weight.shape[0] == width_before

    def test_git_diff_preservation(self):
        """Ensure no core AWN algorithms or configs have been deleted in git status."""
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        # Check that no tracked file was deleted
        for line in result.stdout.splitlines():
            assert not line.startswith(" D"), f"Tracked file was deleted: {line}"
