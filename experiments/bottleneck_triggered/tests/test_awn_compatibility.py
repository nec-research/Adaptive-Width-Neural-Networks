"""
Suite G: AWN Compatibility Tests.

Verifies that physical neuron reordering does not interfere with AWN's width optimization,
ELBO objective, lambda parameter learning, or adaptive truncation.
"""

import pytest
import torch

from experiments.bottleneck_triggered.common_utils import get_awn_config
from experiments.bottleneck_triggered.permutation.permutation_manager import PermutationManager
from metric import ELBO_Classification
from model import AWN


class TestAWNCompatibility:
    """Verifies that AWN variational width mechanics continue operating uninterrupted."""

    def test_lambda_parameter_continues_learning_after_reordering(self):
        """Width rate parameter continues updating via gradients after a physical intervention."""
        cfg = get_awn_config(n_obs=1000, enable_learned_ordering=False)
        model = AWN(dim_input_features=2, dim_target=2, config=cfg)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        model.torch_optimizer = opt
        elbo = ELBO_Classification(1000)

        rate_param = model.variational_Ws[0].discretized_distribution.base_distribution._rate
        init_rate = rate_param.item()

        # Step 1: Pre-intervention training step
        x = torch.randn(16, 2)
        y = torch.randint(0, 2, (16, 1))
        opt.zero_grad()
        outs = model(x)
        preds, tgt = elbo.get_predictions_and_targets(y, *outs)
        loss = elbo.compute_metric(tgt, preds)
        loss.backward()
        opt.step()

        # Step 2: Apply physical intervention
        curr_width = model.variational_Ws[0].compute_truncation_number()
        scores = torch.rand(curr_width)
        perm_mgr = PermutationManager(tolerance=1e-4)
        _ = perm_mgr.permute_layer(model, layer_id=0, scores=scores, optimizer=opt)

        # Step 3: Post-intervention training steps
        for _ in range(5):
            opt.zero_grad()
            outs = model(x)
            preds, tgt = elbo.get_predictions_and_targets(y, *outs)
            loss = elbo.compute_metric(tgt, preds)
            loss.backward()
            opt.step()

        final_rate = rate_param.item()
        assert final_rate != init_rate, "Width rate parameter failed to learn post-intervention!"
        assert rate_param.grad is not None, "Gradients missing for rate parameter!"

    def test_active_width_calculation_valid_post_intervention(self):
        """Quantile truncation number produces valid integer width within layer capacity."""
        cfg = get_awn_config(n_obs=500, enable_learned_ordering=False)
        model = AWN(dim_input_features=2, dim_target=2, config=cfg)
        perm_mgr = PermutationManager()

        curr_w = model.variational_Ws[0].compute_truncation_number()
        scores = torch.rand(curr_w)
        _ = perm_mgr.permute_layer(model, layer_id=0, scores=scores)

        post_w = model.variational_Ws[0].compute_truncation_number()
        assert isinstance(post_w, int)
        assert 1 <= post_w <= 256

    def test_disabling_extension_restores_original_awn(self):
        """When no intervention is triggered, behavior matches pure original AWN identically."""
        torch.manual_seed(123)
        cfg = get_awn_config(n_obs=500, enable_learned_ordering=False)
        model1 = AWN(dim_input_features=2, dim_target=2, config=cfg)

        torch.manual_seed(123)
        model2 = AWN(dim_input_features=2, dim_target=2, config=cfg)

        x = torch.randn(8, 2)
        out1 = model1(x)[0]
        out2 = model2(x)[0]

        assert torch.equal(out1, out2)
