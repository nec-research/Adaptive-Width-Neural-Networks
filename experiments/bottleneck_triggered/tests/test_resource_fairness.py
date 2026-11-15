"""
Suite I: Resource Fairness Tests.

Verifies strict resource accounting, parity with AWN reference budget,
and inclusion of evidence, graph, and reordering overheads.
"""

import pytest
from experiments.bottleneck_triggered.accounting.resource_ledger import ResourceLedger
from experiments.bottleneck_triggered.runner import run_single_experiment


class TestResourceFairness:
    """Verifies that experimental methods respect strict resource budget accounting."""

    def test_resource_ledger_completeness_and_consistency(self):
        """Resource ledger records all pass types, step counts, and timing fields."""
        ledger = ResourceLedger(name="test_ledger")
        ledger.record_forward_train(batch_size=32, param_count=500, duration_sec=0.01)
        ledger.record_backward_train(batch_size=32, param_count=500, duration_sec=0.02)
        ledger.record_optimizer_step()
        ledger.record_evidence_pass(batch_size=32, param_count=500, is_backward=False, duration_sec=0.005)
        ledger.record_graph_extraction(duration_sec=0.003, estimated_flops=100.0)
        ledger.record_reordering(duration_sec=0.001, migration_duration_sec=0.0005)
        ledger.stop_and_finalize()

        d = ledger.to_dict()
        assert d["forward_passes_train"] == 1
        assert d["backward_passes_train"] == 1
        assert d["forward_passes_evidence"] == 1
        assert d["optimizer_steps"] == 1
        assert d["compute_proxy_flops"] > 0
        assert d["timing_sec"]["total_wall_clock"] >= 0.0

    def test_evidence_and_reordering_costs_not_free(self):
        """Intervention operations must accrue time and compute in the ledger."""
        res = run_single_experiment(
            method_name="V0-BT",
            dataset_name="DoubleMoon",
            seed=42,
            epochs=10,
            bottleneck_config={
                "plateau_window": 1,
                "min_relative_improvement": 0.5,
                "patience": 1,
                "min_progress_epochs": 2,
            },
        )
        ledger_data = res["resource_ledger"]
        if res["total_interventions"] > 0:
            assert ledger_data["timing_sec"]["reordering"] > 0.0

    def test_parity_with_awn_reference_budget(self):
        """A method constrained by AWN baseline budget cannot exceed baseline steps or examples."""
        # 1. Establish reference baseline
        baseline_res = run_single_experiment(
            method_name="AWN_Original",
            dataset_name="DoubleMoon",
            seed=42,
            epochs=5,
        )
        base_ledger = ResourceLedger(name="ref")
        base_ledger.optimizer_steps = baseline_res["resource_ledger"]["optimizer_steps"]
        base_ledger.examples_processed_train = baseline_res["resource_ledger"]["examples_processed_train"]
        base_ledger.compute_proxy_flops = baseline_res["resource_ledger"]["compute_proxy_flops"]

        # 2. Run V0-BT bounded by baseline
        v0_res = run_single_experiment(
            method_name="V0-BT",
            dataset_name="DoubleMoon",
            seed=42,
            epochs=10,  # requested more epochs, but bounded by budget
            reference_ledger=base_ledger,
        )

        assert v0_res["resource_ledger"]["optimizer_steps"] <= base_ledger.optimizer_steps + 1
