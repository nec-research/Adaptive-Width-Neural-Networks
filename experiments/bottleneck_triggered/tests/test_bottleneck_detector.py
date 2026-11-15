"""
Suite B: Bottleneck Detector Unit Tests.

Covers all 12 mandatory requirements from Section 8.B.
"""

import math
import pytest
from experiments.bottleneck_triggered.methods.bottleneck_detector import BottleneckDetector


class TestBottleneckDetector:
    """Rigorous unit tests for bottleneck detection logic and safeguards."""

    def test_1_continuous_improvement_no_trigger(self):
        """Monotonically decreasing loss with substantial improvement never triggers."""
        detector = BottleneckDetector(
            plateau_window=3,
            min_relative_improvement=0.01,
            patience=2,
            min_progress_epochs=5,
        )
        losses = [1.0, 0.85, 0.70, 0.55, 0.40, 0.30, 0.20, 0.10]
        for ep, loss in enumerate(losses, start=1):
            triggered, event = detector.record_and_evaluate(ep, loss)
            assert not triggered, f"Unexpected trigger at epoch {ep} with improving loss {loss}"

    def test_2_single_noisy_validation_no_trigger(self):
        """A single noisy spike or flat point must not trigger if patience > 1."""
        detector = BottleneckDetector(
            plateau_window=3,
            min_relative_improvement=0.01,
            patience=3,
            min_progress_epochs=4,
        )
        # Improving trajectory with one flat point
        losses = [1.0, 0.8, 0.6, 0.4, 0.4, 0.2, 0.1]
        for ep, loss in enumerate(losses, start=1):
            triggered, event = detector.record_and_evaluate(ep, loss)
            assert not triggered, f"Noisy measurement at epoch {ep} triggered prematurely"

    def test_3_sustained_plateau_triggers(self):
        """A sustained flat plateau exceeding patience window must trigger."""
        detector = BottleneckDetector(
            plateau_window=3,
            min_relative_improvement=0.01,
            patience=2,
            min_progress_epochs=4,
        )
        # Initial improvement then plateau of length >= K + P
        losses = [1.0, 0.8, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        triggered_epochs = []
        for ep, loss in enumerate(losses, start=1):
            triggered, event = detector.record_and_evaluate(ep, loss)
            if triggered:
                triggered_epochs.append(ep)

        assert len(triggered_epochs) >= 1
        assert detector.diagnostic_log[-2].decision == "TRIGGERED" or detector.diagnostic_log[-1].decision == "TRIGGERED" or len(triggered_epochs) > 0

    def test_4_worsening_loss_not_classified_as_improvement(self):
        """Worsening loss (increasing) has negative relative improvement and counts as plateau/stagnation."""
        detector = BottleneckDetector(
            plateau_window=2,
            min_relative_improvement=0.01,
            patience=2,
            min_progress_epochs=3,
        )
        losses = [0.5, 0.5, 0.6, 0.7, 0.8]
        triggered_any = False
        for ep, loss in enumerate(losses, start=1):
            triggered, event = detector.record_and_evaluate(ep, loss)
            if event.relative_improvement is not None:
                assert event.relative_improvement < 0.0
            if triggered:
                triggered_any = True
        assert triggered_any

    def test_5_nan_and_infinite_values_handled_safely(self):
        """NaN and Inf in validation loss must be rejected safely without crashing or triggering."""
        detector = BottleneckDetector()
        for ep, bad_val in enumerate([float("nan"), float("inf"), float("-inf")], start=1):
            triggered, event = detector.record_and_evaluate(ep, bad_val)
            assert not triggered
            assert event.decision == "INVALID_VAL_LOSS"

    def test_6_insufficient_history_no_trigger(self):
        """Intervention must not trigger if history length <= plateau_window."""
        detector = BottleneckDetector(plateau_window=5, min_progress_epochs=1)
        losses = [0.5, 0.5, 0.5, 0.5, 0.5]  # exactly 5
        for ep, loss in enumerate(losses, start=1):
            triggered, event = detector.record_and_evaluate(ep, loss)
            assert not triggered
            assert event.decision == "INSUFFICIENT_HISTORY"

    def test_7_min_progress_constraint_respected(self):
        """Plateau before min_progress_epochs must not trigger."""
        detector = BottleneckDetector(
            plateau_window=2,
            patience=1,
            min_progress_epochs=10,
        )
        losses = [0.5] * 8
        for ep, loss in enumerate(losses, start=1):
            triggered, event = detector.record_and_evaluate(ep, loss)
            assert not triggered
            assert event.decision == "BEFORE_MIN_PROGRESS"

    def test_8_patience_enforced_strictly(self):
        """Trigger requires exactly `patience` consecutive below-threshold checks."""
        detector = BottleneckDetector(
            plateau_window=2,
            min_relative_improvement=0.01,
            patience=3,
            min_progress_epochs=2,
        )
        losses = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8]
        patience_counts = []
        triggers = []
        for ep, loss in enumerate(losses, start=1):
            trig, ev = detector.record_and_evaluate(ep, loss)
            triggers.append(trig)
            patience_counts.append(ev.patience_counter)

        # Triggers only when count reaches patience (3) at epoch 7
        assert triggers.count(True) == 1
        assert triggers[-1] is True

    def test_9_cooldown_interval_respected(self):
        """After a trigger, no new intervention is allowed during cooldown interval."""
        detector = BottleneckDetector(
            plateau_window=2,
            min_relative_improvement=0.01,
            patience=1,
            min_progress_epochs=3,
            min_interval_between_interventions=5,
        )
        # Epoch 5 triggers. Epochs 6, 7, 8 must be blocked by cooldown of 5.
        losses = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
        triggered_epochs = []
        for ep, loss in enumerate(losses, start=1):
            trig, ev = detector.record_and_evaluate(ep, loss)
            if trig:
                triggered_epochs.append(ep)

        assert triggered_epochs == [5]
        assert detector.diagnostic_log[-1].decision == "IN_COOLDOWN"

    def test_10_scheduler_transition_fluctuation_resilience(self):
        """Temporary bump from LR step followed by steep drop does not trigger."""
        detector = BottleneckDetector(
            plateau_window=2,
            min_relative_improvement=0.01,
            patience=2,
            min_progress_epochs=3,
        )
        # Drop, bump at lr transition, then sharp drop
        losses = [1.0, 0.8, 0.6, 0.65, 0.4, 0.2]
        for ep, loss in enumerate(losses, start=1):
            trig, ev = detector.record_and_evaluate(ep, loss)
            assert not trig

    def test_11_identical_inputs_produce_identical_decisions(self):
        """Deterministic behavior: identical loss history yields identical trigger decisions."""
        d1 = BottleneckDetector(plateau_window=3, patience=2, min_progress_epochs=4)
        d2 = BottleneckDetector(plateau_window=3, patience=2, min_progress_epochs=4)

        history = [1.2, 1.0, 0.85, 0.70, 0.701, 0.7005, 0.699, 0.699]
        for ep, val in enumerate(history, start=1):
            t1, ev1 = d1.record_and_evaluate(ep, val)
            t2, ev2 = d2.record_and_evaluate(ep, val)
            assert t1 == t2
            assert ev1.decision == ev2.decision
            assert ev1.relative_improvement == ev2.relative_improvement

    def test_12_trigger_independent_of_evidence_method(self):
        """Detector state does not receive or depend on evidence scores."""
        detector = BottleneckDetector()
        assert not hasattr(detector, "evidence_scores")
        assert not hasattr(detector, "ranker")
