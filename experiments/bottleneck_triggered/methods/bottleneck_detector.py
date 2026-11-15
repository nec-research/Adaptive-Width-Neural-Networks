"""
Bottleneck Detector for Performance-Triggered Neuron Reordering.

Monitors validation loss trajectory to identify genuine performance plateaus,
preventing interventions triggered by noisy measurements or temporary fluctuations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DiagnosticEvent:
    """Record of a single bottleneck evaluation step."""
    epoch: int
    val_loss: float
    relative_improvement: Optional[float]
    patience_counter: int
    decision: str  # e.g. "TRIGGERED", "NO_PLATEAU", "IN_COOLDOWN", "BEFORE_MIN_PROGRESS", "INSUFFICIENT_HISTORY", "INVALID_VAL_LOSS"
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "val_loss": self.val_loss,
            "relative_improvement": self.relative_improvement,
            "patience_counter": self.patience_counter,
            "decision": self.decision,
            "reason": self.reason,
        }


class BottleneckDetector:
    """Detects validation loss plateaus using a relative improvement formulation.

    Formula:
        Delta_K(t) = (L_val(t - K) - L_val(t)) / (|L_val(t - K)| + eps)

    An intervention qualifies when Delta_K(t) < threshold for `patience` consecutive
    monitoring steps, provided:
      1. Training progress >= min_progress_epochs.
      2. Time since last intervention >= min_interval_between_interventions (cooldown).
      3. val_loss is finite.
      4. Plateau is reset after trigger so repeated triggers do not fire on the same plateau.
    """

    def __init__(
        self,
        plateau_window: int = 5,
        min_relative_improvement: float = 0.005,
        patience: int = 3,
        min_progress_epochs: int = 10,
        min_interval_between_interventions: int = 10,
        max_interventions: Optional[int] = None,
        abs_val_loss_threshold: Optional[float] = None,
        eps: float = 1e-8,
    ):
        if plateau_window < 1:
            raise ValueError(f"plateau_window must be >= 1, got {plateau_window}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")

        self.plateau_window = plateau_window
        self.min_relative_improvement = min_relative_improvement
        self.patience = patience
        self.min_progress_epochs = min_progress_epochs
        self.min_interval_between_interventions = min_interval_between_interventions
        self.max_interventions = max_interventions
        self.abs_val_loss_threshold = abs_val_loss_threshold
        self.eps = eps

        # State
        self.history_val_loss: List[float] = []
        self.history_epochs: List[int] = []
        self.patience_counter: int = 0
        self.last_intervention_epoch: Optional[int] = None
        self.total_interventions: int = 0
        self.diagnostic_log: List[DiagnosticEvent] = []

    def reset(self) -> None:
        """Reset internal state for a new run."""
        self.history_val_loss.clear()
        self.history_epochs.clear()
        self.patience_counter = 0
        self.last_intervention_epoch = None
        self.total_interventions = 0
        self.diagnostic_log.clear()

    def record_and_evaluate(self, epoch: int, val_loss: float) -> Tuple[bool, DiagnosticEvent]:
        """Record validation loss for the given epoch and evaluate whether an intervention triggers.

        Returns:
            (should_trigger, diagnostic_event)
        """
        # 1. Validate numerical sanity
        if not math.isfinite(val_loss):
            event = DiagnosticEvent(
                epoch=epoch,
                val_loss=val_loss,
                relative_improvement=None,
                patience_counter=self.patience_counter,
                decision="INVALID_VAL_LOSS",
                reason=f"Validation loss {val_loss} is non-finite (NaN or Inf). Intervention rejected.",
            )
            self.diagnostic_log.append(event)
            return False, event

        self.history_val_loss.append(val_loss)
        self.history_epochs.append(epoch)

        # 2. Check maximum interventions cap
        if self.max_interventions is not None and self.total_interventions >= self.max_interventions:
            event = DiagnosticEvent(
                epoch=epoch,
                val_loss=val_loss,
                relative_improvement=None,
                patience_counter=self.patience_counter,
                decision="MAX_INTERVENTIONS_REACHED",
                reason=f"Reached maximum allowed interventions ({self.max_interventions}).",
            )
            self.diagnostic_log.append(event)
            return False, event

        # 3. Check minimum training progress constraint
        if epoch < self.min_progress_epochs:
            event = DiagnosticEvent(
                epoch=epoch,
                val_loss=val_loss,
                relative_improvement=None,
                patience_counter=self.patience_counter,
                decision="BEFORE_MIN_PROGRESS",
                reason=f"Epoch {epoch} < min_progress_epochs {self.min_progress_epochs}.",
            )
            self.diagnostic_log.append(event)
            return False, event

        # 4. Check cooldown constraint (minimum interval between interventions)
        if self.last_intervention_epoch is not None:
            epochs_since_last = epoch - self.last_intervention_epoch
            if epochs_since_last < self.min_interval_between_interventions:
                event = DiagnosticEvent(
                    epoch=epoch,
                    val_loss=val_loss,
                    relative_improvement=None,
                    patience_counter=self.patience_counter,
                    decision="IN_COOLDOWN",
                    reason=f"Cooldown active: {epochs_since_last} epochs since last trigger < {self.min_interval_between_interventions}.",
                )
                self.diagnostic_log.append(event)
                return False, event

        # 5. Check history length for plateau window
        K = self.plateau_window
        if len(self.history_val_loss) <= K:
            event = DiagnosticEvent(
                epoch=epoch,
                val_loss=val_loss,
                relative_improvement=None,
                patience_counter=self.patience_counter,
                decision="INSUFFICIENT_HISTORY",
                reason=f"Need > {K} validation checkpoints, currently have {len(self.history_val_loss)}.",
            )
            self.diagnostic_log.append(event)
            return False, event

        # 6. Compute relative improvement over window K
        # Delta_K(t) = (L_val(t - K) - L_val(t)) / (|L_val(t - K)| + eps)
        loss_t_minus_k = self.history_val_loss[-1 - K]
        loss_t = self.history_val_loss[-1]
        relative_imp = (loss_t_minus_k - loss_t) / (abs(loss_t_minus_k) + self.eps)

        # 7. Check optional absolute loss threshold
        if self.abs_val_loss_threshold is not None and loss_t <= self.abs_val_loss_threshold:
            self.patience_counter = 0
            event = DiagnosticEvent(
                epoch=epoch,
                val_loss=val_loss,
                relative_improvement=relative_imp,
                patience_counter=self.patience_counter,
                decision="BELOW_ABS_THRESHOLD",
                reason=f"Validation loss {loss_t:.5f} already below target threshold {self.abs_val_loss_threshold}.",
            )
            self.diagnostic_log.append(event)
            return False, event

        # 8. Evaluate whether improvement is below threshold
        # Positive improvement means loss decreased. If relative_imp < threshold, we are plateauing or worsening.
        if relative_imp < self.min_relative_improvement:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                # Trigger intervention!
                self.total_interventions += 1
                self.last_intervention_epoch = epoch
                # Reset patience counter so continuous plateau does not repeatedly trigger
                self.patience_counter = 0
                event = DiagnosticEvent(
                    epoch=epoch,
                    val_loss=val_loss,
                    relative_improvement=relative_imp,
                    patience_counter=self.patience_counter,
                    decision="TRIGGERED",
                    reason=(
                        f"Plateau confirmed: relative improvement {relative_imp:.6f} < "
                        f"threshold {self.min_relative_improvement} for {self.patience} consecutive checks."
                    ),
                )
                self.diagnostic_log.append(event)
                return True, event
            else:
                event = DiagnosticEvent(
                    epoch=epoch,
                    val_loss=val_loss,
                    relative_improvement=relative_imp,
                    patience_counter=self.patience_counter,
                    decision="PLATEAU_ACCUMULATING",
                    reason=(
                        f"Relative improvement {relative_imp:.6f} < threshold {self.min_relative_improvement}. "
                        f"Patience count {self.patience_counter}/{self.patience}."
                    ),
                )
                self.diagnostic_log.append(event)
                return False, event
        else:
            # Active improvement observed -> reset patience counter
            self.patience_counter = 0
            event = DiagnosticEvent(
                epoch=epoch,
                val_loss=val_loss,
                relative_improvement=relative_imp,
                patience_counter=self.patience_counter,
                decision="IMPROVING",
                reason=f"Active improvement: {relative_imp:.6f} >= threshold {self.min_relative_improvement}. Patience reset.",
            )
            self.diagnostic_log.append(event)
            return False, event

    def get_diagnostic_summary(self) -> Dict[str, Any]:
        return {
            "total_evaluations": len(self.diagnostic_log),
            "total_interventions": self.total_interventions,
            "last_intervention_epoch": self.last_intervention_epoch,
            "events": [e.to_dict() for e in self.diagnostic_log],
        }
