"""
Resource Ledger for Strict Accounting and Budget Enforcement.

Tracks forward/backward passes, evidence collection overhead, graph extraction costs,
reordering time, optimizer migration time, compute proxies, and memory usage.
Enforces that experimental methods cannot exceed the resource budget of the AWN baseline.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ResourceLedger:
    """Tracks and reports resource usage during model training and intervention."""

    name: str = "default"
    # Pass and step counts
    forward_passes_train: int = 0
    forward_passes_val: int = 0
    forward_passes_evidence: int = 0
    backward_passes_train: int = 0
    backward_passes_evidence: int = 0
    optimizer_steps: int = 0
    examples_processed_train: int = 0
    examples_processed_val: int = 0

    # Timing accumulators (seconds)
    train_forward_backward_sec: float = 0.0
    val_evaluation_sec: float = 0.0
    evidence_collection_sec: float = 0.0
    graph_extraction_sec: float = 0.0
    reordering_sec: float = 0.0
    optimizer_migration_sec: float = 0.0
    total_wall_clock_sec: float = 0.0

    # Memory (MB)
    peak_memory_mb: float = 0.0

    # Compute proxy (approximate FLOPs: 2 * active_params * batch_size per forward pass, 4 * params * batch_size per backward)
    compute_proxy_flops: float = 0.0

    # Budget limits (optional enforcement)
    max_optimizer_steps: Optional[int] = None
    max_compute_proxy_flops: Optional[float] = None
    max_examples_processed: Optional[int] = None

    _start_time: float = field(default_factory=time.perf_counter, repr=False)

    def __post_init__(self):
        if not tracemalloc.is_tracing():
            try:
                tracemalloc.start()
            except Exception:
                pass

    def record_forward_train(self, batch_size: int, param_count: int, duration_sec: float = 0.0) -> None:
        self.forward_passes_train += 1
        self.examples_processed_train += batch_size
        self.compute_proxy_flops += 2.0 * param_count * batch_size
        self.train_forward_backward_sec += duration_sec
        self._update_peak_memory()

    def record_backward_train(self, batch_size: int, param_count: int, duration_sec: float = 0.0) -> None:
        self.backward_passes_train += 1
        self.compute_proxy_flops += 4.0 * param_count * batch_size
        self.train_forward_backward_sec += duration_sec
        self._update_peak_memory()

    def record_optimizer_step(self) -> None:
        self.optimizer_steps += 1

    def record_val_eval(self, batch_size: int, duration_sec: float) -> None:
        self.forward_passes_val += 1
        self.examples_processed_val += batch_size
        self.val_evaluation_sec += duration_sec
        self._update_peak_memory()

    def record_evidence_pass(
        self,
        batch_size: int,
        param_count: int,
        is_backward: bool = False,
        duration_sec: float = 0.0,
    ) -> None:
        if is_backward:
            self.backward_passes_evidence += 1
            self.compute_proxy_flops += 4.0 * param_count * batch_size
        else:
            self.forward_passes_evidence += 1
            self.compute_proxy_flops += 2.0 * param_count * batch_size
        self.evidence_collection_sec += duration_sec
        self._update_peak_memory()

    def record_graph_extraction(self, duration_sec: float, estimated_flops: float = 0.0) -> None:
        self.graph_extraction_sec += duration_sec
        self.compute_proxy_flops += estimated_flops
        self._update_peak_memory()

    def record_reordering(self, duration_sec: float, migration_duration_sec: float = 0.0) -> None:
        self.reordering_sec += duration_sec
        self.optimizer_migration_sec += migration_duration_sec
        self._update_peak_memory()

    def _update_peak_memory(self) -> None:
        try:
            current, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / (1024.0 * 1024.0)
            if peak_mb > self.peak_memory_mb:
                self.peak_memory_mb = peak_mb
        except Exception:
            pass

    def stop_and_finalize(self) -> None:
        self.total_wall_clock_sec = time.perf_counter() - self._start_time
        self._update_peak_memory()

    def is_budget_exhausted(self) -> bool:
        """Returns True if any configured budget limit has been reached or exceeded."""
        if self.max_optimizer_steps is not None and self.optimizer_steps >= self.max_optimizer_steps:
            return True
        if self.max_compute_proxy_flops is not None and self.compute_proxy_flops >= self.max_compute_proxy_flops:
            return True
        if self.max_examples_processed is not None and self.examples_processed_train >= self.max_examples_processed:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "forward_passes_train": self.forward_passes_train,
            "forward_passes_val": self.forward_passes_val,
            "forward_passes_evidence": self.forward_passes_evidence,
            "backward_passes_train": self.backward_passes_train,
            "backward_passes_evidence": self.backward_passes_evidence,
            "total_forward_passes": (
                self.forward_passes_train + self.forward_passes_val + self.forward_passes_evidence
            ),
            "total_backward_passes": self.backward_passes_train + self.backward_passes_evidence,
            "optimizer_steps": self.optimizer_steps,
            "examples_processed_train": self.examples_processed_train,
            "examples_processed_val": self.examples_processed_val,
            "compute_proxy_flops": self.compute_proxy_flops,
            "timing_sec": {
                "train_forward_backward": round(self.train_forward_backward_sec, 4),
                "val_evaluation": round(self.val_evaluation_sec, 4),
                "evidence_collection": round(self.evidence_collection_sec, 4),
                "graph_extraction": round(self.graph_extraction_sec, 4),
                "reordering": round(self.reordering_sec, 4),
                "optimizer_migration": round(self.optimizer_migration_sec, 4),
                "total_wall_clock": round(self.total_wall_clock_sec, 4),
            },
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }
