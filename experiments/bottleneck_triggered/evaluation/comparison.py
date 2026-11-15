"""
Statistical Analysis and Historical Comparison Engine.

Performs paired statistical testing across matching seeds, computes effect sizes
(Cohen's d), confidence intervals, and formats structured comparison reports.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def compute_paired_statistics(
    x: List[float],
    y: List[float],
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """Computes paired difference, standard error, 95% CI, Cohen's d, and paired t-test p-value."""
    arr_x = np.array(x, dtype=np.float64)
    arr_y = np.array(y, dtype=np.float64)

    if len(arr_x) != len(arr_y) or len(arr_x) < 2:
        return {
            "mean_x": float(np.mean(arr_x)) if len(arr_x) > 0 else 0.0,
            "mean_y": float(np.mean(arr_y)) if len(arr_y) > 0 else 0.0,
            "mean_diff": float(np.mean(arr_x - arr_y)) if len(arr_x) == len(arr_y) and len(arr_x) > 0 else 0.0,
            "std_diff": 0.0,
            "p_value": 1.0,
            "cohens_d": 0.0,
            "ci_95": (0.0, 0.0),
        }

    diffs = arr_x - arr_y
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se_diff = std_diff / math.sqrt(n) if n > 0 else 0.0

    # Degrees of freedom and t critical for 95% CI (approx 2.776 for df=4, 2.571 for df=5)
    # Using scipy.stats if available or standard t approximations
    try:
        from scipy import stats
        t_stat, p_val = stats.ttest_rel(arr_x, arr_y)
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
    except Exception:
        # Fallback t-statistic calculation
        t_stat = mean_diff / se_diff if se_diff > 1e-12 else 0.0
        t_crit = 2.776  # df = 4 approximation
        # rough two-tailed p-value approximation
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))

    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # Cohen's d for paired samples: mean_diff / std_diff
    cohens_d = mean_diff / std_diff if std_diff > 1e-12 else 0.0

    return {
        "mean_x": float(np.mean(arr_x)),
        "std_x": float(np.std(arr_x, ddof=1)) if n > 1 else 0.0,
        "mean_y": float(np.mean(arr_y)),
        "std_y": float(np.std(arr_y, ddof=1)) if n > 1 else 0.0,
        "mean_diff": round(mean_diff, 4),
        "std_diff": round(std_diff, 4),
        "se_diff": round(se_diff, 4),
        "ci_95": (round(float(ci_lower), 4), round(float(ci_upper), 4)),
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_val), 4),
        "cohens_d": round(float(cohens_d), 4),
    }


def format_historical_comparison_table(
    dataset_name: str,
    historical_results: Dict[str, Any],
    new_results: Dict[str, List[Dict[str, Any]]],
) -> str:
    """Generates markdown table comparing historical baselines with new bottleneck-triggered methods."""
    lines = [
        f"### Performance & Capacity Comparison: {dataset_name}",
        "",
        "| Method | Evidence Source | Schedule Policy | Historical / New | Test Acc (%) | Learned Width | Prefix AUC | Training Time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    # 1. Historical methods from tensorboard_logs/paper_equivalent_benchmark.json
    hist_ds = historical_results.get("raw_results", {}).get(dataset_name, {})

    method_mapping = [
        ("Fixed Baseline", "None (Fixed Arch)", "Fixed", "Historical Reference"),
        ("AWN Paper Baseline (Identity)", "Neuron Index", "Continuous Static", "Historical Reference"),
        ("AWN + V0 (Evidence Baseline)", "Local Weight Norms", "Periodic Dynamic (freq=5)", "Historical Reference"),
        ("AWN + V1 (GRAPHyco Teacher)", "Local + Teacher", "Periodic Dynamic (freq=5)", "Historical Reference"),
    ]

    for m_name, ev_source, sched, label in method_mapping:
        runs = hist_ds.get(m_name, [])
        if runs:
            accs = [r["final_test_acc"] * 100.0 for r in runs]
            widths = [r["final_width"] for r in runs]
            times = [r["total_time_sec"] for r in runs]
            aucs = []
            for r in runs:
                tc = r.get("truncation_curves", {}).get("identity_truncation", {})
                if tc:
                    from .metrics import compute_auc_from_prefix_curve
                    aucs.append(compute_auc_from_prefix_curve(tc))
            mean_auc = f"{np.mean(aucs):.3f}" if aucs else "N/A"
            lines.append(
                f"| {m_name} | {ev_source} | {sched} | {label} | "
                f"{np.mean(accs):.2f} ± {np.std(accs):.2f} | {np.mean(widths):.1f} ± {np.std(widths):.1f} | "
                f"{mean_auc} | {np.mean(times):.2f}s |"
            )

    # 2. New Bottleneck-Triggered Methods
    new_method_specs = [
        ("V0-BT", "Weight Norm Product", "Bottleneck-Triggered", "New Experiment"),
        ("V1-BT", "Gradient-Activation Taylor", "Bottleneck-Triggered", "New Experiment"),
        ("V2-BT", "GRAPHyco Relational Graph", "Bottleneck-Triggered", "New Experiment"),
    ]

    for m_name, ev_source, sched, label in new_method_specs:
        runs = new_results.get(m_name, [])
        if runs:
            accs = [r["final_test_acc"] * 100.0 for r in runs]
            widths = [r["final_width"] for r in runs]
            times = [r["resource_ledger"]["timing_sec"]["total_wall_clock"] for r in runs]
            aucs = [r["auc_prefix_curve"] for r in runs]
            lines.append(
                f"| {m_name} | {ev_source} | {sched} | {label} | "
                f"{np.mean(accs):.2f} ± {np.std(accs):.2f} | {np.mean(widths):.1f} ± {np.std(widths):.1f} | "
                f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f} | {np.mean(times):.2f}s |"
            )

    return "\n".join(lines)
