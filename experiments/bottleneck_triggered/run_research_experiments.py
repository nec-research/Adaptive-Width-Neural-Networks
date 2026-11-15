"""
Gate 6 Research Experiment Execution Script.

Runs the controlled experimental suite for:
  - Reference Baseline: Original AWN
  - V0-BT: Bottleneck-Triggered Weight-Based Neuron Reordering
  - V1-BT: Bottleneck-Triggered Gradient-Based Neuron Reordering
  - V2-BT: Bottleneck-Triggered Graph-Based Neuron Reordering

Across 3 benchmark datasets: DoubleMoon, Spiral, SpiralHard.
Across 5 deterministic seeds: [42, 123, 456, 789, 1000].
Enforces strict resource budgets anchored to the measured original AWN reference.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

from experiments.bottleneck_triggered.accounting.resource_ledger import ResourceLedger
from experiments.bottleneck_triggered.evaluation.comparison import (
    compute_paired_statistics,
    format_historical_comparison_table,
)
from experiments.bottleneck_triggered.runner import run_single_experiment


def main():
    print("=" * 80)
    print("GATE 6: BOTTLENECK-TRIGGERED EVIDENCE-BASED NEURON REORDERING")
    print("STRICTLY CONTROLLED RESEARCH BENCHMARK")
    print("=" * 80)

    datasets = ["DoubleMoon", "Spiral", "SpiralHard"]
    seeds = [42, 123, 456, 789, 1000]
    bt_methods = ["V0-BT", "V1-BT", "V2-BT"]

    dataset_configs = {
        "DoubleMoon": {"epochs": 30, "lr": 0.01},
        "Spiral": {"epochs": 50, "lr": 0.01},
        "SpiralHard": {"epochs": 50, "lr": 0.01},
    }

    # Shared Bottleneck Detection Policy across all methods and datasets
    bottleneck_config = {
        "plateau_window": 3,
        "min_relative_improvement": 0.005,
        "patience": 2,
        "min_progress_epochs": 6,
        "min_interval_between_interventions": 5,
    }

    evidence_configs = {
        "V0-BT": {},
        "V1-BT": {"num_batches": 5},
        "V2-BT": {
            "edge_weight_threshold": 0.05,
            "w_struct": 0.4,
            "w_in": 0.3,
            "w_out": 0.3,
        },
    }

    output_dir = "experiments/bottleneck_triggered/reports"
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "research_experiments.json")

    all_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        d: {m: [] for m in ["AWN_Original"] + bt_methods} for d in datasets
    }

    total_runs = len(datasets) * len(seeds) * (1 + len(bt_methods))
    completed_runs = 0
    t0_all = time.perf_counter()

    for dname in datasets:
        dcfg = dataset_configs[dname]
        epochs = dcfg["epochs"]
        lr = dcfg["lr"]

        print(f"\n{'='*70}")
        print(f"BENCHMARK DATASET: {dname} (Epochs={epochs}, LR={lr})")
        print(f"{'='*70}")

        for seed in seeds:
            # -------------------------------------------------------------
            # Step 1: Establish Reference Budget with Original AWN
            # -------------------------------------------------------------
            completed_runs += 1
            print(f"[{completed_runs:02d}/{total_runs}] {dname} | Seed {seed} | AWN_Original (Ref Budget)...", end=" ", flush=True)
            base_res = run_single_experiment(
                method_name="AWN_Original",
                dataset_name=dname,
                seed=seed,
                epochs=epochs,
                lr=lr,
                bottleneck_config=bottleneck_config,
            )
            all_results[dname]["AWN_Original"].append(base_res)
            print(f"DONE in {base_res['resource_ledger']['timing_sec']['total_wall_clock']:.1f}s | Acc={base_res['final_test_acc']*100:.2f}% | Width={base_res['final_width']}")

            # Extract reference budget
            ref_ledger = ResourceLedger(name=f"ref_{dname}_seed{seed}")
            ref_ledger.optimizer_steps = base_res["resource_ledger"]["optimizer_steps"]
            ref_ledger.examples_processed_train = base_res["resource_ledger"]["examples_processed_train"]
            ref_ledger.compute_proxy_flops = base_res["resource_ledger"]["compute_proxy_flops"]

            # -------------------------------------------------------------
            # Step 2: Run V0-BT, V1-BT, and V2-BT under Bounded Reference Budget
            # -------------------------------------------------------------
            for method in bt_methods:
                completed_runs += 1
                print(f"[{completed_runs:02d}/{total_runs}] {dname} | Seed {seed} | {method}...", end=" ", flush=True)
                m_res = run_single_experiment(
                    method_name=method,
                    dataset_name=dname,
                    seed=seed,
                    epochs=epochs,
                    lr=lr,
                    bottleneck_config=bottleneck_config,
                    evidence_config=evidence_configs[method],
                    reference_ledger=ref_ledger,
                )
                all_results[dname][method].append(m_res)
                print(
                    f"DONE in {m_res['resource_ledger']['timing_sec']['total_wall_clock']:.1f}s | "
                    f"Acc={m_res['final_test_acc']*100:.2f}% | Width={m_res['final_width']} | "
                    f"AUC={m_res['auc_prefix_curve']:.4f} | Int={m_res['total_interventions']}"
                )

    total_time_all = time.perf_counter() - t0_all
    print(f"\nAll experiments finished in {total_time_all/60.0:.2f} minutes.")

    # -------------------------------------------------------------------------
    # Statistical Synthesis
    # -------------------------------------------------------------------------
    stats_summary: Dict[str, Any] = {}

    for dname in datasets:
        stats_summary[dname] = {}
        base_runs = all_results[dname]["AWN_Original"]
        base_accs = [r["final_test_acc"] * 100.0 for r in base_runs]
        base_widths = [r["final_width"] for r in base_runs]
        base_aucs = [r["auc_prefix_curve"] for r in base_runs]

        for method in bt_methods:
            m_runs = all_results[dname][method]
            m_accs = [r["final_test_acc"] * 100.0 for r in m_runs]
            m_widths = [r["final_width"] for r in m_runs]
            m_aucs = [r["auc_prefix_curve"] for r in m_runs]

            stats_acc = compute_paired_statistics(m_accs, base_accs)
            stats_width = compute_paired_statistics(m_widths, base_widths)
            stats_auc = compute_paired_statistics(m_aucs, base_aucs)

            stats_summary[dname][method] = {
                "accuracy": stats_acc,
                "width": stats_width,
                "auc": stats_auc,
                "interventions_mean": float(np.mean([r["total_interventions"] for r in m_runs])),
            }

    # Save comprehensive research experiment payload
    final_payload = {
        "metadata": {
            "protocol": "Bottleneck-Triggered Evidence-Based Neuron Reordering Specification",
            "datasets": datasets,
            "seeds": seeds,
            "methods": ["AWN_Original"] + bt_methods,
            "bottleneck_config": bottleneck_config,
            "evidence_configs": evidence_configs,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_benchmark_time_sec": round(total_time_all, 2),
        },
        "raw_results": all_results,
        "statistical_summary": stats_summary,
    }

    with open(results_path, "w") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\nResearch experiment results saved to: {results_path}")

    # Print summary tables
    print("\n" + "=" * 90)
    print("SUMMARY RESULTS: PREDICTIVE PERFORMANCE & CAPACITY (Mean ± Std over 5 Seeds)")
    print("=" * 90)
    print(f"| Dataset | Method | Test Acc (%) | Learned Width | Prefix AUC | Int / Run | Wall Time (s) |")
    print(f"|---|---|---|---|---|---|---|")

    for dname in datasets:
        for m in ["AWN_Original"] + bt_methods:
            runs = all_results[dname][m]
            accs = [r["final_test_acc"] * 100.0 for r in runs]
            widths = [r["final_width"] for r in runs]
            aucs = [r["auc_prefix_curve"] for r in runs]
            ints = [r["total_interventions"] for r in runs]
            times = [r["resource_ledger"]["timing_sec"]["total_wall_clock"] for r in runs]

            print(
                f"| {dname} | {m} | {np.mean(accs):.2f} ± {np.std(accs):.2f} | "
                f"{np.mean(widths):.1f} ± {np.std(widths):.1f} | {np.mean(aucs):.3f} ± {np.std(aucs):.3f} | "
                f"{np.mean(ints):.1f} | {np.mean(times):.2f}s |"
            )

    print("\n" + "=" * 90)
    print("HISTORICAL COMPARISON TABLE (Contextual Alignment)")
    print("=" * 90)

    # Load historical results
    hist_path = "tensorboard_logs/paper_equivalent_benchmark.json"
    if os.path.exists(hist_path):
        with open(hist_path, "r") as f:
            hist_data = json.load(f)
        for dname in datasets:
            md_table = format_historical_comparison_table(dname, hist_data, all_results[dname])
            print(f"\n{md_table}")


if __name__ == "__main__":
    main()
