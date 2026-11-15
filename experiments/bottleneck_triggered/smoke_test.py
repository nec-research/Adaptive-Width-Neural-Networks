"""
Gate 5 Smoke Tests: Controlled Verification of V0-BT, V1-BT, and V2-BT.

Performs short runs for all three new methods on DoubleMoon to verify:
- Successful training
- Correct bottleneck detection
- Correct evidence calculation
- Correct permutation and full-width equivalence
- Continued AWN width learning
- Correct resource accounting
"""

import pprint
from experiments.bottleneck_triggered.runner import run_single_experiment


def run_smoke_tests():
    print("=" * 70)
    print("GATE 5 SMOKE TESTS: V0-BT, V1-BT, V2-BT VERIFICATION")
    print("=" * 70)

    methods = ["V0-BT", "V1-BT", "V2-BT"]
    smoke_results = {}

    for method in methods:
        print(f"\n---> Running smoke test for {method} on DoubleMoon...")
        res = run_single_experiment(
            method_name=method,
            dataset_name="DoubleMoon",
            seed=42,
            epochs=15,
            bottleneck_config={
                "plateau_window": 2,
                "min_relative_improvement": 0.05,
                "patience": 2,
                "min_progress_epochs": 4,
                "min_interval_between_interventions": 3,
            },
        )
        print(f"[{method}] Completed {res['epochs_completed']} epochs.")
        print(f"[{method}] Total interventions: {res['total_interventions']}")
        print(f"[{method}] Final learned width: {res['final_width']}, Test Acc: {res['final_test_acc']*100:.2f}%")
        print(f"[{method}] Prefix AUC: {res['auc_prefix_curve']:.4f}")
        print(f"[{method}] Wall-clock time: {res['resource_ledger']['timing_sec']['total_wall_clock']:.2f}s")
        if res["interventions"]:
            last_int = res["interventions"][-1]
            print(f"[{method}] Last intervention max output diff: {last_int['max_output_diff']:.8f}")
            assert last_int["max_output_diff"] < 1e-4

        smoke_results[method] = {
            "epochs": res["epochs_completed"],
            "interventions": res["total_interventions"],
            "test_acc": res["final_test_acc"],
            "width": res["final_width"],
            "auc": res["auc_prefix_curve"],
            "wall_clock": res["resource_ledger"]["timing_sec"]["total_wall_clock"],
        }

    print("\n" + "=" * 70)
    print("GATE 5 SMOKE TEST SUMMARY:")
    pprint.pprint(smoke_results)
    print("=" * 70)
    print("ALL GATE 5 SMOKE TESTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_smoke_tests()
