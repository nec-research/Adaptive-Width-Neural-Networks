# Bottleneck-Triggered AWN: correctness patch audit

Base branch: `V(1-3)BT`.

This patch intentionally leaves the archived `reports/research_experiments.json` unchanged. Its metrics were produced **before these fixes** and are **not** evidence of post-patch performance.

## Changes

1. **Baseline AWN evaluation** (`model.py`): `forward(skip_update=True)` now derives widths from *existing physical layers*, not an undefined local variable or a potentially stale width quantile. It avoids layer resizing during evaluation.
2. **V1 gradient evidence**: scores now use the actual hidden activation consumed by the next layer in AWN's gated forward, captured using a temporary forward pre-hook. Classification cross-entropy provides the task gradient; regularization terms do not directly differentiate with respect to the captured hidden activation. The method restores the previous model mode and does not write parameter gradients.
3. **Stable neuron ranking**: uses stable descending `torch.argsort`, with original-index order for exact float32 ties.
4. **Permutation telemetry**: retains the no-gate/full-width functional symmetry check and additionally records changes under AWN's *actual* position-dependent gates as `awn_output_diff` and `awn_loss_diff`. Unlike the no-gate check, these two quantities are **not** expected to be zero.
5. **Runner fairness**: V1 uses an independent non-shuffled evidence loader, so collecting evidence doesn't consume the training loader's shuffle RNG. Matches baseline training-step and example limits; records extra evidence FLOPs instead of stopping training prematurely at a 5%-margin proxy FLOP ceiling.
6. **Tests**: align preservation assertions with actual baseline AWN, test skip-update evaluation, stable ties, mode restoration, and distinguish gated-vs-no-gate intervention effects.

## Evaluation caveats

- **No experimental rerun is claimed.** Comparing the saved historical numbers against newly modified code would be invalid until both branches are rerun in the same pinned environment.
- The old `reports/research_experiments.json` includes nonstandard JSON `NaN` entries; it was not rewritten to preserve historical evidence.
- Equal training steps are not equal total compute; evidence and graph extraction are additional costs. Both axes should be reported.
- The plateau detector observes validation-loss stagnation, not a directly demonstrated structural bottleneck.
- V2's unnormalized graph score components need separate normalization/ablation.
- The intervention remains a heuristic physical ordering, not a differentiable learned permutation.

## Suggested verification

```bash
uv sync --locked
uv run --with pytest pytest -q experiments/bottleneck_triggered/tests
uv run python -m experiments.bottleneck_triggered.run_research_experiments
```

Save new-run metrics to a **new, dated file** and compare paired seeds, test accuracy, active width, prefix AUC, train steps, total compute proxy, and wall-clock cost. Do not overwrite the pre-patch report.

## Open research controls

- Random permutations at identical detector epochs
- Periodic scheduling with each evidence method
- Actual-gate intervention diagnostics before/after every reordering
- V2 component normalization and graph-feature ablations
