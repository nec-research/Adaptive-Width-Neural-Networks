#!/usr/bin/env python3
"""Benchmark exponential probability-vector implementations.

Run from the repository root, for example:

    python benchmarks/benchmark_distribution.py --device cpu
    python benchmarks/benchmark_distribution.py --device mps
    python benchmarks/benchmark_distribution.py --device cuda
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from distribution import DiscretizedDistribution  # noqa: E402


DEFAULT_CASES = (
    (0.02, 128),
    (0.004, 512),
    (0.0005, 5_000),
    (0.0001, 23_000),
)


def parse_case(value: str) -> tuple[float, int]:
    try:
        rate_text, width_text = value.split(":", maxsplit=1)
        rate = float(rate_text)
        width = int(width_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Expected RATE:WIDTH, received {value!r}."
        ) from error

    if rate <= 0.0 or width <= 0:
        raise argparse.ArgumentTypeError("RATE and WIDTH must both be positive.")
    return rate, width


def choose_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        device = torch.device(requested_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


def make_distribution(rate: float, device: torch.device) -> DiscretizedDistribution:
    distribution = DiscretizedDistribution(
        base_distribution={
            "class_name": "distribution.Exponential",
            "args": {"rate": rate, "boundary": 1e-6},
        }
    )
    # The repository's overridden to() methods mutate in place and return None.
    distribution.to(device)
    return distribution


def old_probability_vector(
    distribution: DiscretizedDistribution,
    x: torch.Tensor,
) -> torch.Tensor:
    """Former CDF-difference -> log -> exp -> normalize implementation."""
    log_probabilities = distribution.log_prob(x).squeeze(1)
    probabilities = log_probabilities.exp()
    return probabilities / probabilities.sum()


def new_probability_vector(
    distribution: DiscretizedDistribution,
    x: torch.Tensor,
) -> torch.Tensor:
    """Current softmax implementation."""
    return distribution.compute_probability_vector(x)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def median_time_per_call(
    function: Callable[[], object],
    device: torch.device,
    warmup: int,
    repeats: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        function()
    synchronize(device)

    samples = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter_ns()
        for _ in range(iterations):
            function()
        synchronize(device)
        elapsed_seconds = (time.perf_counter_ns() - start) / 1e9
        samples.append(elapsed_seconds / iterations)

    return statistics.median(samples)


def benchmark_case(
    rate: float,
    width: int,
    device: torch.device,
    warmup: int,
    repeats: int,
    iterations: int,
    include_backward: bool,
) -> tuple[float, float, float | None, float | None]:
    distribution = make_distribution(rate, device)
    x = torch.arange(
        width,
        dtype=torch.get_default_dtype(),
        device=device,
    ).unsqueeze(1)

    with torch.no_grad():
        old_forward = median_time_per_call(
            lambda: old_probability_vector(distribution, x),
            device,
            warmup,
            repeats,
            iterations,
        )
        new_forward = median_time_per_call(
            lambda: new_probability_vector(distribution, x),
            device,
            warmup,
            repeats,
            iterations,
        )

    if not include_backward:
        return old_forward, new_forward, None, None

    loss_weights = torch.linspace(0.0, 1.0, width, device=device)

    def training_step(
        probability_function: Callable[
            [DiscretizedDistribution, torch.Tensor], torch.Tensor
        ],
    ) -> None:
        probabilities = probability_function(distribution, x)
        loss = (probabilities * loss_weights).sum()
        loss.backward()
        distribution.base_distribution._rate.grad = None

    old_training = median_time_per_call(
        lambda: training_step(old_probability_vector),
        device,
        warmup,
        repeats,
        iterations,
    )
    new_training = median_time_per_call(
        lambda: training_step(new_probability_vector),
        device,
        warmup,
        repeats,
        iterations,
    )
    return old_forward, new_forward, old_training, new_training


def microseconds(seconds: float) -> float:
    return seconds * 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Execution device. Auto prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--cases",
        type=parse_case,
        nargs="+",
        metavar="RATE:WIDTH",
        default=DEFAULT_CASES,
        help="Rate/width cases, such as --cases 0.02:128 0.004:512.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Skip the forward-plus-backward benchmark.",
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0 or args.iterations <= 0:
        parser.error(
            "Warmup cannot be negative; repeats and iterations must be positive."
        )

    device = choose_device(args.device)
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(
        f"Warmup: {args.warmup}; repeats: {args.repeats}; "
        f"iterations: {args.iterations}"
    )
    print()

    header = (
        f"{'rate':>10} {'width':>8} "
        f"{'old fwd (us)':>14} {'new fwd (us)':>14} {'fwd speedup':>12}"
    )
    if not args.forward_only:
        header += (
            f" {'old train (us)':>15} {'new train (us)':>15}"
            f" {'train speedup':>14}"
        )
    print(header)
    print("-" * len(header))

    for rate, width in args.cases:
        old_forward, new_forward, old_training, new_training = benchmark_case(
            rate,
            width,
            device,
            args.warmup,
            args.repeats,
            args.iterations,
            include_backward=not args.forward_only,
        )
        row = (
            f"{rate:>10g} {width:>8,d} "
            f"{microseconds(old_forward):>14.3f} "
            f"{microseconds(new_forward):>14.3f} "
            f"{old_forward / new_forward:>11.2f}x"
        )
        if old_training is not None and new_training is not None:
            row += (
                f" {microseconds(old_training):>15.3f}"
                f" {microseconds(new_training):>15.3f}"
                f" {old_training / new_training:>13.2f}x"
            )
        print(row)


if __name__ == "__main__":
    main()
