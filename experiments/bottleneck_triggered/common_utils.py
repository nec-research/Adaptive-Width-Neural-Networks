"""
Common utilities for Bottleneck-Triggered AWN Experiments.

Provides standard dataset split loading and configuration helpers.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import DoubleMoon, Spiral, SpiralHard


def load_dataset_splits(
    dataset_name: str, batch_size: int = 128
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Loads standardized dataset splits for DoubleMoon, Spiral, and SpiralHard."""
    storage_folder = "DATA"
    if dataset_name == "DoubleMoon":
        ds = DoubleMoon(storage_folder=storage_folder)
        split_path = "DATA_SPLITS/DoubleMoon/DoubleMoon_outer1_inner1.splits"
        bs = 32
    elif dataset_name == "Spiral":
        ds = Spiral(storage_folder=storage_folder)
        split_path = "DATA_SPLITS/Spiral/Spiral_outer1_inner1.splits"
        bs = 128
    elif dataset_name == "SpiralHard":
        ds = SpiralHard(storage_folder=storage_folder)
        split_path = "DATA_SPLITS/SpiralHard/SpiralHard_outer1_inner1.splits"
        bs = 128
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    with open(split_path, "rb") as f:
        splits = pickle.load(f)

    outer = splits["outer_folds"][0]
    train_idx = outer["train"]
    val_idx = outer.get("val", [])
    test_idx = outer["test"]

    train_set = Subset(ds, train_idx)
    test_set = Subset(ds, test_idx)
    val_set = Subset(ds, val_idx) if len(val_idx) > 0 else test_set

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    return train_loader, val_loader, test_loader, len(train_set)


class FixedMLP(nn.Module):
    """Standard Fixed-Width MLP matching the paper's tuned baseline."""

    def __init__(
        self, in_features: int = 2, hidden_features: int = 32, out_features: int = 2
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.activation = nn.ReLU6()

    def forward(
        self, x: torch.Tensor, subset_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.fc1(x)
        h = self.activation(h)
        if subset_mask is not None:
            h = h * subset_mask
        return self.fc2(h)


def get_awn_config(
    n_obs: int, enable_learned_ordering: bool = False
) -> Dict[str, Any]:
    """Generates standard AWN model configuration dictionary."""
    return {
        "num_hidden_layers": 1,
        "num_hidden_neurons": 256,
        "share_width_distribution": False,
        "disable_adaptation": False,
        "n_observations": n_obs,
        "quantile": 0.9,
        "theta_prior_scale": 1.0,
        "alpha_prior_mean": 0.01,
        "alpha_prior_scale": 0.01,
        "dynamic_architecture": {
            "class_name": "dynamic_architecture.DynamicMLP",
        },
        "truncated_distribution": {
            "class_name": "distribution.TruncatedDistribution",
            "args": {
                "discretized_distribution": {
                    "class_name": "distribution.DiscretizedDistribution",
                    "args": {
                        "base_distribution": {
                            "class_name": "distribution.Exponential",
                            "args": {
                                "rate": 0.01,
                                "boundary": 0.0005,
                            },
                        }
                    },
                }
            },
        },
        "enable_learned_ordering": enable_learned_ordering,
        "ranking_loss_weight": 0.01,
        "loss": {
            "class_name": "metric.ELBO_Classification",
            "args": {"n_observations": n_obs},
        },
    }
