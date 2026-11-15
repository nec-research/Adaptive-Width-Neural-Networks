"""
Abstract Base Contract for Evidence Providers.

Every evidence provider derives neuron-level importance scores from model state
or behavioral dynamics to inform the permutation manager.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from ..accounting.resource_ledger import ResourceLedger


class BaseEvidenceProvider(ABC):
    """Abstract base class for V0-BT, V1-BT, and V2-BT evidence providers."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute_neuron_scores(
        self,
        model: nn.Module,
        layer_id: int,
        width: int,
        dataloader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu"),
        ledger: Optional[ResourceLedger] = None,
    ) -> Tensor:
        """Compute 1D tensor of neuron importance scores for layer `layer_id` of length `width`.

        Higher scores must indicate higher priority for retained prefix.
        Must handle non-finite or zero values gracefully.
        """
        raise NotImplementedError
