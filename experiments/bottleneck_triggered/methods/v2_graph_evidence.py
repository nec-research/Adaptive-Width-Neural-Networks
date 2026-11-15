"""
V2-BT: Bottleneck-Triggered Graph-Based Neuron Reordering.

Derives neuron importance scores from the network's structural and relational graph topology
using GRAPHyco primitives and neuron-mapped graph metrics:
    S_j^{V2} = w_struct * p_j^{struct} + w_in * a_j^{in} + w_out * a_j^{out}

Adheres strictly to Section 4: every graph metric has an explicit, valid neuron-level mapping.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from graphyco.core.primitives import Edge, GraphState, Node
from graphyco.core.evaluation import perturbation_profile

from .evidence_base import BaseEvidenceProvider
from ..accounting.resource_ledger import ResourceLedger


class V2GraphEvidence(BaseEvidenceProvider):
    """Computes neuron-level structural and transmission evidence via GRAPHyco."""

    def __init__(
        self,
        edge_weight_threshold: float = 0.05,
        w_struct: float = 0.4,
        w_in: float = 0.3,
        w_out: float = 0.3,
    ):
        super().__init__(name="V2-BT")
        self.edge_weight_threshold = edge_weight_threshold
        self.w_struct = w_struct
        self.w_in = w_in
        self.w_out = w_out

    @staticmethod
    def _node_id(layer_id: int, neuron_id: int) -> str:
        return f"L{layer_id}_N{neuron_id}"

    def compute_neuron_scores(
        self,
        model: nn.Module,
        layer_id: int,
        width: int,
        dataloader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cpu"),
        ledger: Optional[ResourceLedger] = None,
    ) -> Tensor:
        """Derives graph scores for neurons 0..width-1 in layer_id using GRAPHyco."""
        t0 = time.perf_counter()

        # Locate layer parameters
        if hasattr(model, "dyn_model"):
            curr_layer = model.dyn_model.get_layer(layer_id)
            next_layer = model.dyn_model.get_layer(layer_id + 1)
        elif hasattr(model, "fc1") and hasattr(model, "fc2") and layer_id == 0:
            curr_layer = model.fc1
            next_layer = model.fc2
        else:
            raise ValueError(f"Unsupported model architecture: {type(model)}")

        W_in_full = curr_layer.weight.detach().cpu()   # [W_curr, d_in]
        W_out_full = next_layer.weight.detach().cpu() # [d_out, W_curr]

        W = min(width, W_in_full.shape[0])
        d_in = W_in_full.shape[1]
        d_out = W_out_full.shape[0]

        # 1. Build bipartite graph state using GRAPHyco primitives
        nodes: Dict[str, Node] = {}
        edges: List[Edge] = []

        # Previous stage nodes (inputs)
        for i in range(d_in):
            nid = self._node_id(layer_id - 1, i)
            nodes[nid] = Node(id=nid, name=nid, purpose="input", active=True)

        # Target hidden stage nodes (layer_id)
        for j in range(W):
            nid = self._node_id(layer_id, j)
            nodes[nid] = Node(id=nid, name=nid, purpose="hidden", active=True)

        # Succeeding stage nodes (outputs)
        for k in range(d_out):
            nid = self._node_id(layer_id + 1, k)
            nodes[nid] = Node(id=nid, name=nid, purpose="output", active=True)

        # Incoming edges: layer_id - 1 -> layer_id
        w_in_slice = W_in_full[:W, :].abs()
        max_in = w_in_slice.max().item() if w_in_slice.numel() > 0 else 1.0
        cutoff_in = self.edge_weight_threshold * (max_in if max_in > 0 else 1.0)

        for j in range(W):
            tgt_id = self._node_id(layer_id, j)
            for i in range(d_in):
                if w_in_slice[j, i].item() >= cutoff_in:
                    edges.append(Edge(source=self._node_id(layer_id - 1, i), target=tgt_id))

        # Outgoing edges: layer_id -> layer_id + 1
        w_out_slice = W_out_full[:, :W].abs()
        max_out = w_out_slice.max().item() if w_out_slice.numel() > 0 else 1.0
        cutoff_out = self.edge_weight_threshold * (max_out if max_out > 0 else 1.0)

        for k in range(d_out):
            tgt_id = self._node_id(layer_id + 1, k)
            for j in range(W):
                if w_out_slice[k, j].item() >= cutoff_out:
                    edges.append(Edge(source=self._node_id(layer_id, j), target=tgt_id))

        # 2. Extract GRAPHyco topological features
        graph_state = GraphState(nodes=nodes, edges=edges)
        try:
            profile = perturbation_profile(graph_state)
        except Exception:
            # Fallback if graph is empty or completely disconnected
            profile = {self._node_id(layer_id, j): 1.0 for j in range(W)}

        # 3. Derive per-neuron relational features
        scores_list: List[float] = []

        for j in range(W):
            nid = self._node_id(layer_id, j)

            # A. Structural perturbation impact
            p_struct = float(profile.get(nid, 1.0))

            # B. Weighted incoming transmission relationship
            row_weights = w_in_slice[j, :]
            a_in = float(row_weights.mean().item()) if row_weights.numel() > 0 else 0.0

            # C. Weighted outgoing transmission relationship
            col_weights = w_out_slice[:, j]
            a_out = float(col_weights.mean().item()) if col_weights.numel() > 0 else 0.0

            # Combined graph evidence score
            score_j = (
                self.w_struct * p_struct
                + self.w_in * a_in
                + self.w_out * a_out
            )
            scores_list.append(score_j)

        scores_tensor = torch.tensor(scores_list, device=device, dtype=torch.float32)

        # Handle non-finite values safely
        scores_tensor = torch.nan_to_num(scores_tensor, nan=0.0, posinf=1e6, neginf=0.0)

        duration = time.perf_counter() - t0
        if ledger is not None:
            # Graph extraction proxy compute
            est_flops = float(len(nodes) * 10 + len(edges) * 20)
            ledger.record_graph_extraction(duration_sec=duration, estimated_flops=est_flops)

        return scores_tensor
