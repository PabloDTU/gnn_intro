from __future__ import annotations

import torch
from torch_geometric.data import Data


@torch.no_grad()
def apply_edge_dropout(batch: Data, p: float) -> Data:
    """
    Randomly drop a fraction p of edges (and corresponding edge_attr rows) from a PyG Batch/Data.
    No-op if p<=0 or attributes missing. Operates in-place on a shallow copy of tensors.
    """
    if p <= 0 or (not hasattr(batch, "edge_index")) or batch.edge_index is None:
        return batch
    edge_index = batch.edge_index
    E = edge_index.size(1)
    if E == 0:
        return batch
    keep_mask = torch.rand(E, device=edge_index.device) > p
    if keep_mask.all():
        return batch
    batch.edge_index = edge_index[:, keep_mask]
    if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
        batch.edge_attr = batch.edge_attr[keep_mask]
    return batch


@torch.no_grad()
def apply_feature_mask(batch: Data, p: float, on_input: bool = True) -> Data:
    """
    Mask (zero) a random fraction p of feature dimensions across all nodes.
    If on_input=True, applies to batch.x. No-op if p<=0 or x missing.
    """
    if p <= 0 or (not hasattr(batch, "x")) or batch.x is None:
        return batch
    x = batch.x
    if x.numel() == 0:
        return batch
    F = x.size(-1)
    mask = (torch.rand(F, device=x.device) > p).float()
    batch.x = x * mask
    return batch
