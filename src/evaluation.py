import torch
import numpy as np
from typing import Dict


def hit_ratio_at_k(ranks: np.ndarray, k: int) -> float:
    """
    Hit Ratio @ K
    ranks: array of rank position of positive item (0-based)
    """
    return np.mean(ranks < k)


def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    """
    NDCG @ K
    ranks: array of rank position of positive item (0-based)
    """
    ndcg = 0.0
    for r in ranks:
        if r < k:
            ndcg += 1.0 / np.log2(r + 2)
    return ndcg / len(ranks)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    k: int = 20,
) -> Dict[str, float]:
    """
    Evaluate model with HR@k and NDCG@k

    dataloader yields:
      {
        "input_seq": (B, L),
        "candidates": (B, C),
        "labels": (B, C)  # not strictly needed
      }
    """

    model.eval()

    all_ranks = []

    for batch in dataloader:
        input_seq = batch["input_seq"].to(device)       # (B, L)
        candidates = batch["candidates"].to(device)     # (B, C)

        # logits: higher = more relevant
        logits = model.predict(input_seq, candidates)   # (B, C)

        if not torch.isfinite(logits).all():
            raise ValueError("NaN/Inf logits detected during evaluation")

        # sort descending
        _, indices = torch.sort(logits, dim=1, descending=True)

        # positive item is always at index 0 in candidates
        pos_index = torch.zeros(
            (indices.size(0),), dtype=torch.long, device=indices.device
        )

        # rank position of positive item
        ranks = (indices == pos_index.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
        all_ranks.append(ranks.cpu().numpy())

    all_ranks = np.concatenate(all_ranks, axis=0)

    hr = hit_ratio_at_k(all_ranks, k)
    ndcg = ndcg_at_k(all_ranks, k)

    return {
        f"HR@{k}": hr,
        f"NDCG@{k}": ndcg
    }
