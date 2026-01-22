import copy
import random
import numpy as np
import torch
from typing import List, Dict, Optional


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility (FL is stochastic!)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# State dict helpers
# =========================================================
def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Deep copy a model state_dict (detach from graph, clone tensors)
    """
    return {k: v.clone().detach() for k, v in state_dict.items()}


def move_state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Move all tensors in state_dict to CPU
    (recommended before aggregation to save GPU memory)
    """
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def load_state_dict_to_model(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]):
    """
    Load state_dict into model (safe wrapper)
    """
    model.load_state_dict(state_dict, strict=True)


# =========================================================
# [NEW] Vector & Math Utils for ADAPT Signals
# =========================================================
def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Flatten a state_dict into a single 1D tensor vector.
    Used for computing cosine similarity or norms.
    """
    # Sort keys to ensure deterministic order (essential for comparison)
    keys = sorted(state_dict.keys())
    # Flatten and concatenate all parameters
    tensors = [state_dict[k].reshape(-1).float().cpu() for k in keys]
    return torch.cat(tensors)


def compute_state_dict_diff(
    target: Dict[str, torch.Tensor],
    source: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Compute delta = target - source.
    Useful for obtaining update vectors: \Delta \theta = \theta_{new} - \theta_{old}
    """
    return {k: target[k] - source[k] for k in target.keys()}


def compute_norm(state_dict: Dict[str, torch.Tensor]) -> float:
    """
    Compute L2 norm of a state_dict (treated as a vector).
    """
    flat = flatten_state_dict(state_dict)
    return float(torch.norm(flat).item())


def compute_cosine_similarity(
    state1: Dict[str, torch.Tensor],
    state2: Dict[str, torch.Tensor],
    eps: float = 1e-8
) -> float:
    """
    Compute Cosine Similarity between two state dicts (interpreted as vectors).
    Formula: (A . B) / (|A| * |B|)
    
    Used for calculating Alignment Signal:
       Align_i = CosSim(ClientUpdate_i, GlobalUpdate)
    """
    v1 = flatten_state_dict(state1)
    v2 = flatten_state_dict(state2)

    # Ensure on same device (CPU recommended for metric calc)
    if v1.device != v2.device:
        v2 = v2.to(v1.device)

    dot = torch.dot(v1, v2)
    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)

    if norm1 < eps or norm2 < eps:
        return 0.0

    return float((dot / (norm1 * norm2)).item())


# =========================================================
# FedAvg aggregation
# =========================================================
def fedavg(
    client_states: List[Dict[str, torch.Tensor]],
    client_weights: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Federated Averaging (FedAvg)

    Args:
        client_states: list of state_dict from clients
        client_weights: list of weights (e.g., num_samples per client)
                        if None -> uniform average

    Returns:
        aggregated_state_dict
    """
    assert len(client_states) > 0, "No client states to aggregate"

    num_clients = len(client_states)

    if client_weights is None:
        client_weights = [1.0] * num_clients
    else:
        client_weights = [float(w) for w in client_weights]

    total_weight = sum(client_weights)
    assert total_weight > 0, "Total client weight must be > 0"

    # Initialize aggregated state with zeros (using first client structure)
    # Ensure we create new tensors to avoid modifying inputs in place
    first_state = client_states[0]
    agg_state = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in first_state.items()}

    # Weighted sum
    for state, weight in zip(client_states, client_weights):
        normalized_weight = weight / total_weight
        for key in agg_state.keys():
            agg_state[key] += state[key].float() * normalized_weight

    return agg_state


# =========================================================
# Client sampling
# =========================================================
def sample_clients(
    all_users: List[int],
    clients_per_round: Optional[int] = None,
    client_fraction: Optional[float] = None,
) -> List[int]:
    """
    Randomly sample clients (users) for one FL round

    Either:
      - specify clients_per_round
      - or specify client_fraction (0 < fraction <= 1)

    Returns:
        list of user ids
    """
    assert (clients_per_round is not None) or (client_fraction is not None), \
        "Specify clients_per_round or client_fraction"

    num_users = len(all_users)

    if client_fraction is not None:
        assert 0 < client_fraction <= 1.0
        clients_per_round = max(1, int(num_users * client_fraction))

    clients_per_round = min(clients_per_round, num_users)

    return random.sample(all_users, clients_per_round)


# =========================================================
# Utility: count trainable parameters (optional debug)
# =========================================================
def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)