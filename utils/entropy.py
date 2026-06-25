import math

import torch


def normalized_entropy_map(probabilities: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return entropy normalized by log(C) for probabilities shaped [B, C, ...]."""
    if probabilities.ndim < 2:
        raise ValueError(
            f"normalized_entropy_map expects at least 2 dimensions [B, C, ...], got shape {tuple(probabilities.shape)}"
        )
    num_classes = probabilities.shape[1]
    if num_classes < 2:
        raise ValueError(f"normalized_entropy_map requires C >= 2, got C={num_classes}")

    log_p = torch.log(probabilities.clamp_min(eps))
    entropy = -(probabilities * log_p).sum(dim=1, keepdim=True)
    max_entropy = math.log(num_classes)
    return (entropy / max_entropy).clamp(0.0, 1.0)
