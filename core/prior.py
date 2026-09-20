"""Convert coarse class predictions to an aligned finer-scale prior."""
import torch
from torch.nn import functional as F


def resize_prior(logits: torch.Tensor, size: tuple) -> torch.Tensor:
    """Nearest-neighbor resize of normalized hard labels over the same extent.

    Input grids must cover the same geographical extent. This operation alone
    does not establish georeferencing or align unrelated image tiles.
    """
    if logits.ndim != 4 or logits.shape[1] < 2:
        raise ValueError("Expected logits with shape (B, classes >= 2, H, W)")
    labels = logits.detach().argmax(dim=1, keepdim=True).float()
    return F.interpolate(labels, size=size, mode="nearest") / (logits.shape[1] - 1)
