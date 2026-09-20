"""Tile extraction and aligned RGB/prior input construction.

The caller must supply already aligned arrays and training-set statistics.
The pyramid preparation and inference pipeline are not included.
"""

import numpy as np
import torch


def tile(array: np.ndarray, row: int, col: int, size: int, fill=None) -> np.ndarray:
    patch = np.array(array[..., row : row + size, col : col + size], copy=True)
    pads = [(0, 0)] * (patch.ndim - 2) + [
        (0, size - patch.shape[-2]),
        (0, size - patch.shape[-1]),
    ]
    return (
        np.pad(patch, pads, mode="edge")
        if fill is None
        else np.pad(patch, pads, constant_values=fill)
    )


def inputs(rgb, prior, row: int, col: int, config: dict, stats: dict) -> torch.Tensor:
    x = tile(rgb, row, col, config["tile_size"]).astype(np.float32)
    x = (x - np.array(stats["mean"], np.float32)[:, None, None]) / np.array(
        stats["std"], np.float32
    )[:, None, None]
    if prior is not None:
        p = tile(prior, row, col, config["tile_size"]).astype(np.float32) / (
            config["classes"] - 1
        )
        x = np.concatenate((x, p[None]), axis=0)
    return torch.from_numpy(x)
