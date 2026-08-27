"""Shared cached tensor-product quadrature for custom CUDA integrands."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np
import torch


@lru_cache(maxsize=32)
def _unit_gauss_legendre_rule(
    dimensions: int,
    points: int,
    device_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.legendre.leggauss(points)
    nodes = ((nodes + 1.0) / 2.0).astype(np.float32)
    weights = (weights / 2.0).astype(np.float32)
    coordinate_grid = np.meshgrid(*([nodes] * dimensions), indexing="ij")
    weight_grid = np.meshgrid(*([weights] * dimensions), indexing="ij")
    coordinates = np.stack([value.reshape(-1) for value in coordinate_grid], axis=1)
    combined_weight = np.ones(coordinates.shape[0], dtype=np.float32)
    for value in weight_grid:
        combined_weight *= value.reshape(-1)
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device_index):
        return (
            torch.tensor(coordinates, dtype=torch.float32, device=device),
            torch.tensor(combined_weight, dtype=torch.float32, device=device),
        )


def unit_tensor_gauss_legendre_cuda(
    function: Callable[[torch.Tensor], torch.Tensor],
    reference: torch.Tensor,
    dimensions: int,
    points: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Evaluate fine and half-order unit-cube rules from cached CUDA grids."""

    if not reference.is_cuda or reference.dtype is not torch.float32:
        raise ValueError("custom quadrature requires a CUDA float32 reference")
    if dimensions < 1 or points < 4:
        raise ValueError("invalid tensor Gauss-Legendre rule")
    device_index = reference.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    fine_x, fine_weight = _unit_gauss_legendre_rule(dimensions, points, device_index)
    coarse_points = max(2, points // 2)
    coarse_x, coarse_weight = _unit_gauss_legendre_rule(dimensions, coarse_points, device_index)
    value = torch.dot(fine_weight, function(fine_x))
    coarse_value = torch.dot(coarse_weight, function(coarse_x))
    return value, torch.abs(value - coarse_value), points**dimensions + coarse_points**dimensions


__all__ = ["unit_tensor_gauss_legendre_cuda"]
