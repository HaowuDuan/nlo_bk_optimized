"""Local build and Python boundary for the standalone fixed-Sobol DIS mapper."""

from __future__ import annotations

from pathlib import Path

import torch

from nlo_torch.custom_kernels.extension import load_cuda_extension


def _load_dis_sobol_extension():
    return load_cuda_extension(
        "nlo_torch_dis_sobol_cuda",
        Path(__file__).with_name("kernel.cu"),
    )


def dis_sobol_endpoint_map_cuda(
    points: torch.Tensor,
    edges: torch.Tensor,
    maxr: float,
    *,
    fold_angle: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map frozen importance samples into both exact endpoint sectors."""

    if (
        not points.is_cuda
        or points.dtype is not torch.float32
        or points.ndim != 2
        or points.shape[1] != 5
        or not points.is_contiguous()
    ):
        raise ValueError("DIS Sobol points must be contiguous CUDA float32 [samples, 5]")
    if (
        edges.device != points.device
        or edges.dtype is not torch.float32
        or edges.ndim != 2
        or edges.shape[0] != 5
        or not edges.is_contiguous()
    ):
        raise ValueError("DIS Sobol edges must be contiguous CUDA float32 [5, bins + 1]")
    outputs = _load_dis_sobol_extension().dis_sobol_endpoint_map(points, edges, maxr, fold_angle)
    return outputs[0], outputs[1], outputs[2], outputs[3]


__all__ = ["dis_sobol_endpoint_map_cuda"]
